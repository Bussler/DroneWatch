"""Fixed-size local observation construction for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dronewatch.config.schema import EnvConfig

from .spaces import (
    VISIBLE_AGENT_FEATURES,
    VISIBLE_OBSTACLE_FEATURES,
    VISIBLE_TARGET_FEATURES,
    agent_ids,
    observation_size,
)


class ObservationBuilder:
    """Build local, padded, fixed-size observations from Rust simulator state."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Create an observation builder for a configured environment."""
        self._config = config or EnvConfig()
        self.observation_size = observation_size(self._config)
        self._relative_velocity_scale = self._config.agents.max_speed * 2.0
        self._obstacle_visibility_scale = self._config.agents.sensing_radius + self._config.obstacles.max_radius

    def build(
        self,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Build one fixed-size local observation vector for every simulator agent."""
        agents = list(state["agents"])
        targets = list(state["targets"])
        obstacles = list(state["obstacles"])
        observations: dict[str, np.ndarray] = {}

        for agent in agents:
            observations[f"agent_{int(agent['id'])}"] = self._build_agent_observation(
                agent=agent,
                agents=agents,
                targets=targets,
                obstacles=obstacles,
                metrics=metrics,
            )

        return {agent_id: observations[agent_id] for agent_id in agent_ids(len(agents))}

    def _build_agent_observation(
        self,
        agent: Mapping[str, Any],
        agents: list[Mapping[str, Any]],
        targets: list[Mapping[str, Any]],
        obstacles: list[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> np.ndarray:
        """Build and validate the complete observation vector for one agent."""
        position = _array2(agent["position"])
        velocity = _array2(agent["velocity"])
        timestep = float(metrics.get("timestep", 0.0))
        horizon = max(float(metrics.get("max_episode_steps", self._config.max_episode_steps)), 1.0)

        values: list[float] = [
            position[0] / self._config.world.width,
            position[1] / self._config.world.height,
            velocity[0] / self._config.agents.max_speed,
            velocity[1] / self._config.agents.max_speed,
            timestep / horizon,
        ]
        values.extend(self._visible_agents(agent, agents, position, velocity))
        values.extend(self._visible_targets(targets, position))
        values.extend(self._visible_obstacles(obstacles, position))
        if self._config.observation.include_communication_summary:
            values.extend(self._communication_summary(agent, agents, position, velocity))

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(f"expected observation shape {(self.observation_size,)}, got {observation.shape}")
        if not np.isfinite(observation).all():
            raise ValueError("observation contains non-finite values")
        return observation

    def _visible_agents(
        self,
        agent: Mapping[str, Any],
        agents: list[Mapping[str, Any]],
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> list[float]:
        """Encode nearest visible neighboring agents with zero padding."""
        agent_id = int(agent["id"])
        visible: list[tuple[float, Mapping[str, Any]]] = []
        for other in agents:
            if int(other["id"]) == agent_id:
                continue
            distance = float(np.linalg.norm(_array2(other["position"]) - position))
            if distance <= self._config.agents.sensing_radius:
                visible.append((distance, other))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, other in visible[: self._config.observation.max_visible_agents]:
            relative_position = _array2(other["position"]) - position
            relative_velocity = _array2(other["velocity"]) - velocity
            agent_features = [
                relative_position[0] / self._config.agents.sensing_radius,
                relative_position[1] / self._config.agents.sensing_radius,
                relative_velocity[0] / self._relative_velocity_scale,
                relative_velocity[1] / self._relative_velocity_scale,
                distance / self._config.agents.sensing_radius,
                1.0,
            ]
            if len(agent_features) != VISIBLE_AGENT_FEATURES:
                raise ValueError(f"expected {VISIBLE_AGENT_FEATURES} visible-agent features, got {len(agent_features)}")
            values.extend(agent_features)

        missing = self._config.observation.max_visible_agents - min(
            len(visible), self._config.observation.max_visible_agents
        )
        values.extend([0.0] * missing * VISIBLE_AGENT_FEATURES)
        return values

    def _visible_targets(self, targets: list[Mapping[str, Any]], position: np.ndarray) -> list[float]:
        """Encode nearest visible targets with discovery flags and zero padding."""
        visible: list[tuple[float, Mapping[str, Any]]] = []
        for target in targets:
            distance = float(np.linalg.norm(_array2(target["position"]) - position))
            if distance <= self._config.agents.sensing_radius:
                visible.append((distance, target))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, target in visible[: self._config.observation.max_visible_targets]:
            relative_position = _array2(target["position"]) - position
            target_features = [
                relative_position[0] / self._config.agents.sensing_radius,
                relative_position[1] / self._config.agents.sensing_radius,
                distance / self._config.agents.sensing_radius,
                1.0 if bool(target["discovered"]) else 0.0,
                1.0,
            ]
            if len(target_features) != VISIBLE_TARGET_FEATURES:
                raise ValueError(
                    f"expected {VISIBLE_TARGET_FEATURES} visible-target features, got {len(target_features)}"
                )
            values.extend(target_features)

        missing = self._config.observation.max_visible_targets - min(
            len(visible), self._config.observation.max_visible_targets
        )
        values.extend([0.0] * missing * VISIBLE_TARGET_FEATURES)
        return values

    def _visible_obstacles(self, obstacles: list[Mapping[str, Any]], position: np.ndarray) -> list[float]:
        """Encode nearest visible circular obstacles with radius normalization."""
        visible: list[tuple[float, Mapping[str, Any]]] = []
        for obstacle in obstacles:
            center_distance = float(np.linalg.norm(_array2(obstacle["position"]) - position))
            radius = float(obstacle["radius"])
            if center_distance <= self._config.agents.sensing_radius + radius:
                visible.append((center_distance, obstacle))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, obstacle in visible[: self._config.observation.max_visible_obstacles]:
            relative_position = _array2(obstacle["position"]) - position
            radius = float(obstacle["radius"])
            obstacle_features = [
                relative_position[0] / self._obstacle_visibility_scale,
                relative_position[1] / self._obstacle_visibility_scale,
                radius / self._config.obstacles.max_radius,
                distance / self._obstacle_visibility_scale,
                1.0,
            ]
            if len(obstacle_features) != VISIBLE_OBSTACLE_FEATURES:
                raise ValueError(
                    f"expected {VISIBLE_OBSTACLE_FEATURES} visible-obstacle features, got {len(obstacle_features)}"
                )
            values.extend(obstacle_features)

        missing = self._config.observation.max_visible_obstacles - min(
            len(visible), self._config.observation.max_visible_obstacles
        )
        values.extend([0.0] * missing * VISIBLE_OBSTACLE_FEATURES)
        return values

    def _communication_summary(
        self,
        agent: Mapping[str, Any],
        agents: list[Mapping[str, Any]],
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> list[float]:
        """Summarize communication-neighbor count and mean relative state."""
        agent_id = int(agent["id"])
        neighbors: list[Mapping[str, Any]] = []
        for other in agents:
            if int(other["id"]) == agent_id:
                continue
            distance = float(np.linalg.norm(_array2(other["position"]) - position))
            if distance <= self._config.agents.communication_radius:
                neighbors.append(other)

        if not neighbors:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        relative_positions = np.asarray([_array2(other["position"]) - position for other in neighbors])
        relative_velocities = np.asarray([_array2(other["velocity"]) - velocity for other in neighbors])
        mean_relative_position = relative_positions.mean(axis=0)
        mean_relative_velocity = relative_velocities.mean(axis=0)
        return [
            len(neighbors) / max(self._config.num_agents - 1, 1),
            mean_relative_position[0] / self._config.agents.communication_radius,
            mean_relative_position[1] / self._config.agents.communication_radius,
            mean_relative_velocity[0] / self._relative_velocity_scale,
            mean_relative_velocity[1] / self._relative_velocity_scale,
        ]


def _array2(value: Any) -> np.ndarray:
    """Convert a value to a finite two-element NumPy vector shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"expected 2D vector, got shape {array.shape}")
    return array
