"""Fixed-size local observation construction for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .spaces import (
    AGENT_DEFAULTS,
    OBSERVATION_DEFAULTS,
    OBSERVATION_SIZE,
    OBSTACLE_DEFAULTS,
    WORLD_DEFAULTS,
    agent_ids,
)


class ObservationBuilder:
    """Build local, padded, fixed-size observations from Rust simulator state."""

    observation_size = OBSERVATION_SIZE
    _relative_velocity_scale = AGENT_DEFAULTS.max_speed * 2.0
    _obstacle_visibility_scale = AGENT_DEFAULTS.sensing_radius + OBSTACLE_DEFAULTS.expected_max_radius

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
        horizon = max(float(metrics.get("max_episode_steps", WORLD_DEFAULTS.max_episode_steps)), 1.0)

        values: list[float] = [
            position[0] / WORLD_DEFAULTS.width,
            position[1] / WORLD_DEFAULTS.height,
            velocity[0] / AGENT_DEFAULTS.max_speed,
            velocity[1] / AGENT_DEFAULTS.max_speed,
            timestep / horizon,
        ]
        values.extend(self._visible_agents(agent, agents, position, velocity))
        values.extend(self._visible_targets(targets, position))
        values.extend(self._visible_obstacles(obstacles, position))
        values.extend(self._communication_summary(agent, agents, position, velocity))

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"expected observation shape {(OBSERVATION_SIZE,)}, got {observation.shape}")
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
            if distance <= AGENT_DEFAULTS.sensing_radius:
                visible.append((distance, other))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, other in visible[: OBSERVATION_DEFAULTS.max_visible_agents]:
            relative_position = _array2(other["position"]) - position
            relative_velocity = _array2(other["velocity"]) - velocity
            values.extend(
                [
                    relative_position[0] / AGENT_DEFAULTS.sensing_radius,
                    relative_position[1] / AGENT_DEFAULTS.sensing_radius,
                    relative_velocity[0] / self._relative_velocity_scale,
                    relative_velocity[1] / self._relative_velocity_scale,
                    distance / AGENT_DEFAULTS.sensing_radius,
                    1.0,
                ]
            )

        missing = OBSERVATION_DEFAULTS.max_visible_agents - min(len(visible), OBSERVATION_DEFAULTS.max_visible_agents)
        values.extend([0.0] * missing * 6)
        return values

    def _visible_targets(self, targets: list[Mapping[str, Any]], position: np.ndarray) -> list[float]:
        """Encode nearest visible targets with discovery flags and zero padding."""
        visible: list[tuple[float, Mapping[str, Any]]] = []
        for target in targets:
            distance = float(np.linalg.norm(_array2(target["position"]) - position))
            if distance <= AGENT_DEFAULTS.sensing_radius:
                visible.append((distance, target))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, target in visible[: OBSERVATION_DEFAULTS.max_visible_targets]:
            relative_position = _array2(target["position"]) - position
            values.extend(
                [
                    relative_position[0] / AGENT_DEFAULTS.sensing_radius,
                    relative_position[1] / AGENT_DEFAULTS.sensing_radius,
                    distance / AGENT_DEFAULTS.sensing_radius,
                    1.0 if bool(target["discovered"]) else 0.0,
                    1.0,
                ]
            )

        missing = OBSERVATION_DEFAULTS.max_visible_targets - min(len(visible), OBSERVATION_DEFAULTS.max_visible_targets)
        values.extend([0.0] * missing * 5)
        return values

    def _visible_obstacles(self, obstacles: list[Mapping[str, Any]], position: np.ndarray) -> list[float]:
        """Encode nearest visible circular obstacles with radius normalization."""
        visible: list[tuple[float, Mapping[str, Any]]] = []
        for obstacle in obstacles:
            center_distance = float(np.linalg.norm(_array2(obstacle["position"]) - position))
            radius = float(obstacle["radius"])
            if center_distance <= AGENT_DEFAULTS.sensing_radius + radius:
                visible.append((center_distance, obstacle))
        visible.sort(key=lambda item: (item[0], int(item[1]["id"])))

        values: list[float] = []
        for distance, obstacle in visible[: OBSERVATION_DEFAULTS.max_visible_obstacles]:
            relative_position = _array2(obstacle["position"]) - position
            radius = float(obstacle["radius"])
            values.extend(
                [
                    relative_position[0] / self._obstacle_visibility_scale,
                    relative_position[1] / self._obstacle_visibility_scale,
                    radius / OBSTACLE_DEFAULTS.expected_max_radius,
                    distance / self._obstacle_visibility_scale,
                    1.0,
                ]
            )

        missing = OBSERVATION_DEFAULTS.max_visible_obstacles - min(
            len(visible), OBSERVATION_DEFAULTS.max_visible_obstacles
        )
        values.extend([0.0] * missing * 5)
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
            if distance <= AGENT_DEFAULTS.communication_radius:
                neighbors.append(other)

        if not neighbors:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        relative_positions = np.asarray([_array2(other["position"]) - position for other in neighbors])
        relative_velocities = np.asarray([_array2(other["velocity"]) - velocity for other in neighbors])
        mean_relative_position = relative_positions.mean(axis=0)
        mean_relative_velocity = relative_velocities.mean(axis=0)
        return [
            len(neighbors) / max(AGENT_DEFAULTS.count - 1, 1),
            mean_relative_position[0] / AGENT_DEFAULTS.communication_radius,
            mean_relative_position[1] / AGENT_DEFAULTS.communication_radius,
            mean_relative_velocity[0] / self._relative_velocity_scale,
            mean_relative_velocity[1] / self._relative_velocity_scale,
        ]


def _array2(value: Any) -> np.ndarray:
    """Convert a value to a finite two-element NumPy vector shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"expected 2D vector, got shape {array.shape}")
    return array
