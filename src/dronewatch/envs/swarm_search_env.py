"""RLlib MultiAgentEnv wrapper around the Rust swarm simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from dronewatch.config.schema import SwarmSearchEnvConfig
from dronewatch.sim import SwarmSimulation

from .observation_builder import ObservationBuilder
from .reward import calculate_team_reward
from .spaces import (
    action_space,
    agent_ids,
    observation_space,
)


class SwarmSearchEnv(MultiAgentEnv):
    """RLlib-compatible multi-agent wrapper for the Rust swarm simulator."""

    metadata = {"name": "SwarmSearch2D"}

    def __init__(self, env_config: SwarmSearchEnvConfig | Mapping[str, Any] | None = None) -> None:
        """Create an environment and validate RLlib configuration with Pydantic.

        Args:
            env_config: Optional `SwarmSearchEnvConfig` or RLlib-provided dictionary.

        Raises:
            pydantic.ValidationError: If the config contains unsupported keys or invalid values.
        """
        super().__init__()
        self._config = (
            env_config
            if isinstance(env_config, SwarmSearchEnvConfig)
            else SwarmSearchEnvConfig.model_validate(env_config or {})
        )
        self._env_config = self._config.env
        self._initial_seed: int | None = self._config.seed
        self._simulation = SwarmSimulation(seed=self._initial_seed, config=self._env_config.to_rust_config_dict())
        self._observation_builder = ObservationBuilder(self._env_config)
        self._agent_ids = agent_ids(self._env_config.num_agents)
        self.possible_agents = list(self._agent_ids)
        self.agents = list(self._agent_ids)
        self.action_space = action_space()
        self.observation_space = observation_space(self._env_config)
        self.action_spaces = {agent_id: self.action_space for agent_id in self._agent_ids}
        self.observation_spaces = {agent_id: self.observation_space for agent_id in self._agent_ids}
        self._episode_reward = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the Rust simulator and return RLlib observations plus infos.

        Args:
            seed: Optional reset seed supplied by RLlib. When omitted, the validated environment
                config seed is used. A value of `None` delegates seed resolution to Rust.
            options: Reserved for Gymnasium/RLlib compatibility and ignored in Phase 2.
        """
        del options
        reset_seed = self._initial_seed if seed is None else seed
        metrics = self._simulation.reset(seed=reset_seed)
        state = self._simulation.state()
        self.agents = list(self._agent_ids)
        self._episode_reward = 0.0
        observations = self._observation_builder.build(state, metrics)
        infos = {agent_id: {"metrics": dict(metrics)} for agent_id in self._agent_ids}
        return observations, infos

    def step(
        self,
        action_dict: Mapping[str, Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance one simulator step from a dictionary of per-agent actions.

        RLlib supplies actions keyed by agent ID, while Rust expects a positionally ordered action
        list. This method validates the mapping, calls Rust, and returns the standard RLlib
        multi-agent payload.
        """
        actions = self._ordered_actions(action_dict)
        result = self._simulation.step(actions)
        events = dict(result["events"])
        metrics = dict(result["metrics"])
        state = result["state"]

        observations = self._observation_builder.build(state, metrics)
        team_reward = calculate_team_reward(events, self._env_config.reward)
        self._episode_reward += team_reward

        terminated = bool(metrics.get("all_targets_discovered", False))
        truncated = bool(metrics.get("horizon_reached", False)) and not terminated
        rewards = {agent_id: team_reward for agent_id in self._agent_ids}
        terminateds = {agent_id: terminated for agent_id in self._agent_ids}
        truncateds = {agent_id: truncated for agent_id in self._agent_ids}
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        if terminated or truncated:
            self.agents = []
        else:
            self.agents = list(self._agent_ids)

        infos = {
            agent_id: {
                "events": events,
                "metrics": metrics,
                "episode_reward": self._episode_reward,
                "episode_length": metrics["timestep"],
            }
            for agent_id in self._agent_ids
        }
        return observations, rewards, terminateds, truncateds, infos

    def snapshot(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the current simulator state and metrics for rendering and evaluation helpers."""
        return self._simulation.state(), self._simulation.metrics()

    def _ordered_actions(self, action_dict: Mapping[str, Any]) -> list[tuple[float, float]]:
        """Validate action values and order them by stable simulator agent ID."""
        missing = [agent_id for agent_id in self._agent_ids if agent_id not in action_dict]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"missing actions for active agents: {joined}")

        ordered_actions: list[tuple[float, float]] = []
        for agent_id in self._agent_ids:
            action = np.asarray(action_dict[agent_id], dtype=np.float64)
            if action.shape != (2,):
                raise ValueError(f"action for {agent_id} must have shape (2,), got {action.shape}")
            if not np.isfinite(action).all():
                raise ValueError(f"action for {agent_id} contains non-finite values")
            ordered_actions.append((float(action[0]), float(action[1])))
        return ordered_actions
