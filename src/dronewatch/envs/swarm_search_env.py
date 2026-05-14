"""RLlib MultiAgentEnv wrapper around the Rust swarm simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from dronewatch.sim import SwarmSimulation

from .observation_builder import ObservationBuilder
from .reward import calculate_team_reward
from .spaces import (
    AGENT_DEFAULTS,
    action_space,
    agent_ids,
    observation_space,
    validate_env_config,
)


class SwarmSearchEnv(MultiAgentEnv):
    """RLlib-compatible multi-agent wrapper for the Phase 1 Rust world."""

    metadata = {"name": "SwarmSearch2D"}

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        config = validate_env_config(env_config)
        self._initial_seed = config.get("seed")
        self._simulation = SwarmSimulation(seed=self._initial_seed)
        self._observation_builder = ObservationBuilder()
        self._agent_ids = agent_ids(AGENT_DEFAULTS.count)
        self.possible_agents = list(self._agent_ids)
        self.agents = list(self._agent_ids)
        self.action_space = action_space()
        self.observation_space = observation_space()
        self.action_spaces = {agent_id: self.action_space for agent_id in self._agent_ids}
        self.observation_spaces = {agent_id: self.observation_space for agent_id in self._agent_ids}
        self._episode_reward = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
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
        actions = self._ordered_actions(action_dict)
        result = self._simulation.step(actions)
        events = dict(result["events"])
        metrics = dict(result["metrics"])
        state = result["state"]

        observations = self._observation_builder.build(state, metrics)
        team_reward = calculate_team_reward(events)
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

    def _ordered_actions(self, action_dict: Mapping[str, Any]) -> list[tuple[float, float]]:
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
