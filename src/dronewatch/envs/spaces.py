"""Spaces and defaults for the Phase 2 SwarmSearch2D wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class WorldDefaults:
    width: float = 100.0
    height: float = 100.0
    max_episode_steps: int = 200


@dataclass(frozen=True)
class AgentDefaults:
    count: int = 16
    max_speed: float = 2.0
    sensing_radius: float = 15.0
    communication_radius: float = 20.0


@dataclass(frozen=True)
class ObstacleDefaults:
    expected_max_radius: float = 6.0


@dataclass(frozen=True)
class ObservationDefaults:
    max_visible_agents: int = 5
    max_visible_targets: int = 5
    max_visible_obstacles: int = 5


@dataclass(frozen=True)
class RewardWeights:
    target_discovered: float = 5.0
    new_coverage_cell: float = 0.02
    agent_collision: float = -0.25
    obstacle_collision: float = -0.5
    step_penalty: float = -0.001


WORLD_DEFAULTS = WorldDefaults()
AGENT_DEFAULTS = AgentDefaults()
OBSTACLE_DEFAULTS = ObstacleDefaults()
OBSERVATION_DEFAULTS = ObservationDefaults()
REWARD_WEIGHTS = RewardWeights()

OWN_STATE_SIZE = 5
VISIBLE_AGENT_FEATURES = 6
VISIBLE_TARGET_FEATURES = 5
VISIBLE_OBSTACLE_FEATURES = 5
COMMUNICATION_SUMMARY_SIZE = 5

OBSERVATION_SIZE = (
    OWN_STATE_SIZE
    + OBSERVATION_DEFAULTS.max_visible_agents * VISIBLE_AGENT_FEATURES
    + OBSERVATION_DEFAULTS.max_visible_targets * VISIBLE_TARGET_FEATURES
    + OBSERVATION_DEFAULTS.max_visible_obstacles * VISIBLE_OBSTACLE_FEATURES
    + COMMUNICATION_SUMMARY_SIZE
)


def agent_ids(num_agents: int = AGENT_DEFAULTS.count) -> list[str]:
    """Return stable RLlib agent IDs for the homogeneous swarm."""
    return [f"agent_{index}" for index in range(num_agents)]


def action_space() -> gym.spaces.Box:
    """Return the per-agent action space accepted by the Rust simulator."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def observation_space() -> gym.spaces.Box:
    """Return the per-agent fixed-size observation space."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBSERVATION_SIZE,),
        dtype=np.float32,
    )


def validate_env_config(env_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate Phase 2 env config values against the fixed Rust defaults."""
    config = dict(env_config or {})
    unsupported = sorted(key for key in config if key not in {"seed"})
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            f"unsupported Phase 2 env_config keys: {joined}; only 'seed' is supported until Rust config bindings are added"
        )
    return config
