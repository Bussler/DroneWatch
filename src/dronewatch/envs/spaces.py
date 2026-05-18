"""Spaces and defaults for the SwarmSearch2D wrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from dronewatch.config.schema import (
    AgentDefaults,
    ObservationConfig,
    ObservationDefaults,
    ObstacleDefaults,
    RewardWeights,
    WorldDefaults,
)

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


def observation_size(config: ObservationConfig | None = None) -> int:
    """Return the fixed observation vector size for a configured environment."""
    observation_config = config or ObservationConfig()
    size = (
        OWN_STATE_SIZE
        + observation_config.max_visible_agents * VISIBLE_AGENT_FEATURES
        + observation_config.max_visible_targets * VISIBLE_TARGET_FEATURES
        + observation_config.max_visible_obstacles * VISIBLE_OBSTACLE_FEATURES
    )
    if observation_config.include_communication_summary:
        size += COMMUNICATION_SUMMARY_SIZE
    return size


def agent_ids(num_agents: int = AGENT_DEFAULTS.count) -> list[str]:
    """Return stable RLlib agent IDs for the homogeneous swarm."""
    return [f"agent_{index}" for index in range(num_agents)]


def action_space() -> gym.spaces.Box:
    """Return the per-agent action space accepted by the Rust simulator."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def observation_space(config: ObservationConfig | None = None) -> gym.spaces.Box:
    """Return the per-agent fixed-size observation space."""
    return gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(observation_size(config),),
        dtype=np.float32,
    )
