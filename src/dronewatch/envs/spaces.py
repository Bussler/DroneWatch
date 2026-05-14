"""Spaces and defaults for the Phase 2 SwarmSearch2D wrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    """Immutable Pydantic base model for Phase 2 configuration objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldDefaults(_FrozenModel):
    """Default world dimensions and episode horizon mirrored from Rust."""

    width: float = Field(default=100.0, gt=0.0, description="World width in continuous coordinate units.")
    height: float = Field(default=100.0, gt=0.0, description="World height in continuous coordinate units.")
    max_episode_steps: int = Field(default=200, gt=0, description="Maximum number of simulator steps per episode.")


class AgentDefaults(_FrozenModel):
    """Default homogeneous swarm settings mirrored from Rust."""

    count: int = Field(default=16, gt=0, description="Number of homogeneous drones in the swarm.")
    max_speed: float = Field(default=2.0, gt=0.0, description="Maximum movement speed per unit timestep.")
    sensing_radius: float = Field(default=15.0, gt=0.0, description="Local sensing radius for observation building.")
    communication_radius: float = Field(
        default=20.0, gt=0.0, description="Distance threshold for communication neighbors."
    )


class ObstacleDefaults(_FrozenModel):
    """Default obstacle settings needed by the Python observation layer."""

    expected_max_radius: float = Field(
        default=6.0,
        gt=0.0,
        description="Expected maximum circular obstacle radius used for observation normalization.",
    )


class ObservationDefaults(_FrozenModel):
    """Fixed observation capacity settings for local entity slots."""

    max_visible_agents: int = Field(default=5, gt=0, description="Maximum visible neighboring agents per observation.")
    max_visible_targets: int = Field(default=5, gt=0, description="Maximum visible targets per observation.")
    max_visible_obstacles: int = Field(default=5, gt=0, description="Maximum visible obstacles per observation.")


class RewardWeights(_FrozenModel):
    """Cooperative reward weights for the Phase 2 Python reward function."""

    target_discovered: float = Field(default=5.0, ge=0.0, description="Reward per newly discovered target.")
    new_coverage_cell: float = Field(default=0.02, ge=0.0, description="Reward per newly covered grid cell.")
    agent_collision: float = Field(default=-0.25, le=0.0, description="Penalty per agent-agent collision pair.")
    obstacle_collision: float = Field(default=-0.5, le=0.0, description="Penalty per agent-obstacle overlap.")
    step_penalty: float = Field(default=-0.001, le=0.0, description="Small per-step time penalty.")


class SwarmSearchEnvConfig(_FrozenModel):
    """Pydantic configuration accepted by the Phase 2 RLlib environment.

    RLlib passes environment configuration as a dictionary, so the environment constructor validates
    that dictionary directly with this model. Extra keys are forbidden until Rust configuration
    bindings and OmegaConf-driven config loading are introduced in later phases.
    """

    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional non-negative reset seed; None delegates seed resolution to the Rust simulator.",
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
