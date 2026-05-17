"""Training entry points and RLlib configuration for DroneWatch."""

from .rllib_config import (
    SHARED_POLICY_ID,
    SWARM_SEARCH_ENV_NAME,
    PPOBuildContext,
    build_ppo_config,
    register_swarm_search_env,
)

__all__ = [
    "PPOBuildContext",
    "SHARED_POLICY_ID",
    "SWARM_SEARCH_ENV_NAME",
    "build_ppo_config",
    "register_swarm_search_env",
]
