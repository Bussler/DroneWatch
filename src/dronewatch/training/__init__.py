"""Training entry points and RLlib configuration for DroneWatch."""

from .rllib_config import (
    SHARED_POLICY_ID,
    build_ppo_config,
    register_swarm_search_env,
)

__all__ = [
    "SHARED_POLICY_ID",
    "build_ppo_config",
    "register_swarm_search_env",
]
