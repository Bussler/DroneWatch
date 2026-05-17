from __future__ import annotations

from dronewatch.training.rllib_config import (
    SHARED_POLICY_ID,
    SWARM_SEARCH_ENV_NAME,
    build_ppo_config,
    shared_policy_mapping_fn,
)


def test_shared_policy_mapping_maps_all_agents_to_one_policy() -> None:
    assert shared_policy_mapping_fn("agent_0", None) == SHARED_POLICY_ID
    assert shared_policy_mapping_fn("agent_15", None) == SHARED_POLICY_ID


def test_build_ppo_config_uses_swarm_search_env_and_env_step_counting() -> None:
    config = build_ppo_config(
        model="feedforward",
        seed=7,
        num_env_runners=0,
        train_batch_size_per_learner=200,
        minibatch_size=64,
        num_epochs=1,
    )

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config == {"seed": 7}
    assert config.count_steps_by == "env_steps"
    assert SHARED_POLICY_ID in config.policies
