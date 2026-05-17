from __future__ import annotations

import pytest
from pydantic import ValidationError

from dronewatch.config.schema import (
    DroneWatchConfig,
    EnvConfig,
    ModelConfig,
    NetworkConfig,
    TrainingConfig,
)
from dronewatch.training.rllib_config import (
    SHARED_POLICY_ID,
    SWARM_SEARCH_ENV_NAME,
    PPOBuildContext,
    build_ppo_config,
    shared_policy_mapping_fn,
)


def test_shared_policy_mapping_maps_all_agents_to_one_policy() -> None:
    assert shared_policy_mapping_fn("agent_0", None) == SHARED_POLICY_ID
    assert shared_policy_mapping_fn("agent_15", None) == SHARED_POLICY_ID


def test_ppo_build_context_defaults_match_smoke_training_values() -> None:
    context = PPOBuildContext()

    assert context.model == "feedforward"
    assert context.seed == 42
    assert context.num_env_runners == 0
    assert context.train_batch_size_per_learner == 1024
    assert context.minibatch_size == 256
    assert context.rollout_fragment_length == "auto"


def test_ppo_build_context_accepts_none_seed() -> None:
    context = PPOBuildContext(seed=None)

    assert context.seed is None


def test_ppo_build_context_is_frozen_and_forbids_extra_fields() -> None:
    context = PPOBuildContext()

    with pytest.raises(ValidationError):
        context.seed = 7  # type: ignore[misc]

    with pytest.raises(ValidationError):
        PPOBuildContext.model_validate({"unexpected": 1})


def test_ppo_build_context_rejects_invalid_numeric_values() -> None:
    invalid_contexts = [
        {"seed": -1},
        {"num_env_runners": -1},
        {"num_learners": -1},
        {"num_gpus_per_learner": -1},
        {"train_batch_size_per_learner": 0},
        {"minibatch_size": 0},
        {"num_epochs": 0},
        {"lr": 0.0},
        {"gamma": 1.1},
        {"lambda_": -0.1},
        {"clip_param": 0.0},
        {"entropy_coeff": -0.01},
        {"vf_loss_coeff": -1.0},
    ]

    for values in invalid_contexts:
        with pytest.raises(ValidationError):
            PPOBuildContext(**values)


def test_ppo_build_context_rejects_invalid_rollout_and_batch_relationship() -> None:
    with pytest.raises(ValidationError):
        PPOBuildContext(rollout_fragment_length=0)

    with pytest.raises(ValidationError):
        PPOBuildContext(rollout_fragment_length="episode")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        PPOBuildContext(train_batch_size_per_learner=128, minibatch_size=256)


def test_build_ppo_config_uses_swarm_search_env_and_env_step_counting() -> None:
    context = PPOBuildContext(
        model="feedforward",
        seed=7,
        num_env_runners=0,
        train_batch_size_per_learner=200,
        minibatch_size=64,
        num_epochs=1,
    )
    config = build_ppo_config(context)

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] == 7
    assert config.env_config["env"]["num_agents"] == 16
    assert config.count_steps_by == "env_steps"
    assert SHARED_POLICY_ID in config.policies


def test_build_ppo_config_without_context_uses_default_context() -> None:
    config = build_ppo_config()

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] == 42
    assert config.env_config["env"]["name"] == SWARM_SEARCH_ENV_NAME
    assert config.count_steps_by == "env_steps"
    assert SHARED_POLICY_ID in config.policies


def test_build_ppo_config_accepts_none_seed() -> None:
    config = build_ppo_config(PPOBuildContext(seed=None))

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] is None


def test_build_ppo_config_accepts_root_config_with_env_and_network_values() -> None:
    root = DroneWatchConfig(
        env=EnvConfig(num_agents=4),
        model=ModelConfig(
            kind="lstm",
            network=NetworkConfig(use_lstm=True, fcnet_hiddens=[32, 16], lstm_cell_size=64, max_seq_len=8),
        ),
        training=TrainingConfig(),
    )

    config = build_ppo_config(root)

    assert config.env_config["env"]["num_agents"] == 4
    assert config.env_config["seed"] == 42
    assert config.model_config["fcnet_hiddens"] == [32, 16]
    assert config.model_config["use_lstm"] is True
    assert config.model_config["lstm_cell_size"] == 64
