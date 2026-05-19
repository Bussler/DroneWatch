from __future__ import annotations

from dronewatch.config.schema import (
    AgentConfig,
    EnvConfig,
    ModelConfig,
    NetworkConfig,
    PPOHyperparameters,
    RayConfig,
    SwarmSearchEnvConfig,
    TrainingConfig,
)
from dronewatch.training.rllib_config import (
    SHARED_POLICY_ID,
    SWARM_SEARCH_ENV_NAME,
    _create_swarm_search_env,
    build_ppo_config,
    shared_policy_mapping_fn,
)


def test_shared_policy_mapping_maps_all_agents_to_one_policy() -> None:
    assert shared_policy_mapping_fn("agent_0", None) == SHARED_POLICY_ID
    assert shared_policy_mapping_fn("agent_15", None) == SHARED_POLICY_ID


def test_build_ppo_config_uses_swarm_search_env_and_env_step_counting() -> None:
    training = TrainingConfig(
        ray=RayConfig(num_env_runners=0),
        ppo=PPOHyperparameters(
            train_batch_size_per_learner=200,
            minibatch_size=64,
            num_epochs=1,
        ),
    )
    config = build_ppo_config(training=training, seed=7)

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] == 7
    assert config.env_config["simulation"]["agents"]["count"] == 16
    assert config.count_steps_by == "env_steps"
    assert SHARED_POLICY_ID in config.policies


def test_build_ppo_config_without_inputs_uses_default_config() -> None:
    config = build_ppo_config()

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] == 133742
    assert config.env_config["name"] == SWARM_SEARCH_ENV_NAME
    assert config.count_steps_by == "env_steps"
    assert SHARED_POLICY_ID in config.policies


def test_build_ppo_config_accepts_none_seed() -> None:
    config = build_ppo_config(seed=None)

    assert config.env == SWARM_SEARCH_ENV_NAME
    assert config.env_config["seed"] is None


def test_build_ppo_config_accepts_env_and_model_values() -> None:
    env_config = SwarmSearchEnvConfig(simulation=EnvConfig(agents=AgentConfig(count=4)))
    model = ModelConfig(
        kind="lstm",
        network=NetworkConfig(use_lstm=True, fcnet_hiddens=[32, 16], lstm_cell_size=64, max_seq_len=8),
    )

    config = build_ppo_config(env_config=env_config, model=model)

    assert config.env_config["simulation"]["agents"]["count"] == 4
    assert config.env_config["seed"] == 133742
    assert config.model_config["fcnet_hiddens"] == [32, 16]
    assert config.model_config["use_lstm"] is True
    assert config.model_config["lstm_cell_size"] == 64


def test_create_swarm_search_env_passes_runtime_seed_outside_structural_config() -> None:
    env = _create_swarm_search_env({"seed": 7, "simulation": {"agents": {"count": 2}}})

    observations, infos = env.reset()

    assert len(observations) == 2
    assert len(infos) == 2
