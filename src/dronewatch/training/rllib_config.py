"""RLlib PPO configuration for SwarmSearch2D."""

from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

from dronewatch.config.schema import (
    ModelConfig,
    ModelKind,
    NetworkConfig,
    ProjectConfig,
    SwarmSearchEnvConfig,
    TrainingConfig,
)
from dronewatch.envs import SwarmSearchEnv

from .callbacks import SwarmSearchMetricsCallback

SWARM_SEARCH_ENV_NAME = "SwarmSearch2D"
SHARED_POLICY_ID = "shared_policy"


def register_swarm_search_env() -> None:
    """Register the DroneWatch multi-agent environment with Ray Tune."""
    register_env(SWARM_SEARCH_ENV_NAME, _create_swarm_search_env)


def shared_policy_mapping_fn(agent_id: str, episode: Any, **kwargs: Any) -> str:
    """Map every homogeneous drone agent to the shared PPO policy."""
    del agent_id, episode, kwargs
    return SHARED_POLICY_ID


def build_ppo_config(
    training: TrainingConfig | None = None,
    env_config: SwarmSearchEnvConfig | None = None,
    model: ModelConfig | None = None,
    seed: int | None = ProjectConfig().seed,
) -> PPOConfig:
    """Build a PPOConfig for shared-policy DroneWatch training."""
    training = training or TrainingConfig()
    env_config = env_config or SwarmSearchEnvConfig()
    model = model or ModelConfig()
    ray_config = training.ray
    ppo = training.ppo
    env_config_payload = env_config.model_dump(mode="json") | {"seed": seed}
    register_swarm_search_env()

    return (
        PPOConfig()
        .framework("torch")
        .environment(env=SWARM_SEARCH_ENV_NAME, env_config=env_config_payload)
        .env_runners(
            num_env_runners=ray_config.num_env_runners,
            num_envs_per_env_runner=ray_config.num_envs_per_env_runner,
            rollout_fragment_length=ppo.rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .learners(num_learners=ray_config.num_learners, num_gpus_per_learner=ray_config.num_gpus_per_learner)
        .training(
            gamma=ppo.gamma,
            lambda_=ppo.lambda_,
            lr=ppo.lr,
            clip_param=ppo.clip_param,
            entropy_coeff=ppo.entropy_coeff,
            vf_loss_coeff=ppo.vf_loss_coeff,
            train_batch_size_per_learner=ppo.train_batch_size_per_learner,
            minibatch_size=ppo.minibatch_size,
            num_epochs=ppo.num_epochs,
        )
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=shared_policy_mapping_fn,
            policies_to_train=[SHARED_POLICY_ID],
            count_steps_by="env_steps",
        )
        .rl_module(model_config=_model_config(model.kind, model.network))
        .callbacks(SwarmSearchMetricsCallback)
        .debugging(seed=seed)
    )


def _model_config(model: ModelKind, network: NetworkConfig | None = None) -> DefaultModelConfig:
    """Return the default RLlib model config for the requested model kind."""
    network = network or NetworkConfig(use_lstm=(model == "lstm"))
    if model == "feedforward":
        return DefaultModelConfig(fcnet_hiddens=network.fcnet_hiddens, fcnet_activation=network.activation)
    if model == "lstm":
        return DefaultModelConfig(
            fcnet_hiddens=network.fcnet_hiddens,
            fcnet_activation=network.activation,
            use_lstm=network.use_lstm,
            lstm_cell_size=network.lstm_cell_size,
            max_seq_len=network.max_seq_len,
        )
    raise ValueError(f"unsupported model kind: {model}")


def _create_swarm_search_env(env_context: Any) -> SwarmSearchEnv:
    """Create a SwarmSearchEnv from RLlib's EnvContext."""
    config = dict(env_context or {})
    seed = config.pop("seed", None)
    if seed is not None:
        worker_index = int(getattr(env_context, "worker_index", 0) or 0)
        vector_index = int(getattr(env_context, "vector_index", 0) or 0)
        seed = int(seed) + worker_index * 1000 + vector_index
    return SwarmSearchEnv(config, seed=seed)
