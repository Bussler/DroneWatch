"""RLlib PPO configuration for SwarmSearch2D."""

from __future__ import annotations

from typing import Any, Literal

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

from dronewatch.envs import SwarmSearchEnv

from .callbacks import SwarmSearchMetricsCallback

SWARM_SEARCH_ENV_NAME = "SwarmSearch2D"
SHARED_POLICY_ID = "shared_policy"
ModelKind = Literal["feedforward", "lstm"]


def register_swarm_search_env() -> None:
    """Register the DroneWatch multi-agent environment with Ray Tune."""
    register_env(SWARM_SEARCH_ENV_NAME, _create_swarm_search_env)


def shared_policy_mapping_fn(agent_id: str, episode: Any, **kwargs: Any) -> str:
    """Map every homogeneous drone agent to the shared PPO policy."""
    del agent_id, episode, kwargs
    return SHARED_POLICY_ID


def build_ppo_config(
    *,
    model: ModelKind = "feedforward",
    seed: int = 42,
    num_env_runners: int = 0,
    num_learners: int = 0,
    num_gpus_per_learner: int = 0,
    train_batch_size_per_learner: int = 1024,
    minibatch_size: int = 256,
    num_epochs: int = 5,
    rollout_fragment_length: int | str = "auto",
    lr: float = 3e-4,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 1.0,
) -> PPOConfig:
    """Build a PPOConfig for shared-policy DroneWatch training."""
    register_swarm_search_env()

    return (
        PPOConfig()
        .framework("torch")
        .environment(env=SWARM_SEARCH_ENV_NAME, env_config={"seed": seed})
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .learners(num_learners=num_learners, num_gpus_per_learner=num_gpus_per_learner)
        .training(
            gamma=gamma,
            lambda_=lambda_,
            lr=lr,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_loss_coeff,
            train_batch_size_per_learner=train_batch_size_per_learner,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
        )
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=shared_policy_mapping_fn,
            policies_to_train=[SHARED_POLICY_ID],
            count_steps_by="env_steps",
        )
        .rl_module(model_config=_model_config(model))
        .callbacks(SwarmSearchMetricsCallback)
        .debugging(seed=seed)
    )


def _model_config(model: ModelKind) -> DefaultModelConfig:
    """Return the default RLlib model config for the requested model kind."""
    if model == "feedforward":
        return DefaultModelConfig(fcnet_hiddens=[256, 256], fcnet_activation="tanh")
    if model == "lstm":
        return DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="tanh",
            use_lstm=True,
            lstm_cell_size=128,
            max_seq_len=20,
        )
    raise ValueError(f"unsupported model kind: {model}")


def _create_swarm_search_env(env_context: Any) -> SwarmSearchEnv:
    """Create a SwarmSearchEnv from RLlib's EnvContext."""
    config = dict(env_context or {})
    seed = config.get("seed")
    if seed is not None:
        worker_index = int(getattr(env_context, "worker_index", 0) or 0)
        vector_index = int(getattr(env_context, "vector_index", 0) or 0)
        seed = int(seed) + worker_index * 1000 + vector_index
    return SwarmSearchEnv({"seed": seed})
