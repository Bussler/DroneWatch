"""RLlib PPO configuration for SwarmSearch2D."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

from dronewatch.envs import SwarmSearchEnv

from .callbacks import SwarmSearchMetricsCallback

SWARM_SEARCH_ENV_NAME = "SwarmSearch2D"
SHARED_POLICY_ID = "shared_policy"
ModelKind = Literal["feedforward", "lstm"]
RolloutFragmentLength = Literal["auto"] | int


class PPOBuildContext(BaseModel):
    """Validated inputs for building a shared-policy PPOConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelKind = Field(default="feedforward", description="Neural network model family for the shared policy.")
    seed: int | None = Field(
        default=42,
        ge=0,
        description="Optional non-negative random seed for RLlib and environment creation.",
    )
    num_env_runners: int = Field(default=0, ge=0, description="Number of RLlib environment runner workers.")
    num_learners: int = Field(default=0, ge=0, description="Number of RLlib learner workers.")
    num_gpus_per_learner: int = Field(default=0, ge=0, description="GPU count assigned to each learner.")
    train_batch_size_per_learner: int = Field(default=1024, gt=0, description="Training batch size per learner.")
    minibatch_size: int = Field(default=256, gt=0, description="PPO stochastic gradient minibatch size.")
    num_epochs: int = Field(default=5, gt=0, description="Number of PPO optimization epochs per train batch.")
    rollout_fragment_length: RolloutFragmentLength = Field(
        default="auto", description="RLlib rollout fragment length or auto-tuning sentinel."
    )
    lr: float = Field(default=3e-4, gt=0.0, description="PPO optimizer learning rate.")
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor.")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="GAE lambda parameter.")
    clip_param: float = Field(default=0.2, gt=0.0, description="PPO policy clipping parameter.")
    entropy_coeff: float = Field(default=0.01, ge=0.0, description="Entropy bonus coefficient.")
    vf_loss_coeff: float = Field(default=1.0, ge=0.0, description="Value function loss coefficient.")

    @model_validator(mode="after")
    def validate_batch_and_rollout_settings(self) -> PPOBuildContext:
        """Validate settings whose constraints depend on multiple fields."""
        if isinstance(self.rollout_fragment_length, int) and self.rollout_fragment_length <= 0:
            raise ValueError("rollout_fragment_length must be 'auto' or a positive integer")
        if self.minibatch_size > self.train_batch_size_per_learner:
            raise ValueError("minibatch_size must be less than or equal to train_batch_size_per_learner")
        return self


def register_swarm_search_env() -> None:
    """Register the DroneWatch multi-agent environment with Ray Tune."""
    register_env(SWARM_SEARCH_ENV_NAME, _create_swarm_search_env)


def shared_policy_mapping_fn(agent_id: str, episode: Any, **kwargs: Any) -> str:
    """Map every homogeneous drone agent to the shared PPO policy."""
    del agent_id, episode, kwargs
    return SHARED_POLICY_ID


def build_ppo_config(context: PPOBuildContext | None = None) -> PPOConfig:
    """Build a PPOConfig for shared-policy DroneWatch training."""
    context = context or PPOBuildContext()
    register_swarm_search_env()

    return (
        PPOConfig()
        .framework("torch")
        .environment(env=SWARM_SEARCH_ENV_NAME, env_config={"seed": context.seed})
        .env_runners(
            num_env_runners=context.num_env_runners,
            rollout_fragment_length=context.rollout_fragment_length,
            batch_mode="complete_episodes",
        )
        .learners(num_learners=context.num_learners, num_gpus_per_learner=context.num_gpus_per_learner)
        .training(
            gamma=context.gamma,
            lambda_=context.lambda_,
            lr=context.lr,
            clip_param=context.clip_param,
            entropy_coeff=context.entropy_coeff,
            vf_loss_coeff=context.vf_loss_coeff,
            train_batch_size_per_learner=context.train_batch_size_per_learner,
            minibatch_size=context.minibatch_size,
            num_epochs=context.num_epochs,
        )
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=shared_policy_mapping_fn,
            policies_to_train=[SHARED_POLICY_ID],
            count_steps_by="env_steps",
        )
        .rl_module(model_config=_model_config(context.model))
        .callbacks(SwarmSearchMetricsCallback)
        .debugging(seed=context.seed)
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
