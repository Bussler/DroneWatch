"""Pydantic configuration models for DroneWatch experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    """Immutable Pydantic base for project configuration objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")


ModelKind = Literal["feedforward", "lstm"]
RolloutFragmentLength = Literal["auto"] | int


class ProjectConfig(_FrozenModel):
    """Project-level metadata and default seed."""

    name: str = Field(default="DroneWatch", min_length=1)
    seed: int | None = Field(default=42, ge=0)


class RuntimeConfig(_FrozenModel):
    """Runtime artifact locations and resolved-config file naming."""

    output_dir: Path = Path("outputs")
    artifact_dir: Path = Path("artifacts")
    resolved_config_filename: str = Field(default="resolved_config.yaml", min_length=1)


class WorldConfig(_FrozenModel):
    """Continuous 2D world dimensions and timestep."""

    width: float = Field(default=100.0, gt=0.0)
    height: float = Field(default=100.0, gt=0.0)
    dt: float = Field(default=1.0, gt=0.0)


class AgentConfig(_FrozenModel):
    """Homogeneous swarm movement, collision, sensing, and communication settings."""

    max_speed: float = Field(default=2.0, gt=0.0)
    collision_radius: float = Field(default=0.75, gt=0.0)
    sensing_radius: float = Field(default=15.0, gt=0.0)
    communication_radius: float = Field(default=20.0, gt=0.0)


class TargetConfig(_FrozenModel):
    """Static target generation settings."""

    count: int = Field(default=20, ge=0)
    discovery_radius: float = Field(default=2.0, gt=0.0)


class ObstacleConfig(_FrozenModel):
    """Circular no-fly obstacle generation settings."""

    count: int = Field(default=8, ge=0)
    min_radius: float = Field(default=2.0, gt=0.0)
    max_radius: float = Field(default=6.0, gt=0.0)

    @model_validator(mode="after")
    def validate_radius_order(self) -> ObstacleConfig:
        """Ensure obstacle radius bounds are ordered."""
        if self.max_radius < self.min_radius:
            raise ValueError("max_radius must be greater than or equal to min_radius")
        return self


class CoverageConfig(_FrozenModel):
    """Coverage grid settings used by simulator metrics."""

    grid_width: int = Field(default=50, gt=0)
    grid_height: int = Field(default=50, gt=0)
    sensing_radius: float = Field(default=10.0, gt=0.0)


class ObservationConfig(_FrozenModel):
    """Fixed local observation capacity and communication-summary settings."""

    max_visible_agents: int = Field(default=5, gt=0)
    max_visible_targets: int = Field(default=5, gt=0)
    max_visible_obstacles: int = Field(default=5, gt=0)
    include_communication_summary: bool = True


class RewardWeights(_FrozenModel):
    """Cooperative reward weights for target search."""

    target_discovered: float = Field(default=5.0, ge=0.0)
    new_coverage_cell: float = Field(default=0.02, ge=0.0)
    agent_collision: float = Field(default=-0.25, le=0.0)
    obstacle_collision: float = Field(default=-0.5, le=0.0)
    step_penalty: float = Field(default=-0.001, le=0.0)


class EnvConfig(_FrozenModel):
    """Complete SwarmSearch2D environment and simulator configuration."""

    name: str = Field(default="SwarmSearch2D", min_length=1)
    num_agents: int = Field(default=16, gt=0)
    max_episode_steps: int = Field(default=200, gt=0)
    world: WorldConfig = Field(default_factory=WorldConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    obstacles: ObstacleConfig = Field(default_factory=ObstacleConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    reward: RewardWeights = Field(default_factory=RewardWeights)

    def to_rust_config_dict(self) -> dict[str, Any]:
        """Return the nested dictionary expected by the Rust simulator binding."""
        return {
            "max_episode_steps": self.max_episode_steps,
            "world": self.world.model_dump(mode="json"),
            "agents": {"count": self.num_agents, **self.agents.model_dump(mode="json")},
            "targets": self.targets.model_dump(mode="json"),
            "obstacles": self.obstacles.model_dump(mode="json"),
            "coverage": self.coverage.model_dump(mode="json"),
        }


class SwarmSearchEnvConfig(_FrozenModel):
    """Validated configuration accepted by the RLlib MultiAgentEnv wrapper."""

    seed: int | None = Field(default=None, ge=0)
    env: EnvConfig = Field(default_factory=EnvConfig)


class NetworkConfig(_FrozenModel):
    """RLlib default model network settings."""

    use_lstm: bool = False
    lstm_cell_size: int = Field(default=128, gt=0)
    max_seq_len: int = Field(default=20, gt=0)
    fcnet_hiddens: list[int] = Field(default_factory=lambda: [256, 256])
    activation: str = Field(default="tanh", min_length=1)

    @model_validator(mode="after")
    def validate_hidden_layers(self) -> NetworkConfig:
        """Ensure all hidden layer sizes are positive."""
        if not self.fcnet_hiddens or any(size <= 0 for size in self.fcnet_hiddens):
            raise ValueError("fcnet_hiddens must contain positive layer sizes")
        return self


class ModelConfig(_FrozenModel):
    """Shared-policy PPO model configuration."""

    algorithm: Literal["PPO"] = "PPO"
    kind: ModelKind = "feedforward"
    shared_policy: bool = True
    network: NetworkConfig = Field(default_factory=NetworkConfig)


class PPOHyperparameters(_FrozenModel):
    """PPO optimizer and sampling hyperparameters."""

    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0)
    lr: float = Field(default=3e-4, gt=0.0)
    clip_param: float = Field(default=0.2, gt=0.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    vf_loss_coeff: float = Field(default=1.0, ge=0.0)
    train_batch_size_per_learner: int = Field(default=1024, gt=0)
    minibatch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=5, gt=0)
    rollout_fragment_length: RolloutFragmentLength = "auto"

    @model_validator(mode="after")
    def validate_batch_and_rollout_settings(self) -> PPOHyperparameters:
        """Validate fields whose constraints depend on multiple values."""
        if isinstance(self.rollout_fragment_length, int) and self.rollout_fragment_length <= 0:
            raise ValueError("rollout_fragment_length must be 'auto' or a positive integer")
        if self.minibatch_size > self.train_batch_size_per_learner:
            raise ValueError("minibatch_size must be less than or equal to train_batch_size_per_learner")
        return self


class RayConfig(_FrozenModel):
    """Local Ray/RLlib worker allocation settings."""

    num_env_runners: int = Field(default=0, ge=0)
    num_envs_per_env_runner: int = Field(default=1, gt=0)
    num_learners: int = Field(default=0, ge=0)
    num_gpus_per_learner: int = Field(default=0, ge=0)
    num_cpus: int | None = Field(default=None, gt=0)
    num_gpus: int | None = Field(default=None, ge=0)


class TrainingStopConfig(_FrozenModel):
    """Training stop criteria used by local PPO runs."""

    iterations: int = Field(default=10, gt=0)
    timesteps_total: int | None = Field(default=None, gt=0)


class CheckpointConfig(_FrozenModel):
    """Checkpoint persistence settings for local PPO training."""

    directory: Path = Path("artifacts/checkpoints/ppo")
    frequency_iters: int = Field(default=5, gt=0)
    keep_last: int | None = Field(default=5, gt=0)


class TrainingEvaluationConfig(_FrozenModel):
    """Evaluation settings applied after a training run."""

    enabled: bool = True
    episodes: int = Field(default=5, ge=0)
    report_path: Path = Path("artifacts/reports/ppo_eval_report.json")
    render: bool = True
    gif_path: Path = Path("artifacts/gifs/ppo_eval_episode.gif")
    render_stride: int = Field(default=4, gt=0)


class TrainingConfig(_FrozenModel):
    """Local PPO training settings."""

    seed: int | None = Field(default=None, ge=0)
    stop: TrainingStopConfig = Field(default_factory=TrainingStopConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    ppo: PPOHyperparameters = Field(default_factory=PPOHyperparameters)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: TrainingEvaluationConfig = Field(default_factory=TrainingEvaluationConfig)


class EvaluationConfig(_FrozenModel):
    """Standalone checkpoint evaluation settings."""

    checkpoint: Path | None = None
    episodes: int = Field(default=10, gt=0)
    seed: int | None = Field(default=None, ge=0)
    report_path: Path = Path("artifacts/reports/ppo_eval_report.json")
    render: bool = False
    gif_path: Path = Path("artifacts/gifs/ppo_eval_episode.gif")
    render_stride: int = Field(default=4, gt=0)


class RandomPolicyConfig(_FrozenModel):
    """Random-policy baseline run settings."""

    episodes: int = Field(default=1, gt=0)
    seed: int | None = Field(default=None, ge=0)
    report_path: Path = Path("artifacts/reports/random_policy_report.json")
    render: bool = False
    gif_path: Path = Path("artifacts/gifs/random_policy_episode.gif")
    render_stride: int = Field(default=4, gt=0)


class BaselineConfig(_FrozenModel):
    """Baseline policy configurations."""

    random: RandomPolicyConfig = Field(default_factory=RandomPolicyConfig)


class RenderingConfig(_FrozenModel):
    """Shared rendering settings."""

    fps: int = Field(default=12, gt=0)


class PathsConfig(_FrozenModel):
    """Common artifact paths used by scripts and configs."""

    checkpoint_dir: Path = Path("artifacts/checkpoints/ppo")
    report_dir: Path = Path("artifacts/reports")
    gif_dir: Path = Path("artifacts/gifs")


class TuneConfig(_FrozenModel):
    """Validated Ray Tune metadata and search-space placeholder for later phases."""

    enabled: bool = False
    metric: str = "mean_target_discovery_rate"
    mode: Literal["max", "min"] = "max"
    num_samples: int = Field(default=1, gt=0)
    search_space: dict[str, Any] = Field(default_factory=dict)


class PPOBuildContext(_FrozenModel):
    """Validated inputs for building a shared-policy PPOConfig."""

    model: ModelKind = "feedforward"
    seed: int | None = Field(default=42, ge=0)
    num_env_runners: int = Field(default=0, ge=0)
    num_learners: int = Field(default=0, ge=0)
    num_gpus_per_learner: int = Field(default=0, ge=0)
    train_batch_size_per_learner: int = Field(default=1024, gt=0)
    minibatch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=5, gt=0)
    rollout_fragment_length: RolloutFragmentLength = "auto"
    lr: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_param: float = Field(default=0.2, gt=0.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    vf_loss_coeff: float = Field(default=1.0, ge=0.0)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    @model_validator(mode="after")
    def validate_batch_and_rollout_settings(self) -> PPOBuildContext:
        """Validate fields whose constraints depend on multiple values."""
        PPOHyperparameters(
            gamma=self.gamma,
            lambda_=self.lambda_,
            lr=self.lr,
            clip_param=self.clip_param,
            entropy_coeff=self.entropy_coeff,
            vf_loss_coeff=self.vf_loss_coeff,
            train_batch_size_per_learner=self.train_batch_size_per_learner,
            minibatch_size=self.minibatch_size,
            num_epochs=self.num_epochs,
            rollout_fragment_length=self.rollout_fragment_length,
        )
        return self


class DroneWatchConfig(_FrozenModel):
    """Fully composed and validated DroneWatch experiment configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)

    def training_seed(self) -> int | None:
        """Return the effective seed for training-created environments."""
        return self.training.seed if self.training.seed is not None else self.project.seed

    def evaluation_seed(self) -> int | None:
        """Return the effective seed for standalone PPO evaluation."""
        return self.evaluation.seed if self.evaluation.seed is not None else self.project.seed

    def random_seed(self) -> int | None:
        """Return the effective seed for random-policy rollouts."""
        return self.baseline.random.seed if self.baseline.random.seed is not None else self.project.seed

    def swarm_env_config(self, seed: int | None) -> dict[str, Any]:
        """Return the environment config dictionary passed through RLlib."""
        return {"seed": seed, "env": self.env.model_dump(mode="json")}

    def ppo_build_context(self) -> PPOBuildContext:
        """Create the compact RLlib build context from the root experiment config."""
        ppo = self.training.ppo
        return PPOBuildContext(
            model=self.model.kind,
            seed=self.training_seed(),
            num_env_runners=self.training.ray.num_env_runners,
            num_learners=self.training.ray.num_learners,
            num_gpus_per_learner=self.training.ray.num_gpus_per_learner,
            train_batch_size_per_learner=ppo.train_batch_size_per_learner,
            minibatch_size=ppo.minibatch_size,
            num_epochs=ppo.num_epochs,
            rollout_fragment_length=ppo.rollout_fragment_length,
            lr=ppo.lr,
            gamma=ppo.gamma,
            lambda_=ppo.lambda_,
            clip_param=ppo.clip_param,
            entropy_coeff=ppo.entropy_coeff,
            vf_loss_coeff=ppo.vf_loss_coeff,
            network=self.model.network,
        )


class WorldDefaults(_FrozenModel):
    """Backward-compatible world defaults mirrored from Phase 2."""

    width: float = Field(default=100.0, gt=0.0)
    height: float = Field(default=100.0, gt=0.0)
    max_episode_steps: int = Field(default=200, gt=0)


class AgentDefaults(_FrozenModel):
    """Backward-compatible homogeneous swarm defaults mirrored from Phase 2."""

    count: int = Field(default=16, gt=0)
    max_speed: float = Field(default=2.0, gt=0.0)
    sensing_radius: float = Field(default=15.0, gt=0.0)
    communication_radius: float = Field(default=20.0, gt=0.0)


class ObstacleDefaults(_FrozenModel):
    """Backward-compatible obstacle normalization default."""

    expected_max_radius: float = Field(default=6.0, gt=0.0)


class ObservationDefaults(_FrozenModel):
    """Backward-compatible fixed observation capacity defaults."""

    max_visible_agents: int = Field(default=5, gt=0)
    max_visible_targets: int = Field(default=5, gt=0)
    max_visible_obstacles: int = Field(default=5, gt=0)
