from __future__ import annotations

from pathlib import Path

from dronewatch.config.schema import (
    CheckpointConfig,
    DroneWatchConfig,
    LoggingConfig,
    MlflowConfig,
    PPOHyperparameters,
    RayConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    TrainingStopConfig,
)
from dronewatch.training.train_ppo import (
    _learner_progress,
    _training_progress,
    train_ppo,
)


def test_training_progress_extracts_env_runner_task_metrics() -> None:
    progress = _training_progress(
        3,
        {
            "env_runners": {
                "episode_return_mean": 1.25,
                "dronewatch/target_discovery_rate": 0.5,
            },
        },
    )

    assert progress["iteration"] == 3
    assert progress["episode_return_mean"] == 1.25
    assert progress["target_discovery_rate"] == 0.5


def test_learner_progress_extracts_shared_policy_and_aggregate_metrics() -> None:
    progress = _learner_progress(
        {
            "learners": {
                "__all_modules__": {
                    "num_env_steps_trained_lifetime": 512,
                    "module_train_batch_size_mean": 128,
                },
                "shared_policy": {
                    "policy_loss": -0.1,
                    "vf_loss": 0.2,
                    "entropy": 1.5,
                    "mean_kl_loss": 0.03,
                    "ignored_metric": 99.0,
                },
            }
        }
    )

    assert progress["learner_policy_loss"] == -0.1
    assert progress["learner_vf_loss"] == 0.2
    assert progress["learner_entropy"] == 1.5
    assert progress["learner_mean_kl_loss"] == 0.03
    assert "learner_ignored_metric" not in progress


def test_learner_progress_returns_empty_when_learners_missing() -> None:
    assert _learner_progress({}) == {}
    assert _learner_progress({"learners": None}) == {}


def test_train_ppo_single_iteration_logs_metrics_and_keeps_distinct_checkpoints(tmp_path: Path) -> None:
    config = DroneWatchConfig(
        training=TrainingConfig(
            stop=TrainingStopConfig(iterations=1),
            ray=RayConfig(num_env_runners=0, num_learners=0),
            ppo=PPOHyperparameters(
                train_batch_size_per_learner=256,
                minibatch_size=128,
                num_epochs=1,
            ),
            checkpoint=CheckpointConfig(directory=tmp_path / "checkpoints", frequency_iters=1),
            evaluation=TrainingEvaluationConfig(enabled=False, episodes=0),
        ),
        logging=LoggingConfig(mlflow=MlflowConfig(enabled=False)),
    )
    summary = train_ppo(config=config)

    last_result = summary["last_result"]

    assert "target_discovery_rate" in last_result
    assert "coverage_ratio" in last_result
    assert 0.0 <= last_result["target_discovery_rate"] <= 1.0
    assert 0.0 <= last_result["coverage_ratio"] <= 1.0

    checkpoints = summary["checkpoints"]
    assert len(checkpoints) == 2
    assert summary["final_checkpoint"] == checkpoints[-1]
    assert checkpoints[0] != checkpoints[1]

    periodic_checkpoint = Path(checkpoints[0])
    final_checkpoint = Path(checkpoints[1])
    assert periodic_checkpoint.exists()
    assert final_checkpoint.exists()
    assert periodic_checkpoint.name == "iteration_0001"
    assert final_checkpoint.name == "final"
