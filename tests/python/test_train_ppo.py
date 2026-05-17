from __future__ import annotations

from pathlib import Path

from dronewatch.training.train_ppo import train_ppo


def test_train_ppo_single_iteration_logs_metrics_and_keeps_distinct_checkpoints(tmp_path: Path) -> None:
    summary = train_ppo(
        iterations=1,
        seed=42,
        model="feedforward",
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_frequency=1,
        eval_episodes=0,
        num_env_runners=0,
        num_learners=0,
        train_batch_size_per_learner=256,
        minibatch_size=128,
        num_epochs=1,
    )

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
