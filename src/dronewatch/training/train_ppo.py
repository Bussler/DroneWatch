"""Train shared-policy PPO on SwarmSearch2D with RLlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ray

from dronewatch.evaluation.evaluate import evaluate_checkpoint
from dronewatch.training.rllib_config import (
    ModelKind,
    PPOBuildContext,
    build_ppo_config,
)


def train_ppo(
    *,
    iterations: int = 1,
    seed: int = 42,
    model: ModelKind = "feedforward",
    checkpoint_dir: str | Path = Path("artifacts/checkpoints/ppo"),
    checkpoint_frequency: int = 1,
    eval_episodes: int = 0,
    eval_report_path: str | Path | None = None,
    num_env_runners: int = 0,
    num_learners: int = 0,
    num_gpus_per_learner: int = 0,
    train_batch_size_per_learner: int = 1024,
    minibatch_size: int = 256,
    num_epochs: int = 5,
) -> dict[str, Any]:
    """Train PPO and return a compact run summary."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be greater than zero")

    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    ppo_context = PPOBuildContext(
        model=model,
        seed=seed,
        num_env_runners=num_env_runners,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        train_batch_size_per_learner=train_batch_size_per_learner,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
    )
    algorithm = build_ppo_config(ppo_context).build_algo()

    checkpoints: list[str] = []
    last_result: dict[str, Any] = {}
    final_checkpoint = ""
    try:
        for iteration in range(1, iterations + 1):
            last_result = algorithm.train()
            print(json.dumps(_training_progress(iteration, last_result), sort_keys=True))

            if iteration % checkpoint_frequency == 0:
                checkpoint = _save_checkpoint(algorithm, output_dir / f"iteration_{iteration:04d}")
                checkpoints.append(checkpoint)
                print(f"saved checkpoint: {checkpoint}")

        final_checkpoint = _save_checkpoint(algorithm, output_dir / "final")
        if final_checkpoint not in checkpoints:
            checkpoints.append(final_checkpoint)
        print(f"saved final checkpoint: {final_checkpoint}")
    finally:
        algorithm.stop()

    evaluation_report: dict[str, Any] | None = None
    if eval_episodes > 0:
        evaluation_report = evaluate_checkpoint(
            checkpoint=final_checkpoint,
            episodes=eval_episodes,
            seed=seed,
            report_path=eval_report_path,
            model=model,
            render=True,
            gif_path=output_dir / "evaluation_episode.gif",
            render_stride=4,
        )

    ray.shutdown()

    return {
        "iterations": iterations,
        "seed": seed,
        "model": model,
        "checkpoint_dir": str(output_dir),
        "checkpoints": checkpoints,
        "final_checkpoint": final_checkpoint,
        "last_result": _training_progress(iterations, last_result),
        "evaluation_report": evaluation_report,
    }


def _save_checkpoint(algorithm: Any, checkpoint_dir: Path) -> str:
    """Save an RLlib Algorithm checkpoint and return its path as a string."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = algorithm.save(str(checkpoint_dir.resolve()))
    if hasattr(saved, "checkpoint") and hasattr(saved.checkpoint, "path"):
        return str(saved.checkpoint.path)
    if hasattr(saved, "path"):
        return str(saved.path)
    return str(saved)


def _training_progress(iteration: int, result: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact progress summary from an RLlib training result."""
    env_runners = result.get("env_runners", {})
    custom_metrics = result.get("custom_metrics", {})
    progress: dict[str, Any] = {
        "iteration": iteration,
        "episode_return_mean": env_runners.get("episode_return_mean", result.get("episode_reward_mean")),
        "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
    }

    # The new RLlib API stack exposes callback metrics under env_runners.
    # Fall back to custom_metrics for compatibility with older result layouts.
    target_discovery_rate = env_runners.get(
        "dronewatch/target_discovery_rate",
        custom_metrics.get("dronewatch/target_discovery_rate_mean"),
    )
    coverage_ratio = env_runners.get(
        "dronewatch/coverage_ratio",
        custom_metrics.get("dronewatch/coverage_ratio_mean"),
    )
    if target_discovery_rate is not None:
        progress["target_discovery_rate"] = target_discovery_rate
    if coverage_ratio is not None:
        progress["coverage_ratio"] = coverage_ratio
    return progress


def main() -> None:
    """Command-line entry point for local PPO training."""
    parser = argparse.ArgumentParser(description="Train DroneWatch shared-policy PPO with RLlib.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["feedforward", "lstm"], default="feedforward")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/ppo"))
    parser.add_argument("--checkpoint-frequency", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-report-path", type=Path, default=Path("artifacts/reports/ppo_eval_report.json"))
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--num-gpus-per-learner", type=int, default=0)
    parser.add_argument("--train-batch-size-per-learner", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=5)
    args = parser.parse_args()

    summary = train_ppo(
        iterations=args.iterations,
        seed=args.seed,
        model=args.model,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        eval_episodes=args.eval_episodes,
        eval_report_path=args.eval_report_path,
        num_env_runners=args.num_env_runners,
        num_learners=args.num_learners,
        num_gpus_per_learner=args.num_gpus_per_learner,
        train_batch_size_per_learner=args.train_batch_size_per_learner,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
