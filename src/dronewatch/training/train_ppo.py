"""Train shared-policy PPO on SwarmSearch2D with RLlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ray

from dronewatch.config.loader import (
    load_config,
    resolved_config_path,
    save_resolved_config,
)
from dronewatch.config.schema import DroneWatchConfig
from dronewatch.evaluation.evaluate import evaluate_checkpoint
from dronewatch.training.rllib_config import build_ppo_config


def train_ppo(
    config: DroneWatchConfig,
) -> dict[str, Any]:
    """Train PPO and return a compact run summary."""
    iterations = config.training.stop.iterations
    seed = config.project.seed
    model = config.model.kind
    checkpoint_frequency = config.training.checkpoint.frequency_iters
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be greater than zero")

    output_dir = config.project.artifact_dir / config.training.checkpoint.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config_path = save_resolved_config(config, resolved_config_path(output_dir, config))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algorithm = build_ppo_config(
        training=config.training,
        env_config=config.env,
        model=config.model,
        seed=seed,
    ).build_algo()

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
    train_eval = config.training.evaluation
    if train_eval.enabled and train_eval.episodes > 0:
        evaluation_report = evaluate_checkpoint(
            checkpoint=final_checkpoint,
            episodes=train_eval.episodes,
            seed=seed if seed is not None else 0,
            report_path=config.project.artifact_dir / train_eval.report_path,
            model=model,
            render=train_eval.render,
            gif_path=config.project.artifact_dir / train_eval.gif_path,
            render_stride=train_eval.render_stride,
            env_config=config.env,
            render_fps=config.rendering.fps,
        )

    ray.shutdown()

    return {
        "iterations": iterations,
        "seed": seed,
        "model": model,
        "checkpoint_dir": str(output_dir),
        "resolved_config_path": str(saved_config_path),
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
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args, overrides = parser.parse_known_args()

    summary = train_ppo(config=load_config(args.config, overrides))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
