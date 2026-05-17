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
from dronewatch.training.rllib_config import (
    ModelKind,
    build_ppo_config,
)


def train_ppo(
    *,
    config: DroneWatchConfig | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    model: ModelKind | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_frequency: int | None = None,
    eval_episodes: int | None = None,
    eval_report_path: str | Path | None = None,
    num_env_runners: int | None = None,
    num_learners: int | None = None,
    num_gpus_per_learner: int | None = None,
    train_batch_size_per_learner: int | None = None,
    minibatch_size: int | None = None,
    num_epochs: int | None = None,
) -> dict[str, Any]:
    """Train PPO and return a compact run summary."""
    config = _with_legacy_overrides(
        config or DroneWatchConfig(),
        iterations=iterations,
        seed=seed,
        model=model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=checkpoint_frequency,
        eval_episodes=eval_episodes,
        eval_report_path=eval_report_path,
        num_env_runners=num_env_runners,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        train_batch_size_per_learner=train_batch_size_per_learner,
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
    )
    iterations = config.training.stop.iterations
    seed = config.training_seed()
    model = config.model.kind
    checkpoint_frequency = config.training.checkpoint.frequency_iters
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be greater than zero")

    output_dir = Path(config.training.checkpoint.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config_path = save_resolved_config(config, resolved_config_path(output_dir, config))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algorithm = build_ppo_config(config).build_algo()

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
            report_path=train_eval.report_path,
            model=model,
            render=train_eval.render,
            gif_path=train_eval.gif_path,
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


def _with_legacy_overrides(config: DroneWatchConfig, **overrides: Any) -> DroneWatchConfig:
    """Apply legacy keyword arguments used by tests and older call sites."""
    if not any(value is not None for value in overrides.values()):
        return config

    data = config.model_dump(mode="python")
    if overrides["iterations"] is not None:
        data["training"]["stop"]["iterations"] = overrides["iterations"]
    if overrides["seed"] is not None:
        data["training"]["seed"] = overrides["seed"]
    if overrides["model"] is not None:
        data["model"]["kind"] = overrides["model"]
        data["model"]["network"]["use_lstm"] = overrides["model"] == "lstm"
    if overrides["checkpoint_dir"] is not None:
        data["training"]["checkpoint"]["directory"] = overrides["checkpoint_dir"]
    if overrides["checkpoint_frequency"] is not None:
        data["training"]["checkpoint"]["frequency_iters"] = overrides["checkpoint_frequency"]
    if overrides["eval_episodes"] is not None:
        data["training"]["evaluation"]["episodes"] = overrides["eval_episodes"]
        data["training"]["evaluation"]["enabled"] = overrides["eval_episodes"] > 0
    if overrides["eval_report_path"] is not None:
        data["training"]["evaluation"]["report_path"] = overrides["eval_report_path"]
    for key in ("num_env_runners", "num_learners", "num_gpus_per_learner"):
        if overrides[key] is not None:
            data["training"]["ray"][key] = overrides[key]
    for key in ("train_batch_size_per_learner", "minibatch_size", "num_epochs"):
        if overrides[key] is not None:
            data["training"]["ppo"][key] = overrides[key]
    return DroneWatchConfig.model_validate(data)


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
