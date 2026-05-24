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
from dronewatch.logging import (
    log_artifact_if_enabled,
    log_config_params,
    log_evaluation_report,
    log_metrics,
    set_mlflow_tags,
    start_mlflow_run,
)
from dronewatch.training._progress import (
    learner_progress as _learner_progress,
)
from dronewatch.training._progress import (
    save_checkpoint as _save_checkpoint,
)
from dronewatch.training._progress import (
    training_progress as _training_progress,
)
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

    checkpoints: list[str] = []
    last_result: dict[str, Any] = {}
    final_checkpoint = ""
    evaluation_report: dict[str, Any] | None = None
    output_dir = config.project.artifact_dir / config.training.checkpoint.directory
    report_path = config.project.artifact_dir / config.training.evaluation.report_path

    mlflow_config = config.logging.mlflow
    with start_mlflow_run(
        mlflow_config,
        tags={
            "project": config.project.name,
            "entrypoint": "train_ppo",
            "environment": config.env.name,
            "model": model,
        },
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_config_path = save_resolved_config(config, resolved_config_path(output_dir, config))
        log_config_params(config)
        if mlflow_config.log_config_artifact:
            log_artifact_if_enabled(mlflow_config, saved_config_path, artifact_path="config")

        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            algorithm = build_ppo_config(
                training=config.training,
                env_config=config.env,
                model=config.model,
                seed=seed,
            ).build_algo()
            try:
                for iteration in range(1, iterations + 1):
                    last_result = algorithm.train()
                    progress = _training_progress(iteration, last_result)
                    learner_progress = _learner_progress(last_result)
                    print(json.dumps(progress, sort_keys=True))
                    log_metrics(progress, prefix="train", step=iteration)
                    log_metrics(learner_progress, prefix="learn", step=iteration)

                    if iteration % checkpoint_frequency == 0:
                        checkpoint = _save_checkpoint(algorithm, output_dir / f"iteration_{iteration:04d}")
                        checkpoints.append(checkpoint)
                        print(f"saved checkpoint: {checkpoint}")

                final_checkpoint = _save_checkpoint(algorithm, output_dir / "final")
                if final_checkpoint not in checkpoints:
                    checkpoints.append(final_checkpoint)
                set_mlflow_tags({"final_checkpoint": final_checkpoint})
                print(f"saved final checkpoint: {final_checkpoint}")
            finally:
                algorithm.stop()

            train_eval = config.training.evaluation
            if train_eval.enabled and train_eval.episodes > 0:
                evaluation_report = evaluate_checkpoint(
                    checkpoint=final_checkpoint,
                    episodes=train_eval.episodes,
                    seed=seed if seed is not None else 0,
                    report_path=report_path,
                    model=model,
                    render=train_eval.render,
                    gif_path=config.project.artifact_dir / train_eval.gif_path,
                    render_stride=train_eval.render_stride,
                    env_config=config.env,
                    render_fps=config.rendering.fps,
                )
                log_evaluation_report(evaluation_report, prefix="eval")
                if mlflow_config.log_report_artifact:
                    log_artifact_if_enabled(mlflow_config, report_path, artifact_path="reports")
        finally:
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


def main() -> None:
    """Command-line entry point for local PPO training."""
    parser = argparse.ArgumentParser(description="Train DroneWatch shared-policy PPO with RLlib.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args, overrides = parser.parse_known_args()

    summary = train_ppo(config=load_config(args.config, overrides))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
