"""Run Ray Tune hyperparameter search for shared-policy PPO."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ray
from omegaconf import OmegaConf
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler

from dronewatch.config.loader import (
    load_config,
    resolved_config_path,
    save_resolved_config,
)
from dronewatch.config.schema import DroneWatchConfig
from dronewatch.evaluation.evaluate import evaluate_checkpoint
from dronewatch.evaluation.reporting import write_json_report
from dronewatch.logging import (
    log_artifact_if_enabled,
    log_config_params,
    log_evaluation_report,
    log_metrics,
    set_mlflow_tags,
    start_child_mlflow_run,
    start_mlflow_run,
)
from dronewatch.training.rllib_config import build_ppo_config
from dronewatch.training.utils import (
    learner_progress,
    save_checkpoint,
    training_progress,
)


def tune_ppo(config: DroneWatchConfig) -> dict[str, Any]:
    """Run a Ray Tune PPO sweep and return a compact summary."""
    if not config.tune.search_space:
        raise ValueError("tune.search_space must contain at least one parameter")

    storage_path = config.project.output_dir / config.tune.storage_path
    report_path = config.project.artifact_dir / config.tune.report_path
    best_report_path = config.project.artifact_dir / config.tune.best_report_path
    saved_config_path = resolved_config_path(storage_path, config)
    mlflow_config = config.logging.mlflow
    parent_run_id: str | None = None
    evaluation_report: dict[str, Any] | None = None

    with start_mlflow_run(
        config.project.name,
        mlflow_config,
        tags={
            "project": config.project.name,
            "entrypoint": "tune_ppo",
            "environment": config.env.name,
            "model": config.model.kind,
            "tune_metric": config.tune.metric,
            "tune_mode": config.tune.mode,
        },
    ) as run:
        if run is not None:
            parent_run_id = run.info.run_id
        saved_config_path = save_resolved_config(config, saved_config_path)
        log_config_params(config)
        if mlflow_config.log_config_artifact:
            log_artifact_if_enabled(mlflow_config, saved_config_path, artifact_path="config")

        ray.init(ignore_reinit_error=True, include_dashboard=False, runtime_env=_ray_runtime_env())
        try:
            tuner = tune.Tuner(
                tune.with_parameters(
                    _run_trial,
                    base_config=_trial_base_config(config),
                    parent_run_id=parent_run_id,
                ),
                param_space=_to_ray_search_space(config.tune.search_space),
                tune_config=tune.TuneConfig(
                    metric=config.tune.metric,
                    mode=config.tune.mode,
                    num_samples=config.tune.num_samples,
                    scheduler=_scheduler(config.tune.scheduler, metric=config.tune.metric, mode=config.tune.mode),
                ),
                run_config=RunConfig(
                    name="dronewatch_ppo_tune",
                    storage_path=str(storage_path.resolve()),
                ),
            )
            results = tuner.fit()
        finally:
            ray.shutdown()

        summary = _search_summary(results, metric=config.tune.metric, mode=config.tune.mode)
        write_json_report(report_path, summary)
        if mlflow_config.log_report_artifact:
            log_artifact_if_enabled(mlflow_config, report_path, artifact_path="reports")

        best_checkpoint = summary.get("best_checkpoint", "")
        if best_checkpoint and config.training.evaluation.enabled and config.training.evaluation.episodes > 0:
            evaluation_report = evaluate_checkpoint(
                checkpoint=best_checkpoint,
                episodes=config.training.evaluation.episodes,
                seed=config.project.seed if config.project.seed is not None else 0,
                report_path=best_report_path,
                model=config.model.kind,
                render=config.training.evaluation.render,
                gif_path=config.project.artifact_dir / config.training.evaluation.gif_path,
                render_stride=config.training.evaluation.render_stride,
                env_config=config.env,
                render_fps=config.rendering.fps,
            )
            log_evaluation_report(evaluation_report, prefix="eval")
            if mlflow_config.log_report_artifact:
                log_artifact_if_enabled(mlflow_config, best_report_path, artifact_path="reports")

        set_mlflow_tags(
            {
                "best_trial_id": summary.get("best_trial_id"),
                "best_checkpoint": best_checkpoint,
            }
        )
        log_metrics({"best_metric": summary.get("best_metric_value")}, prefix="tune")

    return {
        "metric": config.tune.metric,
        "mode": config.tune.mode,
        "num_samples": config.tune.num_samples,
        "resolved_config_path": str(saved_config_path),
        "report_path": str(report_path),
        "best_report_path": str(best_report_path) if evaluation_report is not None else "",
        "evaluation_report": evaluation_report,
        **summary,
    }


def _run_trial(
    sampled_params: dict[str, Any],
    *,
    base_config: dict[str, Any],
    parent_run_id: str | None,
) -> None:
    config = _apply_sampled_config(base_config, sampled_params)
    trial_context = tune.get_context()
    trial_id = trial_context.get_trial_id() or "trial"
    checkpoint_dir = config.project.artifact_dir / config.training.checkpoint.directory / trial_id
    checkpoint_frequency = config.training.checkpoint.frequency_iters
    last_metrics: dict[str, Any] = {}

    with start_child_mlflow_run(
        config.project.name,
        config.logging.mlflow,
        parent_run_id=parent_run_id,
        run_name=f"tune-{trial_id}",
        tags={
            "project": config.project.name,
            "entrypoint": "tune_ppo_trial",
            "trial_id": trial_id,
            "environment": config.env.name,
            "model": config.model.kind,
        },
    ):
        log_config_params(sampled_params, prefix="sampled")
        algorithm = build_ppo_config(
            training=config.training,
            env_config=config.env,
            model=config.model,
            seed=config.project.seed,
        ).build_algo()
        try:
            for iteration in range(1, config.training.stop.iterations + 1):
                result = algorithm.train()
                progress = training_progress(iteration, result)
                learner = learner_progress(result)
                log_metrics(progress, prefix="train", step=iteration)
                log_metrics(learner, prefix="learn", step=iteration)

                checkpoint = None
                checkpoint_path = ""
                if iteration % checkpoint_frequency == 0:
                    checkpoint_path = save_checkpoint(algorithm, checkpoint_dir / f"iteration_{iteration:04d}")
                    checkpoint = tune.Checkpoint.from_directory(checkpoint_path)

                last_metrics = _reported_metrics(progress, learner, sampled_params, trial_id, checkpoint_path)
                if checkpoint is None:
                    tune.report(last_metrics)
                else:
                    tune.report(last_metrics, checkpoint=checkpoint)

            final_checkpoint = save_checkpoint(algorithm, checkpoint_dir / "final")
            last_metrics = dict(last_metrics)
            last_metrics["checkpoint_path"] = final_checkpoint
            tune.report(last_metrics, checkpoint=tune.Checkpoint.from_directory(final_checkpoint))
        finally:
            algorithm.stop()


def _reported_metrics(
    progress: Mapping[str, Any],
    learner: Mapping[str, Any],
    sampled_params: Mapping[str, Any],
    trial_id: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    metrics = {key: value for key, value in progress.items() if value is not None}
    metrics.update({f"learn/{key}": value for key, value in learner.items() if value is not None})
    metrics["trial_id"] = trial_id
    metrics["checkpoint_path"] = checkpoint_path
    for key, value in sampled_params.items():
        metrics[f"sampled/{key}"] = value
    return metrics


def _to_ray_search_space(search_space: Mapping[str, Any]) -> dict[str, Any]:
    return {path: _to_ray_sampler(path, spec) for path, spec in search_space.items()}


def _to_ray_sampler(path: str, spec: Any) -> Any:
    if not isinstance(spec, Mapping):
        raise ValueError(f"search_space.{path} must be a mapping with a type field")
    sampler_type = str(spec.get("type", "")).lower()
    if sampler_type in {"choice", "grid_search"}:
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_space.{path}.values must be a non-empty list")
        return tune.choice(values) if sampler_type == "choice" else tune.grid_search(values)
    if sampler_type in {"uniform", "loguniform"}:
        lower = spec.get("lower")
        upper = spec.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"search_space.{path} requires lower and upper")
        return (
            tune.uniform(float(lower), float(upper))
            if sampler_type == "uniform"
            else tune.loguniform(float(lower), float(upper))
        )
    raise ValueError(f"unsupported search_space.{path}.type: {sampler_type!r}")


def _apply_sampled_config(base_config: Mapping[str, Any], sampled_params: Mapping[str, Any]) -> DroneWatchConfig:
    composed = OmegaConf.create(base_config)
    for path, value in sampled_params.items():
        OmegaConf.update(composed, path, value, merge=False)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("sampled config did not resolve to a mapping")
    return DroneWatchConfig.model_validate(data)


def _trial_base_config(config: DroneWatchConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data["project"]["artifact_dir"] = str(config.project.artifact_dir.resolve())
    data["project"]["output_dir"] = str(config.project.output_dir.resolve())
    tracking_uri = data["logging"]["mlflow"]["tracking_uri"]
    if isinstance(tracking_uri, str) and tracking_uri.startswith("file:"):
        uri_path = Path(tracking_uri.removeprefix("file:"))
        data["logging"]["mlflow"]["tracking_uri"] = f"file:{uri_path.resolve()}"
    return data


def _scheduler(name: str, metric: str, mode: str) -> Any:
    scheduler_name = name.lower()
    if scheduler_name == "fifo":
        return FIFOScheduler()
    if scheduler_name == "asha":
        return ASHAScheduler(metric=metric, mode=mode)
    raise ValueError(f"unsupported tune.scheduler: {name}")


def _ray_runtime_env() -> dict[str, Any]:
    import swarm_sim

    package_dir = Path(swarm_sim.__file__).parent
    return {"py_modules": [str(package_dir)]}


def _search_summary(results: Any, metric: str, mode: str) -> dict[str, Any]:
    best_result = results.get_best_result(metric=metric, mode=mode)
    trials: list[dict[str, Any]] = []
    for result in results:
        metrics = dict(result.metrics or {})
        trials.append(
            {
                "trial_id": metrics.get("trial_id", ""),
                "sampled_params": dict(result.config or {}),
                "metric_value": metrics.get(metric),
                "checkpoint_path": metrics.get("checkpoint_path", ""),
                "metrics": _compact_metrics(metrics),
            }
        )

    best_metrics = dict(best_result.metrics or {})
    return {
        "metric": metric,
        "mode": mode,
        "num_trials": len(trials),
        "best_trial_id": best_metrics.get("trial_id", ""),
        "best_metric_value": best_metrics.get(metric),
        "best_checkpoint": best_metrics.get("checkpoint_path", ""),
        "best_params": dict(best_result.config or {}),
        "trials": trials,
    }


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "iteration",
        "episode_return_mean",
        "target_discovery_rate",
        "discovered_target_count",
        "coverage_ratio",
        "collision_count",
        "obstacle_violation_count",
        "connectivity_ratio",
        "average_communication_neighbors",
        "episode_length",
        "success_rate",
        "num_env_steps_sampled_lifetime",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def main() -> None:
    """Command-line entry point for Ray Tune PPO search."""
    parser = argparse.ArgumentParser(description="Tune DroneWatch shared-policy PPO with Ray Tune.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args, overrides = parser.parse_known_args()

    summary = tune_ppo(config=load_config(args.config, overrides))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
