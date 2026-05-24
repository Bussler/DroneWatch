"""Small explicit MLflow integration layer for DroneWatch experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import mlflow
from pydantic import BaseModel

from dronewatch.config.schema import MlflowConfig

MAX_PARAM_VALUE_LENGTH = 5000


@contextmanager
def start_mlflow_run(
    experiment_name: str, config: MlflowConfig, tags: Mapping[str, Any] | None = None
) -> Iterator[Any | None]:
    """Start an MLflow run when enabled, otherwise yield None."""
    if not config.enabled:
        yield None
        return

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = _timestamped_run_name(config.run_name)
    with mlflow.start_run(run_name=run_name, tags=_string_values(tags or {})) as run:
        if config.log_system_metrics and hasattr(mlflow, "enable_system_metrics_logging"):
            mlflow.enable_system_metrics_logging()
        yield run


@contextmanager
def start_child_mlflow_run(
    experiment_name: str,
    config: MlflowConfig,
    parent_run_id: str | None = None,
    tags: Mapping[str, Any] | None = None,
    run_name: str | None = None,
) -> Iterator[Any | None]:
    """Start an MLflow child run when enabled, otherwise yield None."""
    if not config.enabled:
        yield None
        return

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_tags = dict(tags or {})
    if parent_run_id is not None:
        run_tags["mlflow.parentRunId"] = parent_run_id
        run_tags["dronewatch_parent_run_id"] = parent_run_id

    with mlflow.start_run(
        run_name=_timestamped_run_name(run_name or config.run_name),
        nested=mlflow.active_run() is not None,
        tags=_string_values(run_tags),
    ) as run:
        yield run


def set_mlflow_tags(tags: Mapping[str, Any]) -> None:
    """Set tags on the active MLflow run when one exists."""
    if mlflow.active_run() is None:
        return
    mlflow.set_tags(_string_values(tags))


def log_config_params(config: BaseModel | Mapping[str, Any], prefix: str = "config") -> None:
    """Log a flattened config model or mapping as MLflow parameters."""
    if mlflow.active_run() is None:
        return

    params = flatten_params(config, prefix=prefix)
    for chunk in _chunks(params, size=100):
        mlflow.log_params(chunk)


def log_metrics(metrics: Mapping[str, Any], prefix: str, step: int | None = None) -> None:
    """Log numeric metrics under a prefix on the active MLflow run."""
    if mlflow.active_run() is None:
        return

    numeric_metrics = _flatten_numeric_metrics(metrics, prefix=prefix)
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics, step=step)


def log_evaluation_report(report: Mapping[str, Any], prefix: str = "eval") -> None:
    """Log aggregate numeric metrics from an evaluation report."""
    aggregate_metrics = {key: value for key, value in report.items() if key != "episodes"}
    log_metrics(aggregate_metrics, prefix=prefix)


def log_artifact_if_enabled(config: MlflowConfig, path: str | Path | None, artifact_path: str) -> None:
    """Log an artifact path when MLflow is active and the file exists."""
    if not config.enabled or path is None:
        return
    output_path = Path(path)
    if not output_path.exists():
        return

    if mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(output_path), artifact_path=artifact_path)


def flatten_params(data: BaseModel | Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested structured data into MLflow-compatible string parameters."""
    if isinstance(data, BaseModel):
        payload: Any = data.model_dump(mode="json")
    else:
        payload = data
    return _flatten_value(payload, prefix=prefix)


def _flatten_value(value: Any, prefix: str) -> dict[str, str]:
    if isinstance(value, Mapping):
        flattened: dict[str, str] = {}
        for key, child_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_value(child_value, child_prefix))
        return flattened

    if isinstance(value, list | tuple):
        return {prefix: _param_value(json.dumps(value, sort_keys=True, separators=(",", ":")))}
    if value is None:
        return {prefix: "null"}
    return {prefix: _param_value(str(value))}


def _flatten_numeric_metrics(data: Mapping[str, Any], prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        metric_name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            metrics.update(_flatten_numeric_metrics(value, metric_name))
            continue
        if isinstance(value, Real) and not isinstance(value, bool):
            metric_value = float(value)
            if math.isfinite(metric_value):
                metrics[metric_name] = metric_value
    return metrics


def _param_value(value: str) -> str:
    if len(value) <= MAX_PARAM_VALUE_LENGTH:
        return value
    return value[: MAX_PARAM_VALUE_LENGTH - 3] + "..."


def _string_values(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items() if value is not None}


def _timestamped_run_name(run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        return timestamp
    return f"{run_name}_{timestamp}"


def _chunks(data: Mapping[str, str], size: int) -> Iterator[dict[str, str]]:
    items = list(data.items())
    for index in range(0, len(items), size):
        yield dict(items[index : index + size])
