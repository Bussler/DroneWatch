from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from dronewatch.config.schema import MlflowConfig
from dronewatch.logging import (
    flatten_params,
    log_artifact_if_enabled,
    log_config_params,
    log_evaluation_report,
    log_metrics,
    start_mlflow_run,
)


def test_flatten_params_handles_nested_config_values() -> None:
    params = flatten_params(
        {
            "project": {"name": "DroneWatch", "seed": 7},
            "model": {"hiddens": [64, 64]},
            "none_value": None,
        },
        prefix="config",
    )

    assert params["config.project.name"] == "DroneWatch"
    assert params["config.project.seed"] == "7"
    assert params["config.model.hiddens"] == "[64,64]"
    assert params["config.none_value"] == "null"


def test_mlflow_logger_records_params_metrics_and_artifacts(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    config = MlflowConfig(
        tracking_uri=tracking_uri,
        run_name="logger-test",
    )
    artifact_path = tmp_path / "resolved_config.yaml"
    artifact_path.write_text("project:\n  name: DroneWatch\n", encoding="utf-8")

    with start_mlflow_run("dronewatch-test", config, tags={"entrypoint": "test"}) as run:
        assert run is not None
        run_id = run.info.run_id
        log_config_params({"project": {"name": "DroneWatch"}}, prefix="config")
        log_metrics({"iteration": 1, "nested": {"value": 2.5}, "ignored": "text"}, prefix="train", step=1)
        log_evaluation_report(
            {
                "policy": "ppo",
                "num_episodes": 2,
                "mean_reward": 1.25,
                "mean_target_discovery_rate": 0.5,
                "episodes": [{"reward": 1.0}, {"reward": 1.5}],
            },
            prefix="eval",
        )
        log_artifact_if_enabled(config, artifact_path, artifact_path="config")

    client = MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(run_id).data
    artifacts = client.list_artifacts(run_id, "config")

    assert run_data.tags["entrypoint"] == "test"
    assert run_data.params["config.project.name"] == "DroneWatch"
    assert run_data.metrics["train/iteration"] == 1.0
    assert run_data.metrics["train/nested/value"] == 2.5
    assert run_data.metrics["eval/mean_reward"] == 1.25
    assert run_data.metrics["eval/mean_target_discovery_rate"] == 0.5
    assert [artifact.path for artifact in artifacts] == ["config/resolved_config.yaml"]


def test_mlflow_run_disabled_is_noop() -> None:
    with start_mlflow_run("dronewatch-test", MlflowConfig(enabled=False)) as run:
        assert run is None
