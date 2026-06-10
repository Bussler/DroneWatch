from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import dronewatch.evaluation.evaluate as evaluate_module
import dronewatch.training.tune_ppo as tune_ppo_module
import scripts.random_policy as random_policy_module


def test_random_policy_main_uses_standalone_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    report_path = tmp_path / "random_report.json"

    def fake_run_random_policy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"policy": "random"}

    monkeypatch.setattr(random_policy_module, "run_random_policy", fake_run_random_policy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "random_policy",
            "--config",
            "configs/random_policy.yaml",
            "random_policy=debug",
            "random_policy.episodes=2",
            f"random_policy.report_path={report_path}",
            "random_policy.render=false",
            "project.seed=42",
        ],
    )

    random_policy_module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["policy"] == "random"
    assert captured["episodes"] == 2
    assert captured["seed"] == 42
    assert captured["report_path"] == report_path
    assert captured["render"] is False
    assert captured["env_config"].name == "SwarmSearch2D"


def test_evaluate_main_uses_standalone_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    checkpoint_path = tmp_path / "checkpoint"
    report_path = tmp_path / "ppo_report"

    def fake_evaluate_checkpoint(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"policy": "ppo"}

    monkeypatch.setattr(evaluate_module, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            "configs/evaluate.yaml",
            "model=ppo_lstm",
            f"evaluation.checkpoint={checkpoint_path}",
            "evaluation.episodes=3",
            f"evaluation.report_path={report_path}",
            "evaluation.render=false",
            "logging.mlflow.enabled=false",
            "project.seed=123",
        ],
    )

    evaluate_module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["policy"] == "ppo"
    assert captured["checkpoint"] == checkpoint_path
    assert captured["episodes"] == 3
    assert captured["seed"] == 123
    assert captured["report_path"] == report_path / "EvalDroneWatchObstacleAvoidance" / "evaluation_report.json"
    assert captured["model"] == "lstm"
    assert captured["render"] is False
    assert captured["env_config"].name == "SwarmSearch2D"


def test_tune_ppo_main_uses_training_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_tune_ppo(config: Any) -> dict[str, Any]:
        captured["config"] = config
        return {"metric": config.tune.metric, "num_samples": config.tune.num_samples}

    monkeypatch.setattr(tune_ppo_module, "tune_ppo", fake_tune_ppo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_ppo",
            "--config",
            "configs/tune_ppo.yaml",
            "tune.num_samples=2",
            "logging.mlflow.enabled=false",
        ],
    )

    tune_ppo_module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["metric"] == "target_discovery_rate"
    assert output["num_samples"] == 2
    assert captured["config"].training.evaluation.enabled is False
    assert captured["config"].logging.mlflow.enabled is False
