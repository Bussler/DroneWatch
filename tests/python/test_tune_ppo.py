from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dronewatch.config.loader import load_config
from dronewatch.training.tune_ppo import (
    _apply_sampled_config,
    _search_summary,
    _to_ray_search_space,
    _trial_base_config,
)


def test_to_ray_search_space_accepts_initial_sampler_types() -> None:
    search_space = _to_ray_search_space(
        {
            "training.ppo.lr": {"type": "loguniform", "lower": 0.0001, "upper": 0.001},
            "training.ppo.entropy_coeff": {"type": "uniform", "lower": 0.0, "upper": 0.02},
            "training.ppo.num_epochs": {"type": "choice", "values": [3, 5]},
            "model.kind": {"type": "grid_search", "values": ["feedforward", "lstm"]},
        }
    )

    assert set(search_space) == {
        "training.ppo.lr",
        "training.ppo.entropy_coeff",
        "training.ppo.num_epochs",
        "model.kind",
    }


def test_to_ray_search_space_rejects_invalid_specs() -> None:
    with pytest.raises(ValueError, match="type field"):
        _to_ray_search_space({"training.ppo.lr": [0.0001, 0.001]})

    with pytest.raises(ValueError, match="non-empty list"):
        _to_ray_search_space({"training.ppo.lr": {"type": "choice", "values": []}})

    with pytest.raises(ValueError, match="requires lower and upper"):
        _to_ray_search_space({"training.ppo.lr": {"type": "uniform", "lower": 0.0}})

    with pytest.raises(ValueError, match="unsupported"):
        _to_ray_search_space({"training.ppo.lr": {"type": "normal", "mean": 0.0, "sd": 1.0}})


def test_apply_sampled_config_updates_and_revalidates_config() -> None:
    config = load_config("configs/config.yaml", ["training=tune_ppo"])

    sampled = _apply_sampled_config(
        config.model_dump(mode="json"),
        {
            "training.ppo.lr": 0.0007,
            "training.ppo.entropy_coeff": 0.02,
            "training.ppo.train_batch_size_per_learner": 1024,
            "model.kind": "lstm",
            "model.lstm_cell_size": 64,
        },
    )

    assert sampled.training.ppo.lr == 0.0007
    assert sampled.training.ppo.entropy_coeff == 0.02
    assert sampled.training.ppo.train_batch_size_per_learner == 1024
    assert sampled.model.kind == "lstm"
    assert sampled.model.lstm_cell_size == 64


def test_apply_sampled_config_rejects_invalid_combination() -> None:
    config = load_config("configs/config.yaml", ["training=tune_ppo"])

    with pytest.raises(ValidationError):
        _apply_sampled_config(
            config.model_dump(mode="json"),
            {
                "training.ppo.train_batch_size_per_learner": 64,
                "training.ppo.minibatch_size": 128,
            },
        )


def test_trial_base_config_uses_absolute_artifact_paths() -> None:
    config = load_config("configs/config.yaml", ["training=tune_ppo"])
    data = _trial_base_config(config)

    assert Path(data["project"]["artifact_dir"]).is_absolute()
    assert Path(data["project"]["output_dir"]).is_absolute()
    assert data["logging"]["mlflow"]["tracking_uri"].startswith("file:/")


class _FakeResult:
    def __init__(self, metrics: dict[str, Any], config: dict[str, Any]) -> None:
        self.metrics = metrics
        self.config = config


class _FakeResults:
    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results

    def __iter__(self) -> Any:
        return iter(self._results)

    def get_best_result(self, metric: str, mode: str) -> _FakeResult:
        key = max if mode == "max" else min
        return key(self._results, key=lambda result: result.metrics[metric])


def test_search_summary_extracts_best_trial() -> None:
    results = _FakeResults(
        [
            _FakeResult(
                {
                    "trial_id": "trial-a",
                    "target_discovery_rate": 0.25,
                    "checkpoint_path": str(Path("checkpoints/a")),
                    "coverage_ratio": 0.4,
                },
                {"training.ppo.lr": 0.0001},
            ),
            _FakeResult(
                {
                    "trial_id": "trial-b",
                    "target_discovery_rate": 0.5,
                    "checkpoint_path": str(Path("checkpoints/b")),
                    "coverage_ratio": 0.6,
                },
                {"training.ppo.lr": 0.0003},
            ),
        ]
    )

    summary = _search_summary(results, metric="target_discovery_rate", mode="max")

    assert summary["num_trials"] == 2
    assert summary["best_trial_id"] == "trial-b"
    assert summary["best_metric_value"] == 0.5
    assert summary["best_checkpoint"] == str(Path("checkpoints/b"))
    assert summary["best_params"] == {"training.ppo.lr": 0.0003}
    assert summary["trials"][0]["metrics"] == {"target_discovery_rate": 0.25, "coverage_ratio": 0.4}
