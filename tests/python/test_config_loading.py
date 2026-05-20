from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronewatch.config.loader import (
    load_config,
    load_evaluation_config,
    load_random_policy_config,
    save_resolved_config,
)


def test_load_config_composes_default_groups() -> None:
    config = load_config("configs/config.yaml")

    assert config.project.name == "DroneWatch"
    assert config.env.name == "SwarmSearch2D"
    assert config.env.simulation.agents.count == 16
    assert config.model.kind == "feedforward"
    assert config.training.stop.iterations == 10
    assert not hasattr(config, "evaluation")
    assert not hasattr(config, "baseline")


def test_load_config_supports_group_and_field_overrides() -> None:
    config = load_config(
        "configs/config.yaml",
        [
            "model=ppo_lstm",
            "training=debug",
            "env.simulation.agents.count=4",
            "env.simulation.max_episode_steps=12",
            "model.fcnet_hiddens=[64,64]",
            "project.seed=7",
        ],
    )

    assert config.model.kind == "lstm"
    assert config.model.fcnet_hiddens == [64, 64]
    assert config.training.stop.iterations == 1
    assert config.env.simulation.agents.count == 4
    assert config.env.simulation.max_episode_steps == 12
    assert config.project.seed == 7


def test_load_evaluation_config_composes_standalone_groups() -> None:
    config = load_evaluation_config(
        "configs/evaluate.yaml",
        [
            "model=ppo_lstm",
            "evaluation=debug",
            "evaluation.checkpoint=artifacts/checkpoints/ppo/final",
            "evaluation.render=false",
            "project.seed=99",
        ],
    )

    assert config.model.kind == "lstm"
    assert config.evaluation.episodes == 1
    assert config.evaluation.checkpoint == Path("artifacts/checkpoints/ppo/final")
    assert config.evaluation.render is False
    assert config.project.seed == 99
    assert not hasattr(config, "training")


def test_load_random_policy_config_composes_standalone_groups() -> None:
    config = load_random_policy_config(
        "configs/random_policy.yaml",
        [
            "random_policy=debug",
            "random_policy.render=false",
            "random_policy.episodes=2",
            "project.seed=42",
        ],
    )

    assert config.random_policy.episodes == 2
    assert config.random_policy.report_path == Path("reports/random_policy_debug_report.json")
    assert config.random_policy.render is False
    assert config.project.seed == 42
    assert not hasattr(config, "training")
    assert not hasattr(config, "model")


def test_load_config_rejects_invalid_overrides() -> None:
    with pytest.raises(ValidationError):
        load_config("configs/config.yaml", ["training.ppo.minibatch_size=2048"])

    with pytest.raises(ValidationError):
        load_config("configs/config.yaml", ["baseline.random.render=true"])

    with pytest.raises(ValidationError):
        load_config("configs/config.yaml", ["training.stop.timesteps_total=10"])

    with pytest.raises(ValueError, match="key=value"):
        load_config("configs/config.yaml", ["training.debug"])


def test_save_resolved_config_writes_yaml(tmp_path: Path) -> None:
    config = load_config("configs/debug.yaml", ["env.simulation.agents.count=3"])
    path = save_resolved_config(config, tmp_path / "resolved_config.yaml")

    text = path.read_text(encoding="utf-8")
    assert "agents:\n      count: 3" in text
    assert "resolved_config_filename: resolved_config.yaml" in text
