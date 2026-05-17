from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronewatch.config.loader import load_config, save_resolved_config


def test_load_config_composes_default_groups() -> None:
    config = load_config("configs/config.yaml")

    assert config.project.name == "DroneWatch"
    assert config.env.name == "SwarmSearch2D"
    assert config.env.num_agents == 16
    assert config.model.kind == "feedforward"
    assert config.training.stop.iterations == 10


def test_load_config_supports_group_and_field_overrides() -> None:
    config = load_config(
        "configs/config.yaml",
        [
            "model=ppo_lstm",
            "training=debug",
            "env.num_agents=4",
            "env.max_episode_steps=12",
            "model.network.fcnet_hiddens=[64,64]",
            "baseline.random.render=true",
        ],
    )

    assert config.model.kind == "lstm"
    assert config.model.network.use_lstm is True
    assert config.model.network.fcnet_hiddens == [64, 64]
    assert config.training.stop.iterations == 1
    assert config.env.num_agents == 4
    assert config.env.max_episode_steps == 12
    assert config.baseline.random.render is True


def test_load_config_rejects_invalid_overrides() -> None:
    with pytest.raises(ValidationError):
        load_config("configs/config.yaml", ["training.ppo.minibatch_size=2048"])

    with pytest.raises(ValueError, match="key=value"):
        load_config("configs/config.yaml", ["training.debug"])


def test_save_resolved_config_writes_yaml(tmp_path: Path) -> None:
    config = load_config("configs/debug.yaml", ["env.num_agents=3"])
    path = save_resolved_config(config, tmp_path / "resolved_config.yaml")

    text = path.read_text(encoding="utf-8")
    assert "num_agents: 3" in text
    assert "resolved_config_filename: resolved_config.yaml" in text
