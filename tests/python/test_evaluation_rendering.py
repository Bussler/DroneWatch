from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import dronewatch.evaluation.evaluate as evaluate_module
from dronewatch.evaluation.evaluate import evaluate_algorithm
from dronewatch.rendering import SimulationFrame
from dronewatch.training.rllib_config import SHARED_POLICY_ID


class _FakeAlgorithm:
    """Minimal algorithm shape needed by evaluate_algorithm tests."""

    def get_module(self, policy_id: str) -> object:
        assert policy_id == SHARED_POLICY_ID
        return object()


def test_evaluate_algorithm_rejects_render_without_gif_path() -> None:
    with pytest.raises(ValueError, match="gif_path"):
        evaluate_algorithm(algorithm=_FakeAlgorithm(), episodes=1, seed=1, render=True)  # type: ignore[arg-type]


def test_evaluate_algorithm_rejects_invalid_render_stride() -> None:
    with pytest.raises(ValueError, match="render_stride"):
        evaluate_algorithm(algorithm=_FakeAlgorithm(), episodes=1, seed=1, render_stride=0)  # type: ignore[arg-type]


def test_evaluate_algorithm_renders_first_episode_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rendered: dict[str, Any] = {}

    def fake_initial_module_state(_module: object) -> dict[str, Any]:
        return {}

    def fake_compute_action(
        _module: object,
        _observation: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros((2,), dtype=np.float32), state

    def fake_render_episode_gif(frames: list[SimulationFrame], path: str | Path) -> None:
        rendered["frames"] = list(frames)
        rendered["path"] = Path(path)

    monkeypatch.setattr(evaluate_module, "_initial_module_state", fake_initial_module_state)
    monkeypatch.setattr(evaluate_module, "_compute_action", fake_compute_action)
    monkeypatch.setattr(evaluate_module, "render_episode_gif", fake_render_episode_gif)

    gif_path = tmp_path / "ppo_eval.gif"
    report = evaluate_algorithm(
        algorithm=_FakeAlgorithm(),  # type: ignore[arg-type]
        episodes=2,
        seed=5,
        render=True,
        gif_path=gif_path,
        render_stride=500,
    )

    assert report["policy"] == "ppo"
    assert report["num_episodes"] == 2
    assert rendered["path"] == gif_path
    frames = rendered["frames"]
    assert len(frames) == 2
    assert frames[0].simulation_metrics["timestep"] == 0
    assert frames[-1].simulation_metrics["timestep"] == 200
