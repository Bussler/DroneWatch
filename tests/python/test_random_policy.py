from __future__ import annotations

from pathlib import Path

from dronewatch.rendering import SimulationFrame, render_episode_gif
from dronewatch.sim import SwarmSimulation
from scripts.random_policy import run_random_policy


def test_random_policy_rollout_writes_report_and_gif(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    gif_path = tmp_path / "episode.gif"

    report = run_random_policy(
        episodes=1,
        seed=133742,
        report_path=report_path,
        gif_path=gif_path,
        render=True,
    )

    expected_keys = {
        "policy",
        "num_episodes",
        "mean_reward",
        "mean_target_discovery_rate",
        "mean_discovered_target_count",
        "mean_coverage_ratio",
        "mean_collision_count",
        "mean_obstacle_violation_count",
        "mean_connectivity_ratio",
        "success_rate",
        "mean_episode_length",
        "episodes",
    }
    assert expected_keys.issubset(report)
    assert report_path.exists()
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    assert 0.0 <= report["mean_target_discovery_rate"] <= 1.0
    assert 0.0 <= report["mean_coverage_ratio"] <= 1.0
    assert 0.0 <= report["mean_connectivity_ratio"] <= 1.0


def test_random_policy_report_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_random_policy(episodes=1, seed=7, report_path=tmp_path / "first.json")
    second = run_random_policy(episodes=1, seed=7, report_path=tmp_path / "second.json")

    assert first == second


def test_simulation_frame_renders_single_frame_gif(tmp_path: Path) -> None:
    sim = SwarmSimulation(seed=5)
    frame = SimulationFrame.from_snapshots(sim.state(), sim.metrics())
    gif_path = tmp_path / "single_frame.gif"

    render_episode_gif([frame], gif_path)

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
