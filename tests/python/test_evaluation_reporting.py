from __future__ import annotations

from pathlib import Path

import pytest

from dronewatch.evaluation.reporting import (
    aggregate_report,
    episode_summary,
    write_json_report,
)


def test_episode_summary_and_report_schema(tmp_path: Path) -> None:
    summary = episode_summary(
        1.5,
        {
            "target_discovery_rate": 0.25,
            "discovered_target_count": 5,
            "coverage_ratio": 0.4,
            "collision_count": 2,
            "obstacle_violation_count": 1,
            "connectivity_ratio": 0.75,
            "all_targets_discovered": False,
            "timestep": 200,
        },
        {
            "episode_shared_reward": 1.0,
            "episode_local_reward": 0.5,
            "episode_reward_terms": {
                "target_discovery": 0.75,
                "coverage": 0.25,
                "agent_collision": -0.1,
                "obstacle_collision": -0.05,
                "step_penalty": -0.01,
                "remaining_targets": -0.2,
                "success_bonus": 0.0,
                "visible_target_approach": 0.86,
            },
        },
    )

    report = aggregate_report([summary], policy="ppo", extra={"checkpoint": "checkpoint_000001"})

    expected_keys = {
        "policy",
        "num_episodes",
        "mean_reward",
        "mean_shared_reward",
        "mean_local_reward",
        "mean_target_discovery_rate",
        "mean_discovered_target_count",
        "mean_coverage_ratio",
        "mean_collision_count",
        "mean_obstacle_violation_count",
        "mean_connectivity_ratio",
        "mean_reward_term_target_discovery",
        "mean_reward_term_coverage",
        "mean_reward_term_agent_collision",
        "mean_reward_term_obstacle_collision",
        "mean_reward_term_step_penalty",
        "mean_reward_term_remaining_targets",
        "mean_reward_term_success_bonus",
        "mean_reward_term_visible_target_approach",
        "success_rate",
        "mean_episode_length",
        "episodes",
        "checkpoint",
    }
    assert expected_keys.issubset(report)
    assert report["policy"] == "ppo"
    assert report["num_episodes"] == 1
    assert report["mean_target_discovery_rate"] == 0.25
    assert report["mean_shared_reward"] == 1.0
    assert report["mean_local_reward"] == 0.5
    assert report["mean_reward_term_visible_target_approach"] == 0.86

    report_path = tmp_path / "report.json"
    write_json_report(report_path, report)
    assert report_path.exists()


def test_aggregate_report_rejects_empty_episodes() -> None:
    with pytest.raises(ValueError, match="episode_summaries"):
        aggregate_report([], policy="ppo")
