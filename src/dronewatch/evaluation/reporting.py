"""Shared evaluation report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dronewatch.envs.reward import REWARD_TERM_METRIC_NAMES


def episode_summary(
    episode_reward: float,
    metrics: dict[str, Any],
    reward_info: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Create a single-episode metric summary from final simulator metrics."""
    reward_info = reward_info or {}
    reward_terms = reward_info.get("episode_reward_terms", {})
    summary = {
        "reward": float(episode_reward),
        "shared_reward": float(reward_info.get("episode_shared_reward", episode_reward)),
        "local_reward": float(reward_info.get("episode_local_reward", 0.0)),
        "target_discovery_rate": float(metrics["target_discovery_rate"]),
        "discovered_target_count": float(metrics["discovered_target_count"]),
        "coverage_ratio": float(metrics["coverage_ratio"]),
        "collision_count": float(metrics["collision_count"]),
        "obstacle_violation_count": float(metrics["obstacle_violation_count"]),
        "connectivity_ratio": float(metrics["connectivity_ratio"]),
        "success": 1.0 if bool(metrics["all_targets_discovered"]) else 0.0,
        "episode_length": float(metrics["timestep"]),
    }
    for source_name, output_name in REWARD_TERM_METRIC_NAMES.items():
        summary[output_name] = float(reward_terms.get(source_name, 0.0))
    return summary


def aggregate_report(
    episode_summaries: list[dict[str, float]],
    policy: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-episode summaries into the standard evaluation report schema."""
    if not episode_summaries:
        raise ValueError("episode_summaries must contain at least one episode")

    def mean(key: str) -> float:
        return float(np.mean([summary[key] for summary in episode_summaries]))

    report: dict[str, Any] = {
        "policy": policy,
        "num_episodes": len(episode_summaries),
        "mean_reward": mean("reward"),
        "mean_shared_reward": mean("shared_reward"),
        "mean_local_reward": mean("local_reward"),
        "mean_target_discovery_rate": mean("target_discovery_rate"),
        "mean_discovered_target_count": mean("discovered_target_count"),
        "mean_coverage_ratio": mean("coverage_ratio"),
        "mean_collision_count": mean("collision_count"),
        "mean_obstacle_violation_count": mean("obstacle_violation_count"),
        "mean_connectivity_ratio": mean("connectivity_ratio"),
        "success_rate": mean("success"),
        "mean_episode_length": mean("episode_length"),
        "episodes": episode_summaries,
    }
    for output_name in REWARD_TERM_METRIC_NAMES.values():
        report[f"mean_{output_name}"] = mean(output_name)
    if extra is not None:
        report.update(extra)
    return report


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write a JSON report and create parent directories when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
