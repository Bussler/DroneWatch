"""Cooperative reward calculation for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dronewatch.config.schema import RewardWeights

from .spaces import REWARD_WEIGHTS


def calculate_team_reward(
    events: Mapping[str, Any],
    weights: RewardWeights = REWARD_WEIGHTS,
) -> float:
    """Calculate the shared cooperative reward from raw Rust step events.

    The MVP reward includes target discovery, newly covered cells, collision penalties, obstacle
    penalties, and a small step penalty. Communication metrics are intentionally excluded from the
    reward and should remain metrics-only in Phase 2.
    """
    return float(
        float(events.get("targets_discovered", 0)) * weights.target_discovered
        + float(events.get("new_coverage_cells", 0)) * weights.new_coverage_cell
        + float(events.get("agent_collisions", 0)) * weights.agent_collision
        + float(events.get("obstacle_violations", 0)) * weights.obstacle_collision
        + weights.step_penalty
    )
