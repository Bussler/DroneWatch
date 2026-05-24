"""Cooperative reward calculation for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot
from typing import Any

from dronewatch.config.schema import RewardWeights

RewardTerms = dict[str, float]


def calculate_reward_terms(
    events: Mapping[str, Any],
    metrics: Mapping[str, Any],
    visible_target_approach: float,
    weights: RewardWeights,
) -> RewardTerms:
    """Calculate named shared reward terms from step events and post-step metrics."""
    remaining_targets = max(
        float(metrics.get("target_count", 0.0)) - float(metrics.get("discovered_target_count", 0.0)),
        0.0,
    )
    return {
        "target_discovery": float(events.get("targets_discovered", 0.0)) * weights.target_discovered,
        "coverage": float(events.get("new_coverage_cells", 0.0)) * weights.new_coverage_cell,
        "agent_collision": float(events.get("agent_collisions", 0.0)) * weights.agent_collision,
        "obstacle_collision": float(events.get("obstacle_violations", 0.0)) * weights.obstacle_collision,
        "step_penalty": weights.step_penalty,
        "remaining_targets": remaining_targets * weights.remaining_target_penalty,
        "success_bonus": weights.success_bonus if bool(metrics.get("all_targets_discovered", False)) else 0.0,
        "visible_target_approach": float(visible_target_approach) * weights.visible_target_approach,
    }


def calculate_team_reward(
    events: Mapping[str, Any],
    metrics: Mapping[str, Any],
    visible_target_approach: float,
    weights: RewardWeights,
) -> float:
    """Calculate the shared cooperative reward from named reward terms."""
    return float(sum(calculate_reward_terms(events, metrics, visible_target_approach, weights).values()))


def calculate_visible_target_approach(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    sensing_radius: float,
    max_displacement: float,
) -> float:
    """Return mean normalized progress toward each agent's nearest visible undiscovered target."""
    if max_displacement <= 0.0:
        raise ValueError("max_displacement must be positive")

    next_agents = {int(agent["id"]): agent for agent in next_state["agents"]}
    previous_targets = [target for target in previous_state["targets"] if not bool(target["discovered"])]
    approach_values: list[float] = []

    for previous_agent in previous_state["agents"]:
        agent_id = int(previous_agent["id"])
        next_agent = next_agents.get(agent_id)
        if next_agent is None:
            continue

        previous_position = _point2(previous_agent["position"])
        visible_targets = [
            (distance, int(target["id"]), target)
            for target in previous_targets
            if (distance := _distance(previous_position, _point2(target["position"]))) <= sensing_radius
        ]
        if not visible_targets:
            continue

        previous_distance, _target_id, target = min(visible_targets, key=lambda item: (item[0], item[1]))
        next_distance = _distance(_point2(next_agent["position"]), _point2(target["position"]))
        normalized_progress = (previous_distance - next_distance) / max_displacement
        approach_values.append(float(max(-1.0, min(1.0, normalized_progress))))

    if not approach_values:
        return 0.0
    return float(sum(approach_values) / len(approach_values))


def _point2(value: Any) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
