"""Cooperative reward calculation for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot
from typing import Any

from dronewatch.config.schema import RewardWeights

RewardTerms = dict[str, float]
AgentRewardTerms = dict[int, RewardTerms]

REWARD_TERM_KEYS = (
    "target_discovery",
    "coverage",
    "agent_collision",
    "obstacle_collision",
    "step_penalty",
    "remaining_targets",
    "success_bonus",
    "visible_target_approach",
)

LOCAL_REWARD_TERM_KEYS = (
    "target_discovery",
    "agent_collision",
    "obstacle_collision",
    "visible_target_approach",
)


def calculate_shared_reward_terms(
    events: Mapping[str, Any],
    metrics: Mapping[str, Any],
    weights: RewardWeights,
) -> RewardTerms:
    """Calculate reward terms that are distributed equally across the team."""
    remaining_targets = max(
        float(metrics.get("target_count", 0.0)) - float(metrics.get("discovered_target_count", 0.0)),
        0.0,
    )
    return {
        "coverage": float(events.get("new_coverage_cells", 0.0)) * weights.new_coverage_cell,
        "step_penalty": weights.step_penalty,
        "remaining_targets": remaining_targets * weights.remaining_target_penalty,
        "success_bonus": weights.success_bonus if bool(metrics.get("all_targets_discovered", False)) else 0.0,
    }


def calculate_agent_reward_terms(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    sensing_radius: float,
    discovery_radius: float,
    collision_radius: float,
    max_displacement: float,
    weights: RewardWeights,
) -> AgentRewardTerms:
    """Attribute local reward terms to individual agents while preserving total reward mass."""
    next_agents = {int(agent["id"]): agent for agent in next_state["agents"]}
    agent_terms = {agent_id: _empty_local_reward_terms() for agent_id in next_agents}

    _attribute_target_discovery(previous_state, next_state, next_agents, discovery_radius, weights, agent_terms)
    _attribute_agent_collisions(next_agents, collision_radius, weights, agent_terms)
    _attribute_obstacle_violations(next_state, collision_radius, weights, agent_terms)

    approach_by_agent = calculate_visible_target_approach_by_agent(
        previous_state=previous_state,
        next_state=next_state,
        sensing_radius=sensing_radius,
        max_displacement=max_displacement,
    )
    visible_agent_count = len(approach_by_agent)
    if visible_agent_count > 0:
        for agent_id, normalized_progress in approach_by_agent.items():
            agent_terms[agent_id]["visible_target_approach"] += (
                normalized_progress / visible_agent_count
            ) * weights.visible_target_approach

    return agent_terms


def aggregate_agent_reward_terms(agent_reward_terms: AgentRewardTerms) -> RewardTerms:
    """Aggregate attributed local reward terms across all agents."""
    aggregated = _empty_local_reward_terms()
    for terms in agent_reward_terms.values():
        for key, value in terms.items():
            aggregated[key] += value
    return aggregated


def empty_reward_terms() -> RewardTerms:
    """Return a zero-initialized reward-term mapping covering the full reward surface."""
    return {key: 0.0 for key in REWARD_TERM_KEYS}


def combine_reward_terms(shared_terms: RewardTerms, local_terms: RewardTerms) -> RewardTerms:
    """Return the full reward-term mapping in the established key order."""
    return {
        "target_discovery": float(local_terms.get("target_discovery", 0.0)),
        "coverage": float(shared_terms.get("coverage", 0.0)),
        "agent_collision": float(local_terms.get("agent_collision", 0.0)),
        "obstacle_collision": float(local_terms.get("obstacle_collision", 0.0)),
        "step_penalty": float(shared_terms.get("step_penalty", 0.0)),
        "remaining_targets": float(shared_terms.get("remaining_targets", 0.0)),
        "success_bonus": float(shared_terms.get("success_bonus", 0.0)),
        "visible_target_approach": float(local_terms.get("visible_target_approach", 0.0)),
    }


def calculate_visible_target_approach(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    sensing_radius: float,
    max_displacement: float,
) -> float:
    """Return mean normalized progress toward each agent's nearest visible undiscovered target."""
    approach_by_agent = calculate_visible_target_approach_by_agent(
        previous_state=previous_state,
        next_state=next_state,
        sensing_radius=sensing_radius,
        max_displacement=max_displacement,
    )
    if not approach_by_agent:
        return 0.0
    return float(sum(approach_by_agent.values()) / len(approach_by_agent))


def calculate_visible_target_approach_by_agent(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    sensing_radius: float,
    max_displacement: float,
) -> dict[int, float]:
    """Return normalized target-approach progress for each agent with a visible undiscovered target."""
    if max_displacement <= 0.0:
        raise ValueError("max_displacement must be positive")

    next_agents = {int(agent["id"]): agent for agent in next_state["agents"]}
    previous_targets = [target for target in previous_state["targets"] if not bool(target["discovered"])]
    approach_values: dict[int, float] = {}

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
        approach_values[agent_id] = float(max(-1.0, min(1.0, normalized_progress)))

    return approach_values


def _attribute_target_discovery(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    next_agents: Mapping[int, Mapping[str, Any]],
    discovery_radius: float,
    weights: RewardWeights,
    agent_terms: AgentRewardTerms,
) -> None:
    previous_targets = {int(target["id"]): target for target in previous_state["targets"]}

    for target in next_state["targets"]:
        target_id = int(target["id"])
        if not bool(target["discovered"]):
            continue
        if bool(previous_targets.get(target_id, {}).get("discovered", False)):
            continue

        target_position = _point2(target["position"])
        contributors = [
            agent_id
            for agent_id, agent in next_agents.items()
            if _distance(_point2(agent["position"]), target_position) <= discovery_radius
        ]
        if not contributors:
            continue

        reward_share = weights.target_discovered / len(contributors)
        for agent_id in contributors:
            agent_terms[agent_id]["target_discovery"] += reward_share


def _attribute_agent_collisions(
    next_agents: Mapping[int, Mapping[str, Any]],
    collision_radius: float,
    weights: RewardWeights,
    agent_terms: AgentRewardTerms,
) -> None:
    agent_items = sorted(next_agents.items())
    collision_distance = collision_radius * 2.0
    penalty_share = weights.agent_collision / 2.0

    for left_index, (left_id, left_agent) in enumerate(agent_items):
        left_position = _point2(left_agent["position"])
        for right_id, right_agent in agent_items[left_index + 1 :]:
            right_position = _point2(right_agent["position"])
            if _distance(left_position, right_position) <= collision_distance:
                agent_terms[left_id]["agent_collision"] += penalty_share
                agent_terms[right_id]["agent_collision"] += penalty_share


def _attribute_obstacle_violations(
    next_state: Mapping[str, Any],
    collision_radius: float,
    weights: RewardWeights,
    agent_terms: AgentRewardTerms,
) -> None:
    for agent in next_state["agents"]:
        agent_id = int(agent["id"])
        agent_position = _point2(agent["position"])
        violations = 0
        for obstacle in next_state["obstacles"]:
            obstacle_position = _point2(obstacle["position"])
            if _distance(agent_position, obstacle_position) <= collision_radius + float(obstacle["radius"]):
                violations += 1
        if violations > 0:
            agent_terms[agent_id]["obstacle_collision"] += violations * weights.obstacle_collision


def _empty_local_reward_terms() -> RewardTerms:
    return {key: 0.0 for key in LOCAL_REWARD_TERM_KEYS}


def _point2(value: Any) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
