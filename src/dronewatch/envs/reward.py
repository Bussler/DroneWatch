"""Cooperative reward calculation for SwarmSearch2D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot
from typing import Any, Literal

from dronewatch.config.schema import RewardWeights

RewardTerms = dict[str, float]
AgentRewardTerms = dict[int, RewardTerms]


@dataclass(frozen=True)
class RewardTermSpec:
    """Describe one reward term and whether it comes from shared or local attribution."""

    name: str
    scope: Literal["shared", "local"]


@dataclass(frozen=True)
class RewardContext:
    """Bundle geometry and weights for local reward attribution."""

    sensing_radius: float
    discovery_radius: float
    collision_radius: float
    max_displacement: float
    weights: RewardWeights


REWARD_TERMS = (
    RewardTermSpec("target_discovery", "local"),
    RewardTermSpec("coverage", "shared"),
    RewardTermSpec("agent_collision", "local"),
    RewardTermSpec("obstacle_collision", "local"),
    RewardTermSpec("step_penalty", "shared"),
    RewardTermSpec("remaining_targets", "shared"),
    RewardTermSpec("success_bonus", "shared"),
    RewardTermSpec("visible_target_approach", "local"),
)

REWARD_TERM_KEYS = tuple(spec.name for spec in REWARD_TERMS)
LOCAL_REWARD_TERM_KEYS = tuple(spec.name for spec in REWARD_TERMS if spec.scope == "local")
SHARED_REWARD_TERM_KEYS = tuple(spec.name for spec in REWARD_TERMS if spec.scope == "shared")
REWARD_TERM_METRIC_NAMES = {name: f"reward_term_{name}" for name in REWARD_TERM_KEYS}


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
    context: RewardContext,
) -> AgentRewardTerms:
    """Attribute local reward terms to individual agents while preserving total reward mass."""
    next_agents = _agents_by_id(next_state)
    agent_terms = {agent_id: _empty_local_reward_terms() for agent_id in next_agents}

    _attribute_target_discovery(
        previous_state,
        next_state,
        next_agents,
        context.discovery_radius,
        context.weights,
        agent_terms,
    )
    _attribute_agent_collisions(next_agents, context.collision_radius, context.weights, agent_terms)
    _attribute_obstacle_violations(next_state, context.collision_radius, context.weights, agent_terms)

    approach_by_agent = calculate_visible_target_approach_by_agent(
        previous_state=previous_state,
        next_state=next_state,
        context=context,
    )
    visible_agent_count = len(approach_by_agent)
    if visible_agent_count > 0:
        for agent_id, normalized_progress in approach_by_agent.items():
            agent_terms[agent_id]["visible_target_approach"] += (
                normalized_progress / visible_agent_count
            ) * context.weights.visible_target_approach

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
    return _empty_terms(REWARD_TERM_KEYS)


def combine_reward_terms(shared_terms: RewardTerms, local_terms: RewardTerms) -> RewardTerms:
    """Return the full reward-term mapping in the established key order."""
    combined: RewardTerms = {}
    for spec in REWARD_TERMS:
        source_terms = shared_terms if spec.scope == "shared" else local_terms
        combined[spec.name] = float(source_terms.get(spec.name, 0.0))
    return combined


def calculate_visible_target_approach_by_agent(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    context: RewardContext,
) -> dict[int, float]:
    """Return normalized target-approach progress for each agent with a visible undiscovered target."""
    if context.max_displacement <= 0.0:
        raise ValueError("max_displacement must be positive")

    next_agents = _agents_by_id(next_state)
    previous_targets = _undiscovered_targets(previous_state)
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
            if (distance := _distance(previous_position, _point2(target["position"]))) <= context.sensing_radius
        ]
        if not visible_targets:
            continue

        previous_distance, _target_id, target = min(visible_targets, key=lambda item: (item[0], item[1]))
        next_distance = _distance(_point2(next_agent["position"]), _point2(target["position"]))
        normalized_progress = (previous_distance - next_distance) / context.max_displacement
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
    """Attribute target discovery rewards to agents within discovery radius of newly discovered targets."""
    previous_targets = _targets_by_id(previous_state)

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
    """Attribute agent collision penalties to agents within collision radius of each other."""
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
    """Attribute obstacle collision penalties to agents within collision radius of obstacles."""
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
    return _empty_terms(LOCAL_REWARD_TERM_KEYS)


def _empty_terms(keys: tuple[str, ...]) -> RewardTerms:
    return {key: 0.0 for key in keys}


def _agents_by_id(state: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(agent["id"]): agent for agent in state["agents"]}


def _targets_by_id(state: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(target["id"]): target for target in state["targets"]}


def _undiscovered_targets(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [target for target in state["targets"] if not bool(target["discovered"])]


def _point2(value: Any) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
