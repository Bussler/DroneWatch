from __future__ import annotations

from dronewatch.config.schema import RewardWeights
from dronewatch.envs.reward import (
    calculate_reward_terms,
    calculate_team_reward,
    calculate_visible_target_approach,
)


def _metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "target_count": 20,
        "discovered_target_count": 20,
        "all_targets_discovered": False,
    }
    metrics.update(overrides)
    return metrics


def test_calculates_cooperative_reward_from_step_events() -> None:
    reward = calculate_team_reward(
        {
            "targets_discovered": 2,
            "new_coverage_cells": 10,
            "agent_collisions": 1,
            "obstacle_violations": 1,
            "connectivity_ratio": 0.0,
        },
        _metrics(),
        0.0,
        RewardWeights(),
    )

    assert reward == 2 * 5.0 + 10 * 0.02 - 0.25 - 0.5 - 0.001


def test_connectivity_is_not_a_reward_term() -> None:
    weights = RewardWeights()
    low_connectivity = calculate_team_reward({"connectivity_ratio": 0.0}, _metrics(), 0.0, weights)
    high_connectivity = calculate_team_reward({"connectivity_ratio": 1.0}, _metrics(), 0.0, weights)

    assert low_connectivity == high_connectivity == -0.001


def test_reward_weights_are_configurable() -> None:
    reward = calculate_team_reward(
        {"targets_discovered": 1, "new_coverage_cells": 1},
        _metrics(),
        0.0,
        RewardWeights(target_discovered=2.0, new_coverage_cell=0.5, step_penalty=0.0),
    )

    assert reward == 2.5


def test_reward_terms_include_success_remaining_and_approach_shaping() -> None:
    terms = calculate_reward_terms(
        {"targets_discovered": 1},
        _metrics(discovered_target_count=18, all_targets_discovered=True),
        visible_target_approach=0.5,
        weights=RewardWeights(
            target_discovered=2.0,
            step_penalty=0.0,
            success_bonus=10.0,
            remaining_target_penalty=-0.25,
            visible_target_approach=0.4,
        ),
    )

    assert terms == {
        "target_discovery": 2.0,
        "coverage": 0.0,
        "agent_collision": -0.0,
        "obstacle_collision": -0.0,
        "step_penalty": 0.0,
        "remaining_targets": -0.5,
        "success_bonus": 10.0,
        "visible_target_approach": 0.2,
    }


def test_visible_target_approach_rewards_progress_toward_nearest_visible_target() -> None:
    previous_state = {
        "agents": [{"id": 0, "position": [0.0, 0.0]}],
        "targets": [{"id": 0, "position": [10.0, 0.0], "discovered": False}],
    }
    next_state = {
        "agents": [{"id": 0, "position": [2.0, 0.0]}],
        "targets": [{"id": 0, "position": [10.0, 0.0], "discovered": False}],
    }

    approach = calculate_visible_target_approach(
        previous_state,
        next_state,
        sensing_radius=15.0,
        max_displacement=2.0,
    )

    assert approach == 1.0


def test_visible_target_approach_penalizes_moving_away() -> None:
    previous_state = {
        "agents": [{"id": 0, "position": [2.0, 0.0]}],
        "targets": [{"id": 0, "position": [10.0, 0.0], "discovered": False}],
    }
    next_state = {
        "agents": [{"id": 0, "position": [0.0, 0.0]}],
        "targets": [{"id": 0, "position": [10.0, 0.0], "discovered": False}],
    }

    approach = calculate_visible_target_approach(
        previous_state,
        next_state,
        sensing_radius=15.0,
        max_displacement=2.0,
    )

    assert approach == -1.0


def test_visible_target_approach_ignores_invisible_and_discovered_targets() -> None:
    previous_state = {
        "agents": [{"id": 0, "position": [0.0, 0.0]}],
        "targets": [
            {"id": 0, "position": [10.0, 0.0], "discovered": True},
            {"id": 1, "position": [20.0, 0.0], "discovered": False},
        ],
    }
    next_state = {
        "agents": [{"id": 0, "position": [2.0, 0.0]}],
        "targets": [
            {"id": 0, "position": [10.0, 0.0], "discovered": True},
            {"id": 1, "position": [20.0, 0.0], "discovered": False},
        ],
    }

    approach = calculate_visible_target_approach(
        previous_state,
        next_state,
        sensing_radius=15.0,
        max_displacement=2.0,
    )

    assert approach == 0.0
