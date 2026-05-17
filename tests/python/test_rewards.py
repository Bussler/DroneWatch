from __future__ import annotations

from dronewatch.config.schema import RewardWeights
from dronewatch.envs.reward import calculate_team_reward


def test_calculates_cooperative_reward_from_step_events() -> None:
    reward = calculate_team_reward(
        {
            "targets_discovered": 2,
            "new_coverage_cells": 10,
            "agent_collisions": 1,
            "obstacle_violations": 1,
            "connectivity_ratio": 0.0,
        }
    )

    assert reward == 2 * 5.0 + 10 * 0.02 - 0.25 - 0.5 - 0.001


def test_connectivity_is_not_a_reward_term() -> None:
    low_connectivity = calculate_team_reward({"connectivity_ratio": 0.0})
    high_connectivity = calculate_team_reward({"connectivity_ratio": 1.0})

    assert low_connectivity == high_connectivity == -0.001


def test_reward_weights_are_configurable() -> None:
    reward = calculate_team_reward(
        {"targets_discovered": 1, "new_coverage_cells": 1},
        RewardWeights(target_discovered=2.0, new_coverage_cell=0.5, step_penalty=0.0),
    )

    assert reward == 2.5
