from __future__ import annotations

import numpy as np
import pytest

from dronewatch.config.schema import EnvConfig, RewardWeights, SwarmSearchEnvConfig
from dronewatch.envs import SwarmSearchEnv


def test_env_step_returns_rllib_multi_agent_payload() -> None:
    env = SwarmSearchEnv(seed=123)
    observations, _infos = env.reset(seed=123)
    actions = {agent_id: np.zeros(2, dtype=np.float32) for agent_id in observations}

    next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

    assert set(next_observations) == set(observations)
    assert set(rewards) == set(observations)
    assert set(infos) == set(observations)
    assert "__all__" in terminateds
    assert "__all__" in truncateds
    assert infos["agent_0"]["metrics"]["timestep"] == 1
    assert "events" in infos["agent_0"]
    assert "reward_terms" in infos["agent_0"]
    assert "team_reward" in infos["agent_0"]
    assert "per_agent_reward" in infos["agent_0"]
    assert next(iter(rewards.values())) == infos["agent_0"]["per_agent_reward"]
    assert sum(rewards.values()) == pytest.approx(infos["agent_0"]["team_reward"])
    assert set(infos["agent_0"]["reward_terms"]) == {
        "target_discovery",
        "coverage",
        "agent_collision",
        "obstacle_collision",
        "step_penalty",
        "remaining_targets",
        "success_bonus",
        "visible_target_approach",
    }
    assert "reward_mode" not in infos["agent_0"]
    assert isinstance(terminateds["__all__"], bool)
    assert isinstance(truncateds["__all__"], bool)


def test_env_step_rejects_missing_and_non_finite_actions() -> None:
    env = SwarmSearchEnv(seed=123)
    observations, _infos = env.reset(seed=123)

    with pytest.raises(ValueError, match="missing actions"):
        env.step({"agent_0": np.zeros(2, dtype=np.float32)})

    bad_actions = {agent_id: np.zeros(2, dtype=np.float32) for agent_id in observations}
    bad_actions["agent_0"] = np.array([np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        env.step(bad_actions)


def test_env_step_returns_agent_local_rewards() -> None:
    env = SwarmSearchEnv(
        env_config=SwarmSearchEnvConfig(
            simulation=EnvConfig(
                max_episode_steps=5,
                agents={
                    "count": 2,
                    "max_speed": 1.0,
                    "collision_radius": 0.25,
                    "sensing_radius": 5.0,
                    "communication_radius": 5.0,
                },
                targets={"count": 1, "discovery_radius": 1.0},
                obstacles={"count": 0, "min_radius": 1.0, "max_radius": 1.0},
            ),
            reward=RewardWeights(
                target_discovered=5.0,
                new_coverage_cell=0.0,
                agent_collision=0.0,
                obstacle_collision=0.0,
                step_penalty=0.0,
                success_bonus=0.0,
                remaining_target_penalty=0.0,
                visible_target_approach=0.0,
            ),
        ),
        seed=123,
    )

    previous_state = {
        "agents": [
            {"id": 0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]},
            {"id": 1, "position": [4.0, 0.0], "velocity": [0.0, 0.0]},
        ],
        "targets": [{"id": 0, "position": [1.0, 0.0], "discovered": False}],
        "obstacles": [],
    }
    next_state = {
        "agents": [
            {"id": 0, "position": [1.0, 0.0], "velocity": [1.0, 0.0]},
            {"id": 1, "position": [4.0, 0.0], "velocity": [0.0, 0.0]},
        ],
        "targets": [{"id": 0, "position": [1.0, 0.0], "discovered": True}],
        "obstacles": [],
    }

    class StubSimulation:
        def __init__(self) -> None:
            self._state = previous_state

        def state(self) -> dict[str, object]:
            return self._state

        def step(self, actions: object) -> dict[str, object]:
            self._state = next_state
            return {
                "events": {
                    "targets_discovered": 1,
                    "agent_collisions": 0,
                    "obstacle_violations": 0,
                    "new_coverage_cells": 0,
                },
                "metrics": {
                    "timestep": 1,
                    "max_episode_steps": 5,
                    "target_count": 1,
                    "discovered_target_count": 1,
                    "target_discovery_rate": 1.0,
                    "coverage_ratio": 0.0,
                    "covered_cells": 0,
                    "total_coverage_cells": 1,
                    "collision_count": 0,
                    "obstacle_violation_count": 0,
                    "connectivity_ratio": 0.0,
                    "average_communication_neighbors": 0.0,
                    "largest_connected_component_size": 1,
                    "communication_edge_count": 0,
                    "done": True,
                    "all_targets_discovered": True,
                    "horizon_reached": False,
                },
                "state": next_state,
            }

    env._simulation = StubSimulation()
    actions = {"agent_0": np.zeros(2, dtype=np.float32), "agent_1": np.zeros(2, dtype=np.float32)}

    _next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

    assert rewards["agent_0"] > rewards["agent_1"]
    assert sum(rewards.values()) == pytest.approx(infos["agent_0"]["team_reward"])
    assert infos["agent_0"]["local_reward_terms"]["target_discovery"] == 5.0
    assert infos["agent_1"]["local_reward_terms"]["target_discovery"] == 0.0
    assert "reward_mode" not in infos["agent_0"]
    assert terminateds["__all__"] is True
    assert truncateds["__all__"] is False
