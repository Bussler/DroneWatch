from __future__ import annotations

import pytest

from dronewatch.config.schema import EnvConfig, ObstacleConfig, TargetConfig
from dronewatch.sim import SwarmSimulation


def test_rust_world_resets_with_default_config() -> None:
    sim = SwarmSimulation(seed=123)
    state = sim.state()
    metrics = sim.metrics()

    assert len(state["agents"]) == 16
    assert len(state["targets"]) == 20
    assert len(state["obstacles"]) == 8
    assert metrics["timestep"] == 0
    assert 0.0 <= metrics["coverage_ratio"] <= 1.0
    assert 0.0 <= metrics["connectivity_ratio"] <= 1.0


def test_rust_world_steps_and_returns_events_state_and_metrics() -> None:
    sim = SwarmSimulation(seed=123)
    actions = [(1.0, 0.0)] * sim.num_agents

    result = sim.step(actions)

    assert sorted(result) == ["events", "metrics", "state"]
    assert result["metrics"]["timestep"] == 1
    assert len(result["state"]["agents"]) == 16
    assert result["events"]["agent_collisions"] >= 0
    assert result["events"]["obstacle_violations"] >= 0


def test_rust_world_rejects_wrong_action_count() -> None:
    sim = SwarmSimulation(seed=123)

    with pytest.raises(ValueError, match="expected 16 actions"):
        sim.step([(0.0, 0.0)])


def test_scripted_rollout_finishes_with_bounded_metrics() -> None:
    sim = SwarmSimulation(seed=99)

    while not sim.is_done():
        actions = [(1.0, 0.25) if index % 2 == 0 else (-0.25, 1.0) for index in range(sim.num_agents)]
        sim.step(actions)

    metrics = sim.metrics()
    assert metrics["done"] is True
    assert metrics["timestep"] <= metrics["max_episode_steps"]
    assert 0.0 <= metrics["target_discovery_rate"] <= 1.0
    assert 0.0 <= metrics["coverage_ratio"] <= 1.0
    assert 0.0 <= metrics["connectivity_ratio"] <= 1.0


def test_rust_world_accepts_configured_simulation_values() -> None:
    config = EnvConfig(
        num_agents=4,
        max_episode_steps=5,
        targets=TargetConfig(count=3),
        obstacles=ObstacleConfig(count=1, min_radius=2.0, max_radius=3.0),
    )
    sim = SwarmSimulation(seed=123, config=config.to_rust_config_dict())
    state = sim.state()
    metrics = sim.metrics()

    assert len(state["agents"]) == 4
    assert len(state["targets"]) == 3
    assert len(state["obstacles"]) == 1
    assert metrics["max_episode_steps"] == 5
