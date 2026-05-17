from __future__ import annotations

import numpy as np

from dronewatch.config.schema import EnvConfig, ObservationConfig
from dronewatch.envs.observation_builder import ObservationBuilder
from dronewatch.envs.spaces import OBSERVATION_SIZE, observation_size
from dronewatch.sim import SwarmSimulation


def test_observation_builder_returns_fixed_finite_float_vectors() -> None:
    sim = SwarmSimulation(seed=123)
    builder = ObservationBuilder()

    observations = builder.build(sim.state(), sim.metrics())

    assert len(observations) == sim.num_agents
    assert all(observation.shape == (OBSERVATION_SIZE,) for observation in observations.values())
    assert all(observation.dtype == np.float32 for observation in observations.values())
    assert all(np.isfinite(observation).all() for observation in observations.values())


def test_observation_builder_is_deterministic_for_same_state() -> None:
    sim = SwarmSimulation(seed=123)
    builder = ObservationBuilder()

    first = builder.build(sim.state(), sim.metrics())
    second = builder.build(sim.state(), sim.metrics())

    assert first.keys() == second.keys()
    for agent_id in first:
        np.testing.assert_array_equal(first[agent_id], second[agent_id])


def test_observation_builder_keeps_normalized_values_within_unit_range() -> None:
    builder = ObservationBuilder()
    state = {
        "agents": [
            {"id": 0, "position": [10.0, 10.0], "velocity": [2.0, 0.0]},
            {"id": 1, "position": [25.0, 10.0], "velocity": [-2.0, 0.0]},
        ],
        "targets": [
            {"id": 0, "position": [25.0, 10.0], "discovered": False},
        ],
        "obstacles": [
            {"id": 0, "position": [31.0, 10.0], "radius": 6.0},
        ],
    }
    metrics = {
        "timestep": 200,
        "max_episode_steps": 200,
    }

    observations = builder.build(state, metrics)

    for agent_id, observation in observations.items():
        assert np.all(observation >= -1.0), f"{agent_id} contains values below -1.0: {observation[observation < -1.0]}"
        assert np.all(observation <= 1.0), f"{agent_id} contains values above 1.0: {observation[observation > 1.0]}"


def test_observation_builder_uses_configured_shape() -> None:
    config = EnvConfig(
        num_agents=2,
        observation=ObservationConfig(
            max_visible_agents=1,
            max_visible_targets=1,
            max_visible_obstacles=1,
            include_communication_summary=False,
        ),
    )
    sim = SwarmSimulation(seed=123, config=config.to_rust_config_dict())
    builder = ObservationBuilder(config)

    observations = builder.build(sim.state(), sim.metrics())

    assert len(observations) == 2
    assert all(observation.shape == (observation_size(config),) for observation in observations.values())
