from __future__ import annotations

import numpy as np

from dronewatch.config.schema import (
    AgentConfig,
    EnvConfig,
    ObservationConfig,
    SwarmSearchEnvConfig,
)
from dronewatch.envs.observation_builder import ObservationBuilder
from dronewatch.envs.spaces import observation_size
from dronewatch.sim import SwarmSimulation


def test_observation_builder_returns_fixed_finite_float_vectors() -> None:
    sim = SwarmSimulation(seed=123)
    builder = ObservationBuilder()
    default_observation_size = observation_size(ObservationConfig())

    observations = builder.build(sim.state(), sim.metrics())

    assert len(observations) == sim.num_agents
    assert all(observation.shape == (default_observation_size,) for observation in observations.values())
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


def test_observation_builder_includes_normalized_agent_id() -> None:
    config = SwarmSearchEnvConfig(simulation=EnvConfig(agents=AgentConfig(count=4)))
    state = {
        "agents": [
            {"id": 0, "position": [10.0, 10.0], "velocity": [0.0, 0.0]},
            {"id": 1, "position": [10.0, 10.0], "velocity": [0.0, 0.0]},
            {"id": 2, "position": [10.0, 10.0], "velocity": [0.0, 0.0]},
            {"id": 3, "position": [10.0, 10.0], "velocity": [0.0, 0.0]},
        ],
        "targets": [],
        "obstacles": [],
    }
    metrics = {"timestep": 0, "max_episode_steps": 200}
    builder = ObservationBuilder(config.simulation, config.observation)

    observations = builder.build(state, metrics)

    assert observations["agent_0"][0] == 0.0
    assert observations["agent_1"][0] == np.float32(1.0 / 3.0)
    assert observations["agent_2"][0] == np.float32(2.0 / 3.0)
    assert observations["agent_3"][0] == 1.0


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
    config = SwarmSearchEnvConfig(
        simulation=EnvConfig(agents=AgentConfig(count=2)),
        observation=ObservationConfig(
            max_visible_agents=1,
            max_visible_targets=1,
            max_visible_obstacles=1,
            include_communication_summary=False,
        ),
    )
    sim = SwarmSimulation(seed=123, config=config.simulation)
    builder = ObservationBuilder(config.simulation, config.observation)

    observations = builder.build(sim.state(), sim.metrics())

    assert len(observations) == 2
    assert all(observation.shape == (observation_size(config.observation),) for observation in observations.values())
