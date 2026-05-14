from __future__ import annotations

import numpy as np

from dronewatch.envs.observation_builder import ObservationBuilder
from dronewatch.envs.spaces import OBSERVATION_SIZE
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
