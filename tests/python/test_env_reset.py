from __future__ import annotations

import numpy as np

from dronewatch.envs import SwarmSearchEnv
from dronewatch.envs.spaces import OBSERVATION_SIZE


def test_env_reset_returns_observations_and_infos_for_all_agents() -> None:
    env = SwarmSearchEnv({"seed": 123})

    observations, infos = env.reset(seed=123)

    assert len(observations) == 16
    assert len(infos) == 16
    assert set(observations) == {f"agent_{index}" for index in range(16)}
    assert set(infos) == set(observations)
    assert all(observation.shape == (OBSERVATION_SIZE,) for observation in observations.values())
    assert all(observation.dtype == np.float32 for observation in observations.values())
    assert all(np.isfinite(observation).all() for observation in observations.values())
    assert all(env.observation_space.contains(observation) for observation in observations.values())
    assert infos["agent_0"]["metrics"]["timestep"] == 0


def test_env_rejects_unsupported_non_default_config() -> None:
    try:
        SwarmSearchEnv({"num_agents": 8})
    except ValueError as error:
        assert "unsupported Phase 2 env_config keys" in str(error)
    else:
        raise AssertionError("expected unsupported env_config to raise ValueError")
