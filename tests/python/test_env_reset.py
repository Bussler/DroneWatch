from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dronewatch.config.schema import (
    AgentConfig,
    EnvConfig,
    ObservationConfig,
    SwarmSearchEnvConfig,
)
from dronewatch.envs import SwarmSearchEnv
from dronewatch.envs.spaces import (
    OBSERVATION_SIZE,
    observation_size,
)


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
    with pytest.raises(ValidationError, match="num_agents"):
        SwarmSearchEnv({"num_agents": 8})


def test_env_accepts_pydantic_config_and_none_seed() -> None:
    SwarmSearchEnv({"seed": None})
    SwarmSearchEnv(SwarmSearchEnvConfig(seed=None))


def test_env_accepts_configured_agent_count_and_observation_shape() -> None:
    env_config = SwarmSearchEnvConfig(
        simulation=EnvConfig(agents=AgentConfig(count=4)),
        observation=ObservationConfig(
            max_visible_agents=2,
            max_visible_targets=2,
            max_visible_obstacles=1,
            include_communication_summary=False,
        ),
    )
    env = SwarmSearchEnv(env_config.model_copy(update={"seed": 7}).model_dump(mode="json"))

    observations, infos = env.reset(seed=7)

    assert len(observations) == 4
    assert len(infos) == 4
    assert env.observation_space.shape == (observation_size(env_config.observation),)
    assert all(
        observation.shape == (observation_size(env_config.observation),) for observation in observations.values()
    )
