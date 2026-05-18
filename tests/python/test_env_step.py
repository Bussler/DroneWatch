from __future__ import annotations

import numpy as np
import pytest

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
    assert len(set(rewards.values())) == 1
    assert infos["agent_0"]["metrics"]["timestep"] == 1
    assert "events" in infos["agent_0"]
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
