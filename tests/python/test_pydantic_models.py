from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dronewatch.config.schema import (
    AgentConfig,
    DroneWatchConfig,
    DroneWatchEvaluationConfig,
    DroneWatchRandomPolicyConfig,
    EnvConfig,
    ObservationConfig,
    ProjectConfig,
    RewardWeights,
    SwarmSearchEnvConfig,
    WorldConfig,
)


def test_config_models_construct_with_expected_values() -> None:
    env_config = EnvConfig()
    observation_config = ObservationConfig()
    assert env_config.world.width == 100.0
    assert env_config.max_episode_steps == 200
    assert env_config.agents.count == 16
    assert env_config.agents.max_speed == 2.0
    assert env_config.obstacles.max_radius == 6.0
    assert observation_config.max_visible_agents == 5
    assert RewardWeights().target_discovered == 5.0
    training_config = DroneWatchConfig()
    evaluation_config = DroneWatchEvaluationConfig()
    random_policy_config = DroneWatchRandomPolicyConfig()
    assert training_config.env.simulation.agents.count == 16
    assert not hasattr(training_config, "evaluation")
    assert not hasattr(training_config, "baseline")
    assert evaluation_config.evaluation.checkpoint is None
    assert random_policy_config.random_policy.episodes == 1
    assert not hasattr(env_config, "observation")


def test_default_models_are_frozen() -> None:
    defaults = AgentConfig()

    with pytest.raises(ValidationError):
        defaults.count = 8  # type: ignore[misc]


def test_default_models_reject_invalid_numeric_values() -> None:
    with pytest.raises(ValidationError):
        WorldConfig(width=0.0)
    with pytest.raises(ValidationError):
        AgentConfig(max_speed=-1.0)
    with pytest.raises(ValidationError):
        ObservationConfig(max_visible_targets=0)
    with pytest.raises(ValidationError):
        RewardWeights(agent_collision=0.25)
    with pytest.raises(ValidationError):
        DroneWatchConfig.model_validate({"evaluation": {}})


def test_env_config_uses_pydantic_validation() -> None:
    assert ProjectConfig(seed=None).seed is None
    assert ProjectConfig(seed=123).seed == 123
    configured_env = SwarmSearchEnvConfig(simulation=EnvConfig(agents=AgentConfig(count=4)))
    assert configured_env.simulation.agents.count == 4

    with pytest.raises(ValidationError):
        ProjectConfig(seed=-1)
    with pytest.raises(ValidationError):
        SwarmSearchEnvConfig.model_validate({"seed": 123})
    with pytest.raises(ValidationError):
        SwarmSearchEnvConfig.model_validate({"num_agents": 8})
    with pytest.raises(ValidationError):
        SwarmSearchEnvConfig.model_validate({"agents": AgentConfig().model_dump(mode="json")})


def test_env_config_serializes_to_rust_json() -> None:
    config = EnvConfig(agents=AgentConfig(count=4))

    payload = json.loads(config.model_dump_json())

    assert sorted(payload) == ["agents", "coverage", "max_episode_steps", "obstacles", "targets", "world"]
    assert payload["agents"]["count"] == 4
    assert payload["world"]["width"] == 100.0
