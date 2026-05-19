from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dronewatch.config.schema import (
    AgentConfig,
    AgentDefaults,
    DroneWatchConfig,
    DroneWatchEvaluationConfig,
    DroneWatchRandomPolicyConfig,
    EnvConfig,
    ObservationDefaults,
    ObstacleDefaults,
    ProjectConfig,
    RewardWeights,
    SwarmSearchEnvConfig,
    WorldDefaults,
)


def test_default_models_construct_with_expected_values() -> None:
    assert WorldDefaults().width == 100.0
    assert AgentDefaults().count == 16
    assert ObservationDefaults().max_visible_agents == 5
    assert ObstacleDefaults().expected_max_radius == 6.0
    assert RewardWeights().target_discovered == 5.0
    training_config = DroneWatchConfig()
    evaluation_config = DroneWatchEvaluationConfig()
    random_policy_config = DroneWatchRandomPolicyConfig()
    assert training_config.env.simulation.agents.count == 16
    assert not hasattr(training_config, "evaluation")
    assert not hasattr(training_config, "baseline")
    assert evaluation_config.evaluation.checkpoint is None
    assert random_policy_config.random_policy.episodes == 1
    assert EnvConfig().agents.count == 16
    assert not hasattr(EnvConfig(), "observation")


def test_default_models_are_frozen() -> None:
    defaults = AgentDefaults()

    with pytest.raises(ValidationError):
        defaults.count = 8  # type: ignore[misc]


def test_default_models_reject_invalid_numeric_values() -> None:
    with pytest.raises(ValidationError):
        WorldDefaults(width=0.0)
    with pytest.raises(ValidationError):
        AgentDefaults(max_speed=-1.0)
    with pytest.raises(ValidationError):
        ObservationDefaults(max_visible_targets=0)
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
