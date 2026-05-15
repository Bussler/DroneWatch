from __future__ import annotations

import pytest
from pydantic import ValidationError

from dronewatch.envs.spaces import (
    AgentDefaults,
    ObservationDefaults,
    ObstacleDefaults,
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


def test_env_config_uses_pydantic_validation() -> None:
    assert SwarmSearchEnvConfig(seed=None).seed is None
    assert SwarmSearchEnvConfig.model_validate({"seed": 123}).seed == 123

    with pytest.raises(ValidationError):
        SwarmSearchEnvConfig.model_validate({"num_agents": 8})
    with pytest.raises(ValidationError):
        SwarmSearchEnvConfig(seed=-1)
