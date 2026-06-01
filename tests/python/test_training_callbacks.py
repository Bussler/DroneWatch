from __future__ import annotations

from dronewatch.training.callbacks import SwarmSearchMetricsCallback
from dronewatch.training.utils import training_progress


class _FakeEpisode:
    def __init__(self, infos: dict[str, dict[str, object]]) -> None:
        self._infos = infos

    def get_infos(self, index: int) -> dict[str, dict[str, object]]:
        assert index == -1
        return self._infos


class _FakeMetricsLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def log_value(self, key: str, value: float, reduce: str) -> None:
        assert reduce == "mean"
        self.values[key] = value


def test_callback_logs_reward_breakdown_metrics() -> None:
    callback = SwarmSearchMetricsCallback()
    metrics_logger = _FakeMetricsLogger()
    episode = _FakeEpisode(
        {
            "agent_0": {
                "metrics": {
                    "target_discovery_rate": 0.5,
                    "discovered_target_count": 10,
                    "coverage_ratio": 0.7,
                    "collision_count": 2,
                    "obstacle_violation_count": 1,
                    "connectivity_ratio": 0.8,
                    "average_communication_neighbors": 3.0,
                    "timestep": 200,
                    "all_targets_discovered": False,
                },
                "episode_shared_reward": 12.5,
                "episode_local_reward": 4.25,
                "episode_reward_terms": {
                    "target_discovery": 8.0,
                    "coverage": 2.0,
                    "agent_collision": -1.0,
                    "obstacle_collision": -0.5,
                    "step_penalty": -0.2,
                    "remaining_targets": -1.0,
                    "success_bonus": 0.0,
                    "visible_target_approach": 0.95,
                },
            }
        }
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert metrics_logger.values["dronewatch/target_discovery_rate"] == 0.5
    assert metrics_logger.values["dronewatch/shared_reward"] == 12.5
    assert metrics_logger.values["dronewatch/local_reward"] == 4.25
    assert metrics_logger.values["dronewatch/reward_term_target_discovery"] == 8.0
    assert metrics_logger.values["dronewatch/reward_term_agent_collision"] == -1.0
    assert metrics_logger.values["dronewatch/reward_term_visible_target_approach"] == 0.95
    assert metrics_logger.values["dronewatch/success_rate"] == 0.0


def test_training_progress_extracts_reward_breakdown_metrics() -> None:
    progress = training_progress(
        3,
        {
            "env_runners": {
                "episode_return_mean": 1.25,
                "dronewatch/target_discovery_rate": 0.5,
                "dronewatch/shared_reward": 0.75,
                "dronewatch/local_reward": 0.5,
                "dronewatch/reward_term_target_discovery": 1.0,
            },
        },
    )

    assert progress["iteration"] == 3
    assert progress["episode_return_mean"] == 1.25
    assert progress["target_discovery_rate"] == 0.5
    assert progress["shared_reward"] == 0.75
    assert progress["local_reward"] == 0.5
    assert progress["reward_term_target_discovery"] == 1.0
