"""RLlib callbacks for DroneWatch training metrics."""

from __future__ import annotations

from typing import Any

from ray.rllib.callbacks.callbacks import RLlibCallback

TASK_METRIC_KEYS = {
    "target_discovery_rate": "target_discovery_rate",
    "discovered_target_count": "discovered_target_count",
    "coverage_ratio": "coverage_ratio",
    "collision_count": "collision_count",
    "obstacle_violation_count": "obstacle_violation_count",
    "connectivity_ratio": "connectivity_ratio",
    "average_communication_neighbors": "average_communication_neighbors",
    "episode_length": "timestep",
}

REWARD_METRIC_KEYS = (
    "shared_reward",
    "local_reward",
    "reward_term_target_discovery",
    "reward_term_coverage",
    "reward_term_agent_collision",
    "reward_term_obstacle_collision",
    "reward_term_step_penalty",
    "reward_term_remaining_targets",
    "reward_term_success_bonus",
    "reward_term_visible_target_approach",
)

LEARNER_METRIC_KEYS = (
    "policy_loss",
    "vf_loss",
    "vf_loss_unclipped",
    "total_loss",
    "entropy",
    "mean_kl_loss",
    "vf_explained_var",
    "curr_entropy_coeff",
    "curr_kl_coeff",
)


class SwarmSearchMetricsCallback(RLlibCallback):
    """Log simulator task metrics at episode end."""

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_runner: Any,
        metrics_logger: Any,
        env: Any,
        env_index: int,
        rl_module: Any,
        **kwargs: Any,
    ) -> None:
        """Record final simulator metrics through RLlib's metrics logger."""
        del env_runner, env, env_index, rl_module, kwargs
        final_infos = episode.get_infos(-1)
        if not final_infos:
            return

        first_info = next(iter(final_infos.values()))
        metrics = first_info.get("metrics", {})
        for output_name, source_name in TASK_METRIC_KEYS.items():
            if source_name in metrics:
                metrics_logger.log_value(f"dronewatch/{output_name}", float(metrics[source_name]), reduce="mean")

        episode_shared_reward = first_info.get("episode_shared_reward")
        if episode_shared_reward is not None:
            metrics_logger.log_value("dronewatch/shared_reward", float(episode_shared_reward), reduce="mean")

        episode_local_reward = first_info.get("episode_local_reward")
        if episode_local_reward is not None:
            metrics_logger.log_value("dronewatch/local_reward", float(episode_local_reward), reduce="mean")

        reward_terms = first_info.get("episode_reward_terms", {})
        reward_term_metric_names = {
            "target_discovery": "reward_term_target_discovery",
            "coverage": "reward_term_coverage",
            "agent_collision": "reward_term_agent_collision",
            "obstacle_collision": "reward_term_obstacle_collision",
            "step_penalty": "reward_term_step_penalty",
            "remaining_targets": "reward_term_remaining_targets",
            "success_bonus": "reward_term_success_bonus",
            "visible_target_approach": "reward_term_visible_target_approach",
        }
        for source_name, output_name in reward_term_metric_names.items():
            if source_name in reward_terms:
                metrics_logger.log_value(f"dronewatch/{output_name}", float(reward_terms[source_name]), reduce="mean")

        success = 1.0 if bool(metrics.get("all_targets_discovered", False)) else 0.0
        metrics_logger.log_value("dronewatch/success_rate", success, reduce="mean")
