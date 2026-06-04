"""Shared RLlib training progress and checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dronewatch.training.callbacks import (
    LEARNER_METRIC_KEYS,
    REWARD_METRIC_KEYS,
    TASK_METRIC_KEYS,
)
from dronewatch.training.rllib_config import SHARED_POLICY_ID


def save_checkpoint(algorithm: Any, checkpoint_dir: Path) -> str:
    """Save an RLlib Algorithm checkpoint and return its path as a string."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = algorithm.save(str(checkpoint_dir.resolve()))
    if hasattr(saved, "checkpoint") and hasattr(saved.checkpoint, "path"):
        return str(saved.checkpoint.path)
    if hasattr(saved, "path"):
        return str(saved.path)
    return str(saved)


def training_progress(iteration: int, result: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact progress summary from an RLlib training result."""
    env_runners = result.get("env_runners", {})
    custom_metrics = result.get("custom_metrics", {})
    progress: dict[str, Any] = {
        "iteration": iteration,
        "episode_return_mean": env_runners.get("episode_return_mean", result.get("episode_reward_mean")),
        "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
    }

    # RLlib's new API stack exposes callback metrics under env_runners
    for metric_name in (*TASK_METRIC_KEYS, *REWARD_METRIC_KEYS, "success_rate"):
        value = env_runners.get(
            f"dronewatch/{metric_name}",
            custom_metrics.get(f"dronewatch/{metric_name}_mean"),
        )
        if value is not None:
            progress[metric_name] = value
    return progress


def learner_progress(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract PPO learner metrics from an RLlib training result."""
    learners = result.get("learners")
    if not isinstance(learners, Mapping):
        return {}

    merged: dict[str, Any] = {}
    aggregate = learners.get("__all_modules__")
    if isinstance(aggregate, Mapping):
        merged.update(aggregate)
    policy = learners.get(SHARED_POLICY_ID)
    if isinstance(policy, Mapping):
        merged.update(policy)

    return {key: merged[key] for key in LEARNER_METRIC_KEYS if key in merged}
