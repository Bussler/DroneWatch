"""Random policy baseline for SwarmSearch2D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dronewatch.config.loader import (
    load_config,
    resolved_config_path,
    save_resolved_config,
)
from dronewatch.config.schema import SwarmSearchEnvConfig
from dronewatch.envs import SwarmSearchEnv
from dronewatch.evaluation import aggregate_report, episode_summary, write_json_report
from dronewatch.rendering import SimulationFrame
from dronewatch.rendering.render_episode import render_episode_gif


class RandomPolicy:
    """Uniform random policy over the per-agent continuous action space."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a reproducible random policy when `seed` is provided."""
        self._rng = np.random.default_rng(seed)

    def compute_action(self) -> np.ndarray:
        """Sample one continuous `[dx, dy]` action from the valid action range."""
        return self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


def run_random_policy(
    episodes: int = 1,
    seed: int = 42,
    report_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    render: bool = False,
    render_stride: int = 4,
    env_config: SwarmSearchEnvConfig | None = None,
    render_fps: int = 12,
) -> dict[str, Any]:
    """Run a random policy baseline and optionally write a report and GIF."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if render_stride <= 0:
        raise ValueError("render_stride must be greater than zero")

    episode_summaries: list[dict[str, Any]] = []
    first_episode_frames: list[SimulationFrame] = []
    env_config = env_config or SwarmSearchEnvConfig()

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = SwarmSearchEnv(
            env_config.model_copy(update={"seed": episode_seed}).model_dump(mode="json"),
        )
        policy = RandomPolicy(seed=episode_seed)
        observations, _infos = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        final_metrics: dict[str, Any] = {}

        if render and episode_index == 0:
            state_snapshot, metrics_snapshot = env.snapshot()
            first_episode_frames.append(SimulationFrame.from_snapshots(state_snapshot, metrics_snapshot))

        while not done:
            actions = {agent_id: policy.compute_action() for agent_id in observations}
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode_reward += float(next(iter(rewards.values())))
            done = bool(terminateds["__all__"] or truncateds["__all__"])
            final_metrics = dict(next(iter(infos.values()))["metrics"])

            should_capture = (
                render and episode_index == 0 and (done or int(final_metrics["timestep"]) % render_stride == 0)
            )
            if should_capture:
                state_snapshot, _metrics_snapshot = env.snapshot()
                first_episode_frames.append(SimulationFrame.from_snapshots(state_snapshot, final_metrics))

        episode_summaries.append(episode_summary(episode_reward, final_metrics))

    report = aggregate_report(episode_summaries, policy="random")
    if report_path is not None:
        write_json_report(report_path, report)
    if render and gif_path is not None:
        render_episode_gif(first_episode_frames, gif_path, fps=render_fps, env_config=env_config.simulation)
    return report


def main() -> None:
    """Command-line entry point for the random policy baseline."""
    parser = argparse.ArgumentParser(description="Run the DroneWatch random policy baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    resolved_path = save_resolved_config(
        config,
        resolved_config_path(Path(config.baseline.random.report_path).parent, config),
    )

    report = run_random_policy(
        episodes=config.baseline.random.episodes,
        seed=config.random_seed() or 0,
        report_path=config.baseline.random.report_path,
        gif_path=config.baseline.random.gif_path,
        render=config.baseline.random.render,
        render_stride=config.baseline.random.render_stride,
        env_config=config.env,
        render_fps=config.rendering.fps,
    )
    report["resolved_config_path"] = str(resolved_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
