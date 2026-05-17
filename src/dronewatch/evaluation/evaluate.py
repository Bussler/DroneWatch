"""Evaluate a trained RLlib PPO checkpoint on SwarmSearch2D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy

from dronewatch.config.loader import (
    load_config,
    resolved_config_path,
    save_resolved_config,
)
from dronewatch.config.schema import EnvConfig
from dronewatch.envs import SwarmSearchEnv
from dronewatch.evaluation.reporting import (
    aggregate_report,
    episode_summary,
    write_json_report,
)
from dronewatch.rendering import SimulationFrame, render_episode_gif
from dronewatch.training.rllib_config import SHARED_POLICY_ID, register_swarm_search_env


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    episodes: int = 10,
    seed: int = 42,
    report_path: str | Path | None = None,
    model: str | None = None,
    render: bool = False,
    gif_path: str | Path | None = None,
    render_stride: int = 4,
    env_config: EnvConfig | None = None,
    render_fps: int = 12,
) -> dict[str, Any]:
    """Evaluate a checkpoint and optionally write a JSON report."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    register_swarm_search_env()
    checkpoint_path = Path(checkpoint).resolve()
    algorithm = Algorithm.from_checkpoint(str(checkpoint_path))
    try:
        report = evaluate_algorithm(
            algorithm=algorithm,
            episodes=episodes,
            seed=seed,
            checkpoint=str(checkpoint_path),
            model=model,
            render=render,
            gif_path=gif_path,
            render_stride=render_stride,
            env_config=env_config,
            render_fps=render_fps,
        )
    finally:
        algorithm.stop()

    if report_path is not None:
        write_json_report(report_path, report)
    return report


def evaluate_algorithm(
    *,
    algorithm: Algorithm,
    episodes: int,
    seed: int,
    checkpoint: str | None = None,
    model: str | None = None,
    render: bool = False,
    gif_path: str | Path | None = None,
    render_stride: int = 4,
    env_config: EnvConfig | None = None,
    render_fps: int = 12,
) -> dict[str, Any]:
    """Evaluate an instantiated RLlib algorithm on fresh simulator episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    module = algorithm.get_module(SHARED_POLICY_ID)
    initial_state = _initial_module_state(module)
    episode_summaries: list[dict[str, float]] = []
    first_episode_frames: list[SimulationFrame] = []
    env_config = env_config or EnvConfig()

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = SwarmSearchEnv({"seed": episode_seed, "env": env_config.model_dump(mode="json")})
        observations, _infos = env.reset(seed=episode_seed)
        policy_states = {agent_id: _copy_state(initial_state) for agent_id in observations}
        done = False
        episode_reward = 0.0
        final_metrics: dict[str, Any] = {}

        if render and episode_index == 0:
            first_episode_frames.append(_capture_frame(env))

        while not done:
            actions: dict[str, np.ndarray] = {}
            for agent_id, observation in observations.items():
                action, next_state = _compute_action(module, observation, policy_states[agent_id])
                policy_states[agent_id] = next_state
                actions[agent_id] = np.asarray(action, dtype=np.float32)

            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode_reward += float(next(iter(rewards.values())))
            done = bool(terminateds["__all__"] or truncateds["__all__"])
            final_metrics = dict(next(iter(infos.values()))["metrics"])

            should_capture = (
                render and episode_index == 0 and (done or int(final_metrics["timestep"]) % render_stride == 0)
            )
            if should_capture:
                first_episode_frames.append(_capture_frame(env, final_metrics))

        episode_summaries.append(episode_summary(episode_reward, final_metrics))

    extra: dict[str, Any] = {}
    if checkpoint is not None:
        extra["checkpoint"] = checkpoint
    if model is not None:
        extra["model"] = model
    report = aggregate_report(episode_summaries, policy="ppo", extra=extra)
    if render:
        render_episode_gif(first_episode_frames, gif_path, fps=render_fps, env_config=env_config)
    return report


def _capture_frame(env: SwarmSearchEnv, metrics: dict[str, Any] | None = None) -> SimulationFrame:
    """Capture one typed render frame from the current environment state."""
    state_snapshot, metrics_snapshot = env.snapshot()
    return SimulationFrame.from_snapshots(state_snapshot, metrics or metrics_snapshot)


def _initial_module_state(module: RLModule) -> dict[str, torch.Tensor]:
    """Return the shared module's initial recurrent state, if the model uses one."""
    state = module.get_initial_state()
    return {key: value.unsqueeze(0) for key, value in state.items()}


def _copy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy recurrent state tensors so each agent keeps independent memory."""
    return {key: value.clone() for key, value in state.items()}


def _compute_action(
    module: RLModule,
    observation: np.ndarray,
    state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """Compute one deterministic action for the shared module."""
    with torch.no_grad():
        observation_tensor = torch.from_numpy(observation.astype(np.float32))
        input_dict: dict[str, Any] = {Columns.OBS: observation_tensor.unsqueeze(0)}

        if state:
            input_dict[Columns.OBS] = observation_tensor.unsqueeze(0).unsqueeze(0)
            input_dict[Columns.STATE_IN] = state

        output = module.forward_inference(input_dict)

    next_state = dict(output.get(Columns.STATE_OUT, state))
    action = _action_from_module_output(module, output)
    return action, next_state


def _action_from_module_output(module: RLModule, output: dict[str, Any]) -> np.ndarray:
    """Convert RLModule inference output into one clipped environment action.

    RLlib can return actions in a couple of shapes/forms:
        - Include concrete actions under Columns.ACTIONS.
        - Action distribution inputs, usually logits or parameters, under Columns.ACTION_DIST_INPUTS.
    """
    if Columns.ACTIONS in output:
        action = convert_to_numpy(output[Columns.ACTIONS])
    else:
        logits = output[Columns.ACTION_DIST_INPUTS]
        if hasattr(logits, "ndim") and logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits[:, 0]
        distribution = module.get_inference_action_dist_cls().from_logits(logits)
        action = convert_to_numpy(distribution.to_deterministic().sample())

    action_array = np.asarray(action, dtype=np.float32).reshape((-1, *module.action_space.shape))[0]
    return np.clip(action_array, module.action_space.low, module.action_space.high).astype(np.float32)


def main() -> None:
    """Command-line entry point for PPO checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a DroneWatch RLlib PPO checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    if config.evaluation.checkpoint is None:
        raise ValueError("evaluation.checkpoint must be set via config or CLI override")

    resolved_path = save_resolved_config(
        config,
        resolved_config_path(Path(config.evaluation.report_path).parent, config),
    )

    report = evaluate_checkpoint(
        checkpoint=config.evaluation.checkpoint,
        episodes=config.evaluation.episodes,
        seed=config.evaluation_seed() or 0,
        report_path=config.evaluation.report_path,
        model=config.model.kind,
        render=config.evaluation.render,
        gif_path=config.evaluation.gif_path,
        render_stride=config.evaluation.render_stride,
        env_config=config.env,
        render_fps=config.rendering.fps,
    )
    report["resolved_config_path"] = str(resolved_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
