"""Matplotlib episode rendering for the random policy baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle

from dronewatch.config.schema import EnvConfig
from dronewatch.envs.spaces import AGENT_DEFAULTS, WORLD_DEFAULTS
from dronewatch.rendering.frame import SimulationFrame


def render_episode_gif(
    frames: Sequence[SimulationFrame],
    path: str | Path,
    fps: int = 12,
    env_config: EnvConfig | None = None,
) -> None:
    """Render typed simulation frames to a GIF file."""
    if not frames:
        raise ValueError("cannot render an empty episode")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=100):
        for frame in frames:
            _draw_frame(ax, frame, env_config)
            writer.grab_frame()
    plt.close(fig)


def _draw_frame(ax: plt.Axes, frame: SimulationFrame, env_config: EnvConfig | None = None) -> None:
    """Draw one `SimulationFrame` onto an existing matplotlib axes."""
    env_config = env_config or EnvConfig()
    state = frame.world_state
    metrics = frame.simulation_metrics
    ax.clear()
    ax.set_xlim(0.0, env_config.world.width if env_config else WORLD_DEFAULTS.width)
    ax.set_ylim(0.0, env_config.world.height if env_config else WORLD_DEFAULTS.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"step {metrics['timestep']} | targets {metrics['discovered_target_count']}/{metrics['target_count']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.4, alpha=0.3)

    for obstacle in state["obstacles"]:
        x, y = obstacle["position"]
        ax.add_patch(Circle((x, y), obstacle["radius"], color="#d95f02", alpha=0.25))

    undiscovered_targets = [target for target in state["targets"] if not target["discovered"]]
    discovered_targets = [target for target in state["targets"] if target["discovered"]]
    if undiscovered_targets:
        xs = [target["position"][0] for target in undiscovered_targets]
        ys = [target["position"][1] for target in undiscovered_targets]
        ax.scatter(xs, ys, marker="x", color="#7570b3", label="target")
    if discovered_targets:
        xs = [target["position"][0] for target in discovered_targets]
        ys = [target["position"][1] for target in discovered_targets]
        ax.scatter(xs, ys, marker="o", color="#1b9e77", label="discovered")

    agents = state["agents"]
    for left_index, left in enumerate(agents):
        for right in agents[left_index + 1 :]:
            dx = left["position"][0] - right["position"][0]
            dy = left["position"][1] - right["position"][1]
            communication_radius = (
                env_config.agents.communication_radius if env_config else AGENT_DEFAULTS.communication_radius
            )
            if (dx * dx + dy * dy) ** 0.5 <= communication_radius:
                ax.plot(
                    [left["position"][0], right["position"][0]],
                    [left["position"][1], right["position"][1]],
                    color="#999999",
                    alpha=0.2,
                    linewidth=0.6,
                )

    xs = [agent["position"][0] for agent in agents]
    ys = [agent["position"][1] for agent in agents]
    ax.scatter(xs, ys, marker="o", color="#1f78b4", s=24, label="drone")
    ax.legend(loc="upper right", fontsize=8)
