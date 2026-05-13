"""Python-facing helpers for the Rust simulation extension."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import swarm_sim as env  # type: ignore


def rust_version() -> str:
    """Return the version string reported by the Rust extension."""
    return env.version()


class SwarmSimulation:
    """Small Python wrapper around the Phase 1 Rust simulation world."""

    def __init__(self, seed: int | None = None) -> None:
        self._world = env.SwarmWorld(seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return self._world.reset(seed)

    def step(self, actions: Sequence[Sequence[float]]) -> dict[str, Any]:
        rust_actions = [(float(action[0]), float(action[1])) for action in actions]
        return self._world.step(rust_actions)

    def state(self) -> dict[str, Any]:
        return self._world.state()

    def metrics(self) -> dict[str, Any]:
        return self._world.metrics()

    def is_done(self) -> bool:
        return bool(self._world.is_done())

    @property
    def num_agents(self) -> int:
        return len(self.state()["agents"])
