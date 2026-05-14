"""Typed render frame data for DroneWatch episode visualization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SimulationFrame(BaseModel):
    """Snapshot of simulator state and metrics used by the renderer.

    The Rust/Python boundary currently exposes world state and metrics as dictionaries. This model
    gives that pair a stable name and clearer field semantics without duplicating the full nested
    Rust schema in Python.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    world_state: dict[str, Any] = Field(
        description="World state snapshot returned by SwarmSimulation.state() after a reset or step."
    )
    simulation_metrics: dict[str, Any] = Field(
        description="Simulation metrics snapshot returned by SwarmSimulation.metrics() or step infos."
    )

    @classmethod
    def from_snapshots(cls, world_state: Mapping[str, Any], simulation_metrics: Mapping[str, Any]) -> "SimulationFrame":
        """Build a frame from mapping snapshots while storing plain dictionaries."""
        return cls(world_state=dict(world_state), simulation_metrics=dict(simulation_metrics))
