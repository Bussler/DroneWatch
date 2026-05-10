"""Python-facing helpers for the Rust simulation extension."""

from __future__ import annotations

import swarm_sim as env  # type: ignore


def rust_version() -> str:
    """Return the version string reported by the Rust extension."""
    return env.version()
