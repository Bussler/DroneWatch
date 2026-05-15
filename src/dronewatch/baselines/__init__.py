"""Baseline policies and rollout helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["RandomPolicy", "run_random_policy"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .random_policy import RandomPolicy, run_random_policy

        return {"RandomPolicy": RandomPolicy, "run_random_policy": run_random_policy}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
