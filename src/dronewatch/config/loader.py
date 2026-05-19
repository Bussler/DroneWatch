"""OmegaConf composition and resolved-config persistence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .schema import DroneWatchConfig

GROUP_NAMES = {"env", "model", "training", "evaluation", "rendering", "tune"}


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> DroneWatchConfig:
    """Load, compose, override, resolve, and validate a DroneWatch config."""
    path = Path(config_path)
    overrides = list(overrides or [])
    group_overrides, field_overrides = _split_overrides(overrides)
    composed = _compose_config(path, group_overrides)
    if field_overrides:
        composed = OmegaConf.merge(composed, OmegaConf.from_dotlist(field_overrides))
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} did not resolve to a mapping")
    data.pop("defaults", None)
    return DroneWatchConfig.model_validate(data)


def save_resolved_config(config: DroneWatchConfig, path: str | Path) -> Path:
    """Write a fully resolved config model as YAML and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = OmegaConf.create(config.model_dump(mode="json"))
    output_path.write_text(OmegaConf.to_yaml(payload, resolve=True), encoding="utf-8")
    return output_path


def resolved_config_path(base_dir: str | Path, config: DroneWatchConfig) -> Path:
    """Return the standard resolved-config path for a run artifact directory."""
    return Path(base_dir) / config.runtime.resolved_config_filename


def _compose_config(path: Path, group_overrides: Mapping[str, str]) -> DictConfig:
    root = OmegaConf.load(path)
    defaults = list(root.get("defaults", []))
    config_dir = path.parent
    merged = OmegaConf.create({})

    for item in defaults:
        group, name = _parse_default(item)
        selected = group_overrides.get(group, name)
        group_path = config_dir / group / f"{selected}.yaml"
        if not group_path.exists():
            raise FileNotFoundError(f"config group file not found: {group_path}")
        merged = OmegaConf.merge(merged, OmegaConf.load(group_path))

    root_without_defaults = OmegaConf.create(OmegaConf.to_container(root, resolve=False))
    root_without_defaults.pop("defaults", None)
    return OmegaConf.merge(merged, root_without_defaults)


def _parse_default(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        if ":" not in item:
            raise ValueError(f"default entry must be a group mapping, got {item!r}")
        group, name = item.split(":", 1)
        return group.strip(), name.strip()
    if isinstance(item, Mapping):
        if len(item) != 1:
            raise ValueError(f"default mapping must contain exactly one group, got {item!r}")
        group, name = next(iter(item.items()))
        return str(group), str(name)
    raise ValueError(f"unsupported default entry: {item!r}")


def _split_overrides(overrides: list[str]) -> tuple[dict[str, str], list[str]]:
    group_overrides: dict[str, str] = {}
    field_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override}")
        key, value = override.split("=", 1)
        if key in GROUP_NAMES:
            group_overrides[key] = value
        else:
            field_overrides.append(override)
    return group_overrides, field_overrides
