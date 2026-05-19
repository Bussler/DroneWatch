"""OmegaConf composition and resolved-config persistence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from .schema import (
    DroneWatchConfig,
    DroneWatchEvaluationConfig,
    DroneWatchRandomPolicyConfig,
)

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)

TRAINING_GROUP_NAMES = {"env", "model", "training", "rendering", "tune"}
EVALUATION_GROUP_NAMES = {"env", "model", "evaluation", "rendering"}
RANDOM_POLICY_GROUP_NAMES = {"env", "random_policy", "rendering"}


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> DroneWatchConfig:
    """Load, compose, override, resolve, and validate a PPO training config."""
    return _load_typed_config(config_path, overrides, DroneWatchConfig, TRAINING_GROUP_NAMES)


def load_evaluation_config(
    config_path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DroneWatchEvaluationConfig:
    """Load, compose, override, resolve, and validate a standalone evaluation config."""
    return _load_typed_config(config_path, overrides, DroneWatchEvaluationConfig, EVALUATION_GROUP_NAMES)


def load_random_policy_config(
    config_path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DroneWatchRandomPolicyConfig:
    """Load, compose, override, resolve, and validate a standalone random-policy config."""
    return _load_typed_config(config_path, overrides, DroneWatchRandomPolicyConfig, RANDOM_POLICY_GROUP_NAMES)


def save_resolved_config(config: BaseModel, path: str | Path) -> Path:
    """Write a fully resolved config model as YAML and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = OmegaConf.create(config.model_dump(mode="json"))
    output_path.write_text(OmegaConf.to_yaml(payload, resolve=True), encoding="utf-8")
    return output_path


def resolved_config_path(
    base_dir: str | Path, config: DroneWatchConfig | DroneWatchEvaluationConfig | DroneWatchRandomPolicyConfig
) -> Path:
    """Return the standard resolved-config path for a run artifact directory."""
    return Path(base_dir) / config.project.resolved_config_filename


def _load_typed_config(
    config_path: str | Path,
    overrides: Iterable[str] | None,
    model_type: type[ConfigModel],
    group_names: set[str],
) -> ConfigModel:
    path = Path(config_path)
    overrides = list(overrides or [])
    group_overrides, field_overrides = _split_overrides(overrides, group_names)
    composed = _compose_config(path, group_overrides)
    if field_overrides:
        composed = OmegaConf.merge(composed, OmegaConf.from_dotlist(field_overrides))
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} did not resolve to a mapping")
    data.pop("defaults", None)
    return model_type.model_validate(data)


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


def _split_overrides(overrides: list[str], group_names: set[str]) -> tuple[dict[str, str], list[str]]:
    group_overrides: dict[str, str] = {}
    field_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override}")
        key, value = override.split("=", 1)
        if key in group_names:
            group_overrides[key] = value
        else:
            field_overrides.append(override)
    return group_overrides, field_overrides
