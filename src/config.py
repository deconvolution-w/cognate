from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    _parents: tuple[Path, ...] = (),
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path in _parents:
        chain = " -> ".join(str(item) for item in (*_parents, config_path))
        raise ValueError(f"Circular base_config chain: {chain}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    base_value = config.pop("base_config", None)
    if not base_value:
        return config
    base_path = Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_config(base_path, (*_parents, config_path))
    return _deep_merge(base, config)
