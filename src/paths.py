from __future__ import annotations

from pathlib import Path
import os


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    configured = os.environ.get("COGNATE_CONFIG")
    return (
        Path(configured).expanduser()
        if configured
        else project_root() / "configs" / "inference.yaml"
    )


def default_checkpoint_path() -> Path:
    configured = os.environ.get("COGNATE_CHECKPOINT")
    return (
        Path(configured).expanduser()
        if configured
        else project_root() / "checkpoints" / "checkpoint.pth"
    )


def default_hla_lookup_path() -> Path:
    return project_root() / "data" / "examples" / "hla_pseudo_sequences.csv"
