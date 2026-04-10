"""
Configuration Loader
=====================
Loads YAML config with OmegaConf, supports CLI overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config. Defaults to configs/default.yaml
        overrides: List of 'key=value' overrides (e.g., ['peak_detection.separation=8'])

    Returns:
        OmegaConf DictConfig
    """
    if config_path is None:
        # Look for default config relative to project root
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / "configs" / "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def config_to_flat_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf config to flat dict for MLflow logging."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return _flatten(container)


def _flatten(d: dict, prefix: str = "") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items
