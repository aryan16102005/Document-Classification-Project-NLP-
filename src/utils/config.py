"""
Configuration loader — reads YAML config and provides a structured dict.
src/utils/config.py
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load and return the YAML configuration as a Python dict.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {path}. Using empty config.")
        return {}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config loaded from {path}")
    return config or {}
