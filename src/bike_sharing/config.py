"""Configuration loading for the bike_sharing package."""

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = ("seed", "target", "datetime_col", "paths", "drop_columns")
REQUIRED_PATH_KEYS = ("raw_train", "raw_test")
PATH_KEYS_TO_RESOLVE = (
    "raw_train",
    "raw_test",
    "raw_sample_submission",
    "interim_dir",
    "processed_dir",
    "models_dir",
    "reports_dir",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the project YAML config.

    Paths under ``paths`` are resolved to absolute paths anchored at the
    config file's parent directory's parent (i.e. the project root), so the
    same config works regardless of the current working directory.
    """
    config_path = Path(path).resolve()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    project_root = config_path.parent.parent
    _resolve_paths(cfg, project_root)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    missing_paths = [k for k in REQUIRED_PATH_KEYS if k not in cfg["paths"]]
    if missing_paths:
        raise ValueError(f"config['paths'] missing required keys: {missing_paths}")


def _resolve_paths(cfg: dict[str, Any], project_root: Path) -> None:
    paths = cfg["paths"]
    for key in PATH_KEYS_TO_RESOLVE:
        if key in paths:
            paths[key] = str((project_root / paths[key]).resolve())
