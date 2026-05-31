"""Configuration loading for the bike_sharing package."""

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = ("seed", "target", "datetime_col", "paths", "drop_columns", "cv")
PATH_KEYS_TO_RESOLVE = (
    "raw_train",
    "raw_test",
    "raw_sample_submission",
    "interim_dir",
    "processed_dir",
    "models_dir",
    "reports_dir",
)
REQUIRED_PATH_KEYS = PATH_KEYS_TO_RESOLVE


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


def _validate(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(
            "config file did not parse to a mapping; got "
            f"{type(cfg).__name__}. Is the YAML empty or malformed?"
        )
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if not isinstance(cfg["paths"], dict):
        raise ValueError(
            "config['paths'] must be a mapping of name -> relative path; "
            f"got {type(cfg['paths']).__name__}."
        )
    missing_paths = [k for k in REQUIRED_PATH_KEYS if k not in cfg["paths"]]
    if missing_paths:
        raise ValueError(f"config['paths'] missing required keys: {missing_paths}")
    if not isinstance(cfg["cv"], dict) or "n_splits" not in cfg["cv"]:
        raise ValueError(
            "config['cv'] must be a mapping containing 'n_splits'; "
            f"got {cfg['cv']!r}."
        )


def _resolve_paths(cfg: dict[str, Any], project_root: Path) -> None:
    paths = cfg["paths"]
    for key in PATH_KEYS_TO_RESOLVE:
        if key in paths:
            paths[key] = str((project_root / paths[key]).resolve())
