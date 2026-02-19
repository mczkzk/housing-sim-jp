"""TOML config loader with CLI > config > default resolution."""

import argparse
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")

DEFAULTS = {
    "age": 30,
    "savings": 500.0,
    "income": 60.0,
    "children": "33,35",
    "no_child": False,
    "living": 27.0,
    "child_living": 5.0,
    "education": 10.0,
}


def load_config(path: Path | None = None) -> dict:
    """Load TOML config file. Returns empty dict if file doesn't exist."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    # Normalize: TOML `children = [33, 35]` -> comma-separated string for CLI compat
    if "children" in raw and isinstance(raw["children"], list):
        raw["children"] = ",".join(str(x) for x in raw["children"])
    return raw


def resolve(args: argparse.Namespace, config: dict) -> dict:
    """Resolve values with priority: CLI flag > config.toml > hardcoded default."""
    resolved = {}
    for key, default in DEFAULTS.items():
        cli_val = getattr(args, key, None)
        resolved[key] = cli_val if cli_val is not None else config.get(key, default)
    return resolved
