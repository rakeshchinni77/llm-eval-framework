"""
Configuration loader for llm-eval.

Supports YAML and JSON formats with strict validation.
"""

import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from llm_eval.config.schema import EvalConfig


class ConfigLoadError(RuntimeError):
    """Raised when configuration loading or validation fails."""


def _load_raw_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        raise ConfigLoadError(
            f"Unsupported config format '{suffix}'. Use YAML or JSON."
        )

    except Exception as exc:
        raise ConfigLoadError(f"Failed to read config file: {exc}") from exc


def load_config(path: Path) -> EvalConfig:
    """
    Load and validate configuration.

    Fails fast on:
    - IO errors
    - Invalid YAML / JSON
    - Schema violations
    """
    try:
        raw = _load_raw_config(path)

        if not isinstance(raw, dict):
            raise ConfigLoadError("Configuration root must be a mapping/object")

        return EvalConfig.model_validate(raw)

    except ValidationError as exc:
        raise ConfigLoadError(
            f"Configuration validation failed:\n{exc}"
        ) from exc
