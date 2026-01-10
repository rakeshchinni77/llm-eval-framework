"""
Strict Pydantic configuration schema for llm-eval.

All configurations must conform exactly to this schema.
Unknown fields are forbidden.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict, RootModel


# =========================
# Enums
# =========================

class JudgeProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"


# =========================
# Dataset
# =========================

class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path = Field(..., description="Path to benchmark dataset (JSONL or CSV).")

    @field_validator("path")
    @classmethod
    def validate_dataset_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Dataset file does not exist: {v}")
        if v.suffix.lower() not in {".jsonl", ".csv"}:
            raise ValueError("Dataset must be .jsonl or .csv")
        return v


# =========================
# Models
# =========================

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    predictions: Path = Field(..., description="Model output file (JSONL).")

    @field_validator("predictions")
    @classmethod
    def validate_predictions_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Predictions file does not exist: {v}")
        if v.suffix.lower() != ".jsonl":
            raise ValueError("Predictions file must be .jsonl")
        return v


# =========================
# Metrics (Pydantic v2 FIX)
# =========================

class MetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(RootModel[List[MetricConfig]]):
    """
    List-based metric configuration.
    Example:
    metrics:
      - name: bleu
      - name: faithfulness
      - name: llm_judge
        params: {...}
    """

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)


# =========================
# Quality Gates
# =========================

class QualityGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric thresholds (e.g. faithfulness.mean >= 0.7)",
    )


# =========================
# Root Config
# =========================

class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    models: List[ModelConfig]
    metrics: MetricsConfig
    quality_gates: Optional[QualityGateConfig] = None

    @field_validator("models")
    @classmethod
    def at_least_one_model(cls, v: List[ModelConfig]) -> List[ModelConfig]:
        if not v:
            raise ValueError("At least one model must be configured")
        return v
