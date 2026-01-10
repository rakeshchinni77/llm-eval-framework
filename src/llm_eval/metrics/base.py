"""
Base metric abstraction for llm-eval.

All metrics (reference-based, RAG, LLM-as-a-Judge, plugins)
MUST inherit from BaseMetric and implement the required interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class MetricResult:
    """
    Standard metric output container.

    Attributes:
        score: Normalized score in range [0, 1]
        error: Optional error message if metric failed
        metadata: Optional metric-specific diagnostic information
    """

    score: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.

    Design goals:
    - Strict contract
    - Runner-agnostic
    - Plugin-friendly
    - Testable in isolation
    """

    #: Unique metric identifier (used in config & reports)
    name: str

    #: Whether this metric requires a reference / ground-truth answer
    requires_reference: bool = False

    #: Whether this metric requires retrieved context(s)
    requires_context: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Optional metric-specific configuration.

        All keyword arguments MUST be explicit to allow
        config-driven instantiation and validation.
        """
        self.config = kwargs

    @abstractmethod
    def compute(
        self,
        *,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> MetricResult:
        """
        Compute metric score for a single example.

        Args:
            example:
                One benchmark item (query, expected_answer, retrieved_contexts, etc.)
            prediction:
                Model output for the same example

        Returns:
            MetricResult with score in [0, 1]

        Rules:
        - NEVER raise uncaught exceptions
        - On failure, return MetricResult(score=0.0, error="...")
        - MUST be deterministic for same inputs
        """
        raise NotImplementedError
