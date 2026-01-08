"""
Base metric abstraction for llm-eval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricResult(Dict[str, Any]):
    """
    Standard metric result container.

    Required:
    - score: float
    Optional:
    - error: str
    - metadata: dict
    """


class Metric(ABC):
    """
    Abstract base class for all evaluation metrics.
    """

    name: str

    @abstractmethod
    def compute(self, example: Dict, prediction: Dict) -> MetricResult:
        """
        Compute metric for a single example.

        :param example: Dataset row
        :param prediction: Model prediction row
        :return: MetricResult
        """
        raise NotImplementedError
