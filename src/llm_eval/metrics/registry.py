"""
Metric registry and factory.
"""

from typing import Dict, Type

from llm_eval.metrics.base import Metric


_METRIC_REGISTRY: Dict[str, Type[Metric]] = {}


def register_metric(name: str, metric_cls: Type[Metric]) -> None:
    if name in _METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' already registered")
    _METRIC_REGISTRY[name] = metric_cls


def get_metric(name: str) -> Type[Metric]:
    try:
        return _METRIC_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown metric '{name}'") from exc


def list_metrics() -> Dict[str, Type[Metric]]:
    return dict(_METRIC_REGISTRY)
