"""
Metric registry and factory.

This module provides:
- Central registry for metric classes
- Config-driven metric instantiation
- Plugin extensibility without touching core code
"""

from __future__ import annotations

from typing import Dict, Type

from llm_eval.metrics.base import BaseMetric


class MetricRegistry:
    """
    Global metric registry.

    Maps metric name → metric class.
    """

    _registry: Dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, metric_cls: Type[BaseMetric]) -> None:
        """
        Register a metric class.

        Raises:
            ValueError if:
            - metric has no name
            - name collision occurs
            - metric does not inherit BaseMetric
        """
        if not issubclass(metric_cls, BaseMetric):
            raise TypeError(
                f"Metric {metric_cls.__name__} must inherit from BaseMetric"
            )

        name = getattr(metric_cls, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(
                f"Metric class {metric_cls.__name__} must define a string 'name'"
            )

        if name in cls._registry:
            raise ValueError(
                f"Metric '{name}' already registered "
                f"by {cls._registry[name].__name__}"
            )

        cls._registry[name] = metric_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseMetric]:
        """
        Retrieve a metric class by name.
        """
        try:
            return cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown metric '{name}'. Available metrics: {available}"
            ) from exc

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseMetric:
        """
        Instantiate a metric by name with configuration.

        Used by evaluation runner.
        """
        metric_cls = cls.get(name)
        return metric_cls(**kwargs)

    @classmethod
    def list_metrics(cls) -> Dict[str, Type[BaseMetric]]:
        """
        Return all registered metrics.
        """
        return dict(cls._registry)
