class QualityGateError(RuntimeError):
    pass


class QualityGates:
    """
    Defines hard evaluation thresholds for CI/CD.
    """

    RULES = {
        "bleu": {"mean": 0.3},
        "faithfulness": {"mean": 0.7},
    }

    @classmethod
    def validate(cls, aggregated_results: dict) -> None:
        for metric_name, conditions in cls.RULES.items():
            if metric_name not in aggregated_results:
                continue

            for stat, threshold in conditions.items():
                value = aggregated_results[metric_name].get(stat, 0.0)
                if value < threshold:
                    raise QualityGateError(
                        f"Quality gate failed: {metric_name}.{stat} "
                        f"{value:.3f} < {threshold:.3f}"
                    )
