from typing import Dict, List
import statistics


class Aggregator:
    """
    Aggregates metric scores across dataset.
    Computes standard statistics per metric.
    """

    @staticmethod
    def aggregate(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "mean": float(statistics.mean(scores)),
            "median": float(statistics.median(scores)),
            "std": float(statistics.pstdev(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
        }
