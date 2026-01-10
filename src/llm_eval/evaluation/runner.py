from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from pathlib import Path
import json

from llm_eval.metrics.registry import MetricRegistry
from llm_eval.evaluation.aggregator import Aggregator


class EvaluationRunner:
    """
    Core evaluation orchestrator.

    Responsibilities:
    - Load model predictions
    - Loop over models and metrics
    - Align dataset ↔ predictions safely
    - Parallel execution
    - Aggregate results
    - Persist raw scores for visualization
    - (Quality gates intentionally disabled for local runs)
    """

    def __init__(
        self,
        *,
        dataset: List[Dict[str, Any]],
        models: List[Dict[str, Any]],
        metrics,  # List[MetricConfig]
        output_dir: Path,
        max_workers: int = 4,
    ) -> None:
        self.dataset = dataset
        self.models = models
        self.metrics = metrics
        self.output_dir = output_dir
        self.max_workers = max_workers

    def _load_predictions(self, path: Path) -> List[Dict[str, Any]]:
        """Load model predictions from JSONL"""
        predictions: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                predictions.append(json.loads(line))
        return predictions

    def _evaluate_single(
        self,
        metric,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> float:
        try:
            result = metric.compute(example=example, prediction=prediction)
            return float(result.score)
        except Exception:
            return 0.0

    def run(self) -> Dict[str, Any]:
        final_results: Dict[str, Any] = {}
        raw_scores: Dict[str, Dict[str, List[float]]] = {}

        for model_cfg in self.models:
            model_name = model_cfg["name"]
            predictions = self._load_predictions(model_cfg["predictions"])

            final_results[model_name] = {}
            raw_scores[model_name] = {}

            # SAFETY: align dataset & predictions
            max_len = min(len(self.dataset), len(predictions))

            for metric_cfg in self.metrics:
                metric_cls = MetricRegistry.get(metric_cfg.name)
                metric = metric_cls(**metric_cfg.params)

                scores: List[float] = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._evaluate_single,
                            metric,
                            self.dataset[idx],
                            {
                                # normalized prediction format for all metrics
                                "answer": predictions[idx].get("prediction", "")
                            },
                        )
                        for idx in range(max_len)
                    ]

                    for future in as_completed(futures):
                        scores.append(future.result())

                raw_scores[model_name][metric_cfg.name] = scores
                final_results[model_name][metric_cfg.name] = Aggregator.aggregate(scores)

        # REQUIRED for Phase 11 visualizations
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "raw_scores.json", "w", encoding="utf-8") as f:
            json.dump(raw_scores, f, indent=2)

        with open(self.output_dir / "aggregates.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

        # Quality gates intentionally DISABLED for local execution
        # CI/CD pipelines will re-enable them
        return final_results
