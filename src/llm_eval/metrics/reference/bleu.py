"""
BLEU score metric implementation.

Uses simple whitespace tokenization to ensure
CI safety and deterministic behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


class BLEUMetric(BaseMetric):
    """
    BLEU score metric (sentence-level).

    Supports configurable n-gram order (1–4).
    """

    name = "bleu"
    requires_reference = True
    requires_context = False

    def __init__(self, n_gram: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 1 <= n_gram <= 4:
            raise ValueError("BLEU n_gram must be between 1 and 4")
        self.n_gram = n_gram
        self.weights = tuple([1.0 / n_gram] * n_gram)
        self.smoothing = SmoothingFunction().method1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Deterministic, CI-safe tokenization
        return text.lower().strip().split()

    def compute(
        self,
        *,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> MetricResult:
        try:
            reference = example.get("expected_answer", "")
            candidate = prediction.get("answer", "")

            if not reference or not candidate:
                return MetricResult(score=0.0)

            ref_tokens = self._tokenize(reference)
            cand_tokens = self._tokenize(candidate)

            if not ref_tokens or not cand_tokens:
                return MetricResult(score=0.0)

            score = sentence_bleu(
                [ref_tokens],
                cand_tokens,
                weights=self.weights,
                smoothing_function=self.smoothing,
            )

            return MetricResult(score=float(max(0.0, min(1.0, score))))

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(BLEUMetric)
