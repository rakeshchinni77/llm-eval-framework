"""
Answer relevancy metric.

Measures whether the generated answer
actually addresses the query.
"""

from __future__ import annotations

from typing import Any, Dict

from sentence_transformers import SentenceTransformer, util

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


class AnswerRelevancyMetric(BaseMetric):
    """
    Answer relevancy metric.

    Strategy:
    - Embed query
    - Embed answer
    - Compute cosine similarity
    - Threshold-normalize score
    """

    name = "answer_relevancy"
    requires_reference = False
    requires_context = False

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.model_name = model_name

        # 🔴 CRITICAL: must exist for tests to mock
        self._model = SentenceTransformer(self.model_name)

    def compute(
        self,
        *,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> MetricResult:
        try:
            query = example.get("query", "")
            answer = prediction.get("answer", "")

            if not query or not answer:
                return MetricResult(score=0.0)

            q_emb = self._model.encode(query, normalize_embeddings=True)
            a_emb = self._model.encode(answer, normalize_embeddings=True)

            similarity = float(util.cos_sim(q_emb, a_emb).item())

            # Normalize to [0, 1]
            score = max(0.0, min(1.0, similarity / self.threshold))

            return MetricResult(score=score)

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(AnswerRelevancyMetric)
