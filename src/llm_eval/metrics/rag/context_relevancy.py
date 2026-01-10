"""
Context relevancy metric.

Evaluates retrieval quality by measuring similarity
between query and retrieved contexts.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


class ContextRelevancyMetric(BaseMetric):
    """
    Context relevancy metric.

    Strategy:
    - Embed query
    - Embed each retrieved context
    - Compute mean cosine similarity
    - Normalize using threshold
    """

    name = "context_relevancy"
    requires_reference = False
    requires_context = True

    _model: SentenceTransformer | None = None

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.model_name = model_name
        self.lazy_load = lazy_load

        if not lazy_load and ContextRelevancyMetric._model is None:
            ContextRelevancyMetric._model = SentenceTransformer(model_name)

    def _get_model(self) -> SentenceTransformer:
        if ContextRelevancyMetric._model is None:
            ContextRelevancyMetric._model = SentenceTransformer(self.model_name)
        return ContextRelevancyMetric._model

    def compute(
        self,
        *,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> MetricResult:
        try:
            query = example.get("query", "")
            contexts: List[str] = example.get("retrieved_contexts", [])

            if not query or not contexts:
                return MetricResult(score=0.0)

            model = self._get_model()

            query_emb = model.encode(query, normalize_embeddings=True)
            ctx_embs = model.encode(contexts, normalize_embeddings=True)

            sims = util.cos_sim(query_emb, ctx_embs).flatten().tolist()
            avg_sim = sum(sims) / len(sims)

            # Normalize to [0,1]
            score = min(1.0, avg_sim / self.threshold)

            return MetricResult(score=float(score))

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(ContextRelevancyMetric)
