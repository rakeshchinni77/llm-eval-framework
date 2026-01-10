"""
BERTScore metric implementation using sentence-transformers.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, util

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


class BERTScoreMetric(BaseMetric):
    """
    BERTScore using cosine similarity of sentence embeddings.
    """

    name = "bertscore"
    requires_reference = True
    requires_context = False

    _model: SentenceTransformer | None = None
    _embedding_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.lazy_load = lazy_load

        if not lazy_load and BERTScoreMetric._model is None:
            BERTScoreMetric._model = SentenceTransformer(model_name)

    def _get_model(self) -> SentenceTransformer:
        if BERTScoreMetric._model is None:
            BERTScoreMetric._model = SentenceTransformer(self.model_name)
        return BERTScoreMetric._model


    def _embed(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            model = self._get_model()
            emb = model.encode(text, normalize_embeddings=True)
            self._embedding_cache[text] = emb
        return self._embedding_cache[text]


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

            ref_emb = self._embed(reference)
            cand_emb = self._embed(candidate)

            score = util.cos_sim(ref_emb, cand_emb).item()

            # cosine similarity → normalize to [0,1]
            score = (score + 1) / 2

            return MetricResult(score=float(max(0.0, min(1.0, score))))

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(BERTScoreMetric)
