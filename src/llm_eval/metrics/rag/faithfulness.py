from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness metric — detects hallucinations.
    """

    name = "faithfulness"
    requires_reference = False
    requires_context = True

    _model: SentenceTransformer | None = None

    def __init__(
        self,
        threshold: float = 0.6,
        model_name: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.model_name = model_name
        self.lazy_load = lazy_load

        if not lazy_load and FaithfulnessMetric._model is None:
            FaithfulnessMetric._model = SentenceTransformer(model_name)

    def _get_model(self) -> SentenceTransformer:
        if FaithfulnessMetric._model is None:
            FaithfulnessMetric._model = SentenceTransformer(self.model_name)
        return FaithfulnessMetric._model

    @staticmethod
    def _split_claims(text: str) -> List[str]:
        return [s.strip() for s in text.split(".") if s.strip()]

    def compute(
        self,
        *,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> MetricResult:
        try:
            answer = prediction.get("answer", "")
            contexts = example.get("retrieved_contexts", [])

            if not answer or not contexts:
                return MetricResult(score=0.0)

            claims = self._split_claims(answer)
            if not claims:
                return MetricResult(score=0.0)

            model = self._get_model()

            context_text = " ".join(contexts)
            ctx_emb = model.encode(context_text, normalize_embeddings=True)

            supported = 0
            for claim in claims:
                claim_emb = model.encode(claim, normalize_embeddings=True)
                sim = util.cos_sim(claim_emb, ctx_emb).item()
                if sim >= self.threshold:
                    supported += 1

            return MetricResult(score=supported / len(claims))

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(FaithfulnessMetric)
