"""
ROUGE-L metric implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from llm_eval.metrics.base import BaseMetric, MetricResult
from llm_eval.metrics.registry import MetricRegistry


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


class RougeLMetric(BaseMetric):
    """
    ROUGE-L metric based on longest common subsequence.
    """

    name = "rouge_l"
    requires_reference = True
    requires_context = False

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

            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()

            lcs = _lcs_length(ref_tokens, cand_tokens)
            recall = lcs / len(ref_tokens)
            precision = lcs / len(cand_tokens)

            if recall + precision == 0:
                score = 0.0
            else:
                score = (2 * recall * precision) / (recall + precision)

            return MetricResult(score=float(max(0.0, min(1.0, score))))

        except Exception as exc:
            return MetricResult(score=0.0, error=str(exc))


MetricRegistry.register(RougeLMetric)
