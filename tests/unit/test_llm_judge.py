import json
import pytest

from llm_eval.metrics.judge.llm_judge import LLMJudgeMetric
from llm_eval.llm_providers.openai_provider import OpenAIProvider
from llm_eval.llm_providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def example():
    return {"query": "What is the capital of France?"}


@pytest.fixture
def prediction():
    return {"answer": "The capital of France is Paris."}


# -------------------------------------------------
# SUCCESS CASE — VALID JSON RESPONSE
# -------------------------------------------------

def test_llm_judge_success_openai(mocker, example, prediction):
    mocker.patch.object(
        OpenAIProvider,
        "generate",
        return_value=json.dumps(
            {
                "coherence": 5,
                "relevance": 5,
                "safety": 5,
                "reasoning": "Clear and correct answer.",
            }
        ),
    )

    metric = LLMJudgeMetric(
        provider="openai",
        model="gpt-4",
    )

    result = metric.compute(example=example, prediction=prediction)

    assert result.score == pytest.approx(1.0)
    assert result.metadata["coherence"] == 5
    assert result.error is None


# -------------------------------------------------
# RETRY CASE — INVALID JSON THEN SUCCESS
# -------------------------------------------------

def test_llm_judge_invalid_json_retry(mocker, example, prediction):
    mocker.patch.object(
        OpenAIProvider,
        "generate",
        side_effect=[
            "NOT JSON",
            json.dumps(
                {
                    "coherence": 4,
                    "relevance": 4,
                    "safety": 5,
                    "reasoning": "Mostly correct.",
                }
            ),
        ],
    )

    metric = LLMJudgeMetric(
        provider="openai",
        model="gpt-4",
    )

    result = metric.compute(example=example, prediction=prediction)

    assert result.score > 0.0
    assert result.metadata["safety"] == 5
    assert result.error is None


# -------------------------------------------------
# FAILURE CASE — RETRIES EXHAUSTED
# -------------------------------------------------

def test_llm_judge_retry_exhausted_returns_zero(mocker, example, prediction):
    mocker.patch.object(
        OpenAIProvider,
        "generate",
        return_value="INVALID JSON ALWAYS",
    )

    metric = LLMJudgeMetric(
        provider="openai",
        model="gpt-4",
    )

    result = metric.compute(example=example, prediction=prediction)

    assert result.score == 0.0
    assert result.error is not None


# -------------------------------------------------
# PROVIDER FAILURE — MUST NOT CRASH
# -------------------------------------------------

def test_llm_judge_provider_exception_safe(mocker, example, prediction):
    mocker.patch.object(
        AnthropicProvider,
        "generate",
        side_effect=RuntimeError("API down"),
    )

    metric = LLMJudgeMetric(
        provider="anthropic",
        model="claude-3-opus",
    )

    result = metric.compute(example=example, prediction=prediction)

    assert result.score == 0.0
    assert "API down" in result.error
