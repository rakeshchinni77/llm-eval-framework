import numpy as np

from llm_eval.metrics.reference.bertscore import BERTScoreMetric


def test_bertscore_semantic_similarity_mocked(mocker):
    metric = BERTScoreMetric(lazy_load=True)

    # Mock embedding method to avoid model download
    mocker.patch.object(
        metric,
        "_embed",
        side_effect=[
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
        ],
    )

    example = {"expected_answer": "A cat sits on the mat"}
    prediction = {"answer": "There is a cat on the mat"}

    result = metric.compute(example=example, prediction=prediction)

    assert 0.8 < result.score <= 1.0


def test_bertscore_empty():
    metric = BERTScoreMetric(lazy_load=True)
    example = {"expected_answer": "Some text"}
    prediction = {"answer": ""}
    result = metric.compute(example=example, prediction=prediction)
    assert result.score == 0.0
