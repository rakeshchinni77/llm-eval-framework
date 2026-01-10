from llm_eval.metrics.reference.bleu import BLEUMetric


def test_bleu_exact_match():
    metric = BLEUMetric(n_gram=2)
    example = {"expected_answer": "Paris is the capital of France"}
    prediction = {"answer": "Paris is the capital of France"}
    result = metric.compute(example=example, prediction=prediction)
    assert result.score > 0.9


def test_bleu_empty_prediction():
    metric = BLEUMetric()
    example = {"expected_answer": "Some text"}
    prediction = {"answer": ""}
    result = metric.compute(example=example, prediction=prediction)
    assert result.score == 0.0
