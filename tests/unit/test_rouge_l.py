from llm_eval.metrics.reference.rouge_l import RougeLMetric


def test_rouge_l_exact_match():
    metric = RougeLMetric()
    example = {"expected_answer": "machine learning is fun"}
    prediction = {"answer": "machine learning is fun"}
    result = metric.compute(example=example, prediction=prediction)
    assert result.score == 1.0


def test_rouge_l_no_overlap():
    metric = RougeLMetric()
    example = {"expected_answer": "hello world"}
    prediction = {"answer": "foo bar"}
    result = metric.compute(example=example, prediction=prediction)
    assert result.score == 0.0
