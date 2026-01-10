import numpy as np
from llm_eval.metrics.rag.answer_relevancy import AnswerRelevancyMetric


def test_answer_relevancy_off_topic(mocker):
    metric = AnswerRelevancyMetric()

    mocker.patch.object(
        metric._model,
        "encode",
        side_effect=[
            np.array([1.0, 0.0]),  # query
            np.array([0.0, 1.0]),  # answer
        ],
    )

    example = {"query": "Explain neural networks"}
    prediction = {"answer": "The capital of France is Paris"}

    result = metric.compute(example=example, prediction=prediction)
    assert result.score < 0.5
