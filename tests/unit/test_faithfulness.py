import numpy as np
from llm_eval.metrics.rag.faithfulness import FaithfulnessMetric


def test_faithfulness_hallucination(mocker):
    metric = FaithfulnessMetric(lazy_load=True)

    mock_model = mocker.Mock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0]),  # context embedding
        np.array([0.0, 1.0]),  # hallucinated claim embedding
    ]

    mocker.patch.object(metric, "_get_model", return_value=mock_model)

    example = {
        "retrieved_contexts": ["Paris is the capital of France"]
    }
    prediction = {
        "answer": "The Eiffel Tower is in Berlin."
    }

    result = metric.compute(example=example, prediction=prediction)
    assert result.score < 0.5
