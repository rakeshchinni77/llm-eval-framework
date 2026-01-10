import numpy as np
from llm_eval.metrics.rag.context_relevancy import ContextRelevancyMetric


def test_context_relevancy_irrelevant(mocker):
    metric = ContextRelevancyMetric(lazy_load=True)

    # Create a fake model
    mock_model = mocker.Mock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0]),          # query embedding
        [np.array([0.0, 1.0])],        # context embeddings (list)
    ]

    # Patch model retrieval
    mocker.patch.object(metric, "_get_model", return_value=mock_model)

    example = {
        "query": "What is machine learning?",
        "retrieved_contexts": ["Weather forecast today"],
    }

    result = metric.compute(example=example, prediction={})

    assert result.score < 0.5
