import numpy as np
import pytest
from transformers.trainer_utils import EvalPrediction

from evaluation import compute_metrics


@pytest.mark.parametrize(
    "logits, labels, expected",
    [
        (
            np.array([[[0.7, 0.3], [0.4, 0.6]]]),
            np.array([[0, 1]]),
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        ),
        (
            np.array([[[0.7, 0.3], [0.4, 0.6]]]),
            np.array([[0, 0]]),
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        ),
        (
            np.array([[[0.7, 0.3], [0.4, 0.6]]]),
            np.array([[1, 1]]),
            {"precision": 1.0, "recall": 0.5, "f1": 2 / 3},
        ),
        (
            np.array([[[0.7, 0.3], [0.4, 0.6]]]),
            np.array([[1, 0]]),
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        ),
    ],
)
def test_score_computation(logits, labels, expected):
    """
    Test the computation of evaluation metrics (precision, recall, F1).

    This test uses parameterized inputs to check if the compute_metrics function
    correctly calculates the evaluation metrics for given logits and labels.

    Args:
        logits: Logits predicted by the model.
        labels: True labels corresponding to the input data.
        expected: Dictionary containing the expected precision, recall, and F1 values.

    The test verifies that the computed metrics are close to the expected values.
    """

    eval_predictions = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = compute_metrics(eval_predictions)

    # Check if the computed scores match the manual scores
    assert np.isclose(metrics["precision"], expected["precision"])
    assert np.isclose(metrics["recall"], expected["recall"])
    assert np.isclose(metrics["f1"], expected["f1"])
