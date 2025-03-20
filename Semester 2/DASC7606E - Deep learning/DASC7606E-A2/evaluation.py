import evaluate
from transformers.trainer_utils import EvalPrediction

metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    """
    Compute evaluation metrics (precision, recall, f1) for predictions.

    First takes the argmax of the logits to convert them to predictions.
    Then we have to convert both labels and predictions from integers to strings.
    We remove all the values where the label is -100, then pass the results to the metric.compute() method.
    Finally, we return the overall precision, recall, and f1 score.

    Args:
        eval_predictions: Evaluation predictions.

    Returns:
        Dictionary with evaluation metrics. Keys: precision, recall, f1.

    NOTE: You can use `metric_evaluator` to compute metrics for a list of predictions and references.
    """
    # Write your code here.
