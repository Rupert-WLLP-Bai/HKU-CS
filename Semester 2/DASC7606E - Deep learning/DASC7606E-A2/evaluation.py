import evaluate
import torch
from transformers.trainer_utils import EvalPrediction
from constants import ID_TO_LABEL

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
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids

    # CRF 解码后的 predictions 是一个列表，需要转换为张量
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)
    
    # Convert predictions to label IDs
    predictions = predictions.argmax(axis=-1)

    # Remove ignored index (-100) from labels and predictions
    true_labels = [
        [ID_TO_LABEL[label] for label, pred in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]
    true_predictions = [
        [ID_TO_LABEL[pred] for label, pred in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    # Compute metrics using seqeval with zero_division=0 to suppress warnings
    results = metric_evaluator.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=0,  # Suppress warnings for undefined metrics
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }
