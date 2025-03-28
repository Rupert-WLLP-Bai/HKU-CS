import evaluate
import torch
from transformers.trainer_utils import EvalPrediction
from constants import ID_TO_LABEL, LABEL_TO_ID
import numpy as np


metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_predictions.predictions, eval_predictions.label_ids
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