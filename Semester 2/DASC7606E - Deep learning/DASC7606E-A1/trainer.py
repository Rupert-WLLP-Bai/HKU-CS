import torch
from transformers import DetrImageProcessor, Trainer, TrainingArguments

from constants import OUTPUT_DIR, ID2LABEL
from evaluate import compute_metrics
from functools import partial


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the object detection model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # Where to save the model checkpoints
        num_train_epochs=10,  # Adjust number of epochs as needed
        fp16=False,  # Use mixed precision if you have a supported GPU (set to True for faster training)
        per_device_train_batch_size=8,  # Batch size for training
        dataloader_num_workers=4,  # Number of worker processes for data loading
        learning_rate=1e-3,  # Learning rate for fine-tuning
        lr_scheduler_type="cosine",  # Type of learning rate scheduler
        weight_decay=1e-4,  # Weight decay to avoid overfitting
        max_grad_norm=0.1,  # Gradient clipping to avoid exploding gradients
        metric_for_best_model="eval_map",  # Metric to determine the best model
        greater_is_better=True,  # Whether a higher metric is better
        load_best_model_at_end=True,  # Load the best model after training
        eval_strategy="epoch",  # Evaluate at the end of every epoch
        save_strategy="epoch",  # Save the model at the end of every epoch
        save_total_limit=2,  # Keep only the last 2 checkpoints
        remove_unused_columns=False,  # Don't remove columns like 'image' (important for data)
        eval_do_concat_batches=False,  # Ensure proper evaluation when batches are not concatenated
        push_to_hub=False,  # Whether to push the model to the Hub
    )

    return training_args


def build_trainer(model, processor, datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Object detection model.
        processor: Image processor object (e.g., DetrImageProcessor).
        datasets: Datasets for training and evaluation.

    Returns:
        Trainer object for training and evaluation.
    """
    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]

        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])

        return data

    training_args: TrainingArguments = create_training_arguments()

    # Partial function to compute metrics
    compute_metrics_fn = partial(
        compute_metrics, image_processor=processor, id2label=ID2LABEL, threshold=0.0
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
    )
