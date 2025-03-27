import os
from transformers import Trainer

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds
from constants import OUTPUT_DIR

def evaluate_model(checkpoint_path=None):
    """
    Load a trained model from checkpoints and evaluate on test set.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint directory.
                               If None, uses the default OUTPUT_DIR.
    """
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Use provided checkpoint path or default
    if checkpoint_path is None:
        # Find the best checkpoint in the output directory
        checkpoint_path = OUTPUT_DIR
        # Check if there's a "best" directory in the checkpoints
        if os.path.exists(os.path.join(OUTPUT_DIR, "best")):
            checkpoint_path = os.path.join(OUTPUT_DIR, "best")
        # If not, find the latest checkpoint directory
        else:
            checkpoint_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint')]
            if checkpoint_dirs:
                latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
                checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Build datasets
    raw_datasets = build_dataset()
    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"
    
    # Load and preprocess datasets
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)
    
    # Initialize model with correct number of labels
    model = initialize_model()
    
    # Build trainer without training
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    
    # Load the trained weights
    trainer.model = type(model).from_pretrained(checkpoint_path)
    
    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    print("\nTest Metrics:", test_metrics)
    
    return test_metrics

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    evaluate_model()