import os
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import load_file

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
    
    # Check what files are actually in the checkpoint directory
    files = os.listdir(checkpoint_path)
    print(f"Files in checkpoint directory: {files}")
    
    # Load model weights from safetensors file directly
    if "model.safetensors" in files:
        try:
            # Use safetensors library to load the model weights
            # Need to install: pip install safetensors
            state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            model.load_state_dict(state_dict)
            print("Model loaded successfully using safetensors!")
            success = True
        except Exception as e:
            print(f"Error loading model with safetensors: {e}")
            success = False
    
    # Alternatively, try loading with weights_only=False (not secure but may work)
    elif "pytorch_model.bin" in files:
        try:
            state_dict = torch.load(
                os.path.join(checkpoint_path, "pytorch_model.bin"),
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                weights_only=False  # Not secure, but may work
            )
            model.load_state_dict(state_dict)
            print("Model loaded successfully using pytorch_model.bin!")
            success = True
        except Exception as e:
            print(f"Error loading from pytorch_model.bin: {e}")
            success = False
    
    # Last resort: try loading the training_args.bin file (not recommended for weights)
    elif "training_args.bin" in files and not success:
        try:
            # Try to load with explicit weights_only=False (not secure but necessary)
            args = torch.load(
                os.path.join(checkpoint_path, "training_args.bin"), 
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                weights_only=False  # Not secure, but needed for this file
            )
            print("Successfully loaded training arguments, but model weights need to be loaded separately.")
        except Exception as e:
            print(f"Error loading training_args.bin: {e}")
    
    # Build trainer with the loaded model
    trainer = build_trainer(
        model=model,  # Use our loaded model
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    
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