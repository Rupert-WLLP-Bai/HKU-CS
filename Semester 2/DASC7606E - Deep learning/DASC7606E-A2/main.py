import os

from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    """
    Main function to execute model training and evaluation.
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Initialize tokenizer and model
    model = initialize_model()

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()

    raw_datasets = build_dataset()

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # Load and preprocess datasets
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)

    # Build and train the model
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )
    trainer.train()

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
