import os

from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from trainer import build_trainer
from utils import not_change_test_dataset
from pprint import pprint

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """
    Main function to execute model training and evaluation.
    """

    # Build the dataset
    raw_datasets = build_dataset()

    # Initialize the image processor
    processor = initialize_processor()

    # Add preprocessing to the dataset
    datasets = add_preprocessing(raw_datasets, processor)

    # Build the object detection model
    model = initialize_model()

    assert not_change_test_dataset(datasets), "You should not change the test dataset"

    # Build and train the model
    trainer = build_trainer(
        model=model,
        processor=processor,
        datasets=datasets,
    )
    trainer.train()

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    pprint("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
