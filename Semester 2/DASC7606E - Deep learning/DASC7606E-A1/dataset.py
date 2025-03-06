from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets import load_dataset

def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset for object detection.

    Returns:
        The dataset.

    Below is an example of how to load an object detection dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset("cppe-5")
    if "validation" not in dataset_base:
        split = dataset_base["train"].train_test_split(0.15, seed=1337)
        dataset_base["train"] = split["train"]
        dataset_base["validation"] = split["test"]
    ```

    Ref: https://huggingface.co/docs/datasets/v3.2.0/package_reference/main_classes.html#datasets.DatasetDict

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is consistent with the dataset format expected for object detection.
    For example:
        raw_datasets["test"] = load_dataset("cppe-5", split="test")
    """
    # Write your code here.
    raw_datasets = load_dataset("cppe-5")
    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(0.15, seed=1337)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    return raw_datasets

def add_preprocessing(dataset, processor) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Add preprocessing to the dataset.

    Args:
        dataset: The dataset to preprocess.
        processor: The image processor to use for preprocessing.

    Returns:
        The preprocessed dataset.

    In this function, you can add any preprocessing steps to the dataset.
    For example, you can add data augmentation, normalization or formatting to meet the model input, etc.

    Hint:
    # You can use the `with_transform` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.with_transform

    # You can also use the `map` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map

    # For Augmentation, you can use the `albumentations` library.
    # Ref: https://albumentations.ai/docs/

    from functools import partial

    # Create the batch transform functions for training and validation sets
    train_transform_batch = # Callable for train set transforming with batched samples passed
    validation_transform_batch = # Callable for val/test set transforming with batched samples passed

    # Apply transformations to dataset splits
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)
    """
    # Write your code here.
    return dataset
