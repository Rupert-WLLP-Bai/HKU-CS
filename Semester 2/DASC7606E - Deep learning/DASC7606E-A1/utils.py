from datasets import DatasetDict

from constants import (
    TEST_DATASET_FINGERPRINT,
    TEST_DATASET_ROW_NUMBER,
    TEST_DATASET_SIZE_IN_BYTES,
)


def not_change_test_dataset(raw_datasets: DatasetDict) -> bool:
    """
    Check if the test dataset is not changed.

    Args:
        raw_datasets: Raw datasets.

    Returns:
        True if the test dataset is not changed, False otherwise.
    """
    raw_datasets_test = raw_datasets["test"]
    return (
        raw_datasets_test.num_rows == TEST_DATASET_ROW_NUMBER
        and raw_datasets_test.size_in_bytes == TEST_DATASET_SIZE_IN_BYTES
        and raw_datasets_test._fingerprint == TEST_DATASET_FINGERPRINT
    )
