MODEL_NAME = "microsoft/conditional-detr-resnet-50"
ID2LABEL = {0: 'Coverall', 1: 'Face_Shield', 2: 'Gloves', 3: 'Goggles', 4: 'Mask'}
LABEL2ID = {label: id for id, label in ID2LABEL.items()}

OUTPUT_DIR = "checkpoints"

# Constants for the test dataset, don't change these
TEST_DATASET_ROW_NUMBER = 29  # Number of rows of the test dataset
TEST_DATASET_SIZE_IN_BYTES = 485803949  # Size of the test dataset in bytes
TEST_DATASET_FINGERPRINT = "74025daf4d79fad5"  # Fingerprint of the test dataset
