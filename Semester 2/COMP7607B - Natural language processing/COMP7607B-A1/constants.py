# Data and Preprocessing Constants
TEXT8_DATASET_PATH = "data/text8"  # Replace with your actual path if it's different
MIN_FREQ = 5  # Minimum word frequency for inclusion in the vocabulary
MAX_VOCAB_SIZE = 30000  # Maximum size of the vocabulary

# Model and Training Constants
EMBEDDING_DIM = 128  # Dimension of the word embeddings
WINDOW_SIZE = 5  # Context window size
BATCH_SIZE = 1024  # Batch size for training
NUM_EPOCHS = 2  # Number of training epochs
LEARNING_RATE = 0.0001  # Learning rate for the optimizer

# Other Constants
MODEL_SAVE_PATH = "saved_models"  # Path to save the trained models
