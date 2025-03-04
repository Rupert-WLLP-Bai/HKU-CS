# Data and Preprocessing Constants
TEXT8_DATASET_PATH = "data/text8"  # Replace with your actual path if it's different
MIN_FREQ = 3  # Minimum word frequency for inclusion in the vocabulary
MAX_VOCAB_SIZE = 50000  # Maximum size of the vocabulary

# Model and Training Constants
EMBEDDING_DIM = 300  # Dimension of the word embeddings
WINDOW_SIZE = 10  # Context window size
BATCH_SIZE = 512  # Batch size for training
NUM_EPOCHS = 10  # Number of training epochs
LEARNING_RATE = 0.002  # Learning rate for the optimizer

# Other Constants
MODEL_SAVE_PATH = "saved_models"  # Path to save the trained models
