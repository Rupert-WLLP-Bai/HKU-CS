import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_loader
from model import SkipGramModel
from trainer import train_model
from evaluation import word_analogy, visualize_embeddings
from constants import (
    TEXT8_DATASET_PATH,
    BATCH_SIZE,
    WINDOW_SIZE,
    MIN_FREQ,
    MAX_VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
)
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    """
    Main function to run the Word2Vec training and evaluation.

    This function sets up the device, loads the data, builds the model,
    optimizes the model, trains the model, and evaluates the model.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dat: DataLoader[tuple[Tensor, Tensor]]a
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = get_loader(
        text_file=args.text_file,
        batch_size=args.batch_size,
        window_size=args.window_size,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )

    # Model
    model = SkipGramModel(
        vocab_size=len(dataloader.dataset.vocabulary.word2idx),
        embedding_dim=args.embedding_dim,
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        model_save_path=args.model_save_path,
    )

    # Load the best model (or the last saved model)
    best_model_path = os.path.join(
        args.model_save_path, f"model_epoch_{args.num_epochs}.pth"
    )  # Assuming the last epoch is the best
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print(f"Loaded best model from {best_model_path}")
    else:
        print(
            f"Warning: Could not find saved model at {best_model_path}. Using the last trained model."
        )

    # Evaluation
    word2idx = dataloader.dataset.vocabulary.word2idx
    idx2word = dataloader.dataset.vocabulary.idx2word

    # Word Analogy Task
    analogies = [
        ("king", "man", "woman"),
        ("good", "better", "bad"),
        ("london", "england", "france"),
    ]

    for word1, word2, word3 in analogies:
        result = word_analogy(
            model=model,
            word2idx=word2idx,
            idx2word=idx2word,
            word1=word1,
            word2=word2,
            word3=word3,
        )
        print(f"{word1} - {word2} + {word3} = {result}")

    # Visualization
    visualize_embeddings(model=model, word2idx=word2idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Word2Vec model.")

    # fmt: off
    # Data and Preprocessing Arguments
    parser.add_argument("--text_file", type=str, default=TEXT8_DATASET_PATH, help="Path to the text corpus file.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Context window size.")
    parser.add_argument("--min_freq", type=int, default=MIN_FREQ, help="Minimum frequency for a word to be included in the vocabulary.")
    parser.add_argument("--max_vocab_size", type=int, default=MAX_VOCAB_SIZE, help="Maximum vocabulary size.")

    # Model and Training Arguments
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH, help="Path to save the trained model.")
    parser.add_argument("--use_neg_sampling", action="store_true", help="Use negative sampling (only for skipgram).")

    args = parser.parse_args()
    main(args)
    # fmt: on
