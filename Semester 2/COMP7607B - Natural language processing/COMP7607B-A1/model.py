import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Skip-gram model implementation.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size  # Size of the vocabulary
        self.embedding_dim = embedding_dim  # Dimension of the word embeddings

        # Input embedding layer (target word)
        # Maps a word to its representation in the embedding space
        self.u_embeddings: nn.Embedding = self._create_u_embeddings(
            vocab_size,
            embedding_dim,
        )
        # Output embedding layer (context word)
        # Maps a word to its representation in the embedding space
        self.v_embeddings: nn.Embedding = self._create_v_embeddings(
            vocab_size,
            embedding_dim,
        )

        # Initialize the weights of the embedding layers
        self.init_weights()

    def _create_u_embeddings(self, vocab_size: int, embedding_dim: int) -> nn.Embedding:
        # Write Your Code Here
        pass

    def _create_v_embeddings(self, vocab_size: int, embedding_dim: int) -> nn.Embedding:
        # Write Your Code Here
        pass

    def init_weights(self):
        """
        Initializes the embedding weights for the input and output embedding layers.

        This method is called when the model is initialized and
        the weights of the embedding layers are set to random values.
        """
        # Initialize the input embedding weights (target word)
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)

        # Initialize the output embedding weights (context word)
        self.v_embeddings.weight.data.zero_()

    def forward(self, target_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Skip-gram model.

        This method takes in a tensor of target word indices and returns a tensor of predicted
        scores for context words.

        Args:
            target_words: Tensor of target word indices. Shape: (batch_size)

        Returns:
            Tensor of predicted scores for context words. Shape: (batch_size, vocab_size)
        """
        # Get the embeddings for the target words
        # Shape: (batch_size, embedding_dim)
        u_embeds = self.u_embeddings(target_words)

        # Calculate the scores (dot product with all output embeddings)
        # Shape: (batch_size, vocab_size)
        scores = torch.matmul(u_embeds, self.v_embeddings.weight.t())

        return scores
