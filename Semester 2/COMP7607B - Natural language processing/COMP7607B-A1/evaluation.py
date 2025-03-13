import torch
import matplotlib.pyplot as plt
from model import SkipGramModel
from sklearn.manifold import TSNE


def get_query_embed(
    w1_embed: torch.Tensor,
    w2_embed: torch.Tensor,
    w3_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the query vector for the word analogy.

    The query vector is calculated as the vector difference between the
    two given words, and then adding the third word vector to it.

    Args:
        w1_embed: Embedding vector for the first word.
        w2_embed: Embedding vector for the second word.
        w3_embed: Embedding vector for the third word.

    Returns:
        The query vector.
    """
    # Write Your Code Here
    return w1_embed - w2_embed + w3_embed


def get_embedding(
    model: SkipGramModel,
    word2idx: dict[str, int],
    word: str,
    device: torch.device,
) -> torch.Tensor:
    # Write Your Code Here
    word_idx = torch.tensor([word2idx[word]], dtype=torch.long, device=device)
    return model.u_embeddings(word_idx).squeeze(0)


def word_analogy(
    model: SkipGramModel,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    word1: str,
    word2: str,
    word3: str,
) -> None | list[str]:
    """
    Performs a word analogy task: word1 - word2 + word3 = ?

    Args:
        model: The trained Word2Vec model.
        word2idx: Dictionary mapping words to indices.
        idx2word: Dictionary mapping indices to words.
        word1, word2, word3: The words for the analogy task.

    Returns:
        The top 5 closest words based on cosine similarity.
    """
    # Check if all words are in the vocabulary
    if word1 not in word2idx or word2 not in word2idx or word3 not in word2idx:
        print("One or more words are not in the vocabulary.")
        return None

    with torch.no_grad():
        device = next(model.parameters()).device
        # Get the embeddings for the input words
        w1_embed = get_embedding(model, word2idx, word1, device)
        w2_embed = get_embedding(model, word2idx, word2, device)
        w3_embed = get_embedding(model, word2idx, word3, device)

        # Calculate the query vector
        query_embed = get_query_embed(w1_embed, w2_embed, w3_embed)

        # Get all the embeddings from the model
        all_embeds = model.u_embeddings.weight.data

    # Calculate cosine similarity between the query vector and all embeddings
    similarities = torch.matmul(query_embed, all_embeds.t()) / (
        torch.norm(query_embed) * torch.norm(all_embeds, dim=1)
    )

    # Get the top 5 most similar words (excluding the input words)
    _, closest_idxs = torch.topk(similarities, k=8)
    closest_words: list[str] = [
        idx2word[idx.item()]
        for idx in closest_idxs
        if idx2word[idx.item()] not in [word1, word2, word3]
    ][:10]

    return closest_words


def get_reduced_embeddings(embeddings):
    """
    Apply t-SNE for dimensionality reduction.

    Args:
        embeddings: The word embeddings to reduce.

    Returns:
        The reduced embeddings.
    """
    # Write Your Code Here
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(embeddings)


def visualize_embeddings(
    model: SkipGramModel,
    word2idx: dict[str, int],
    num_words: int = 200,
):
    """
    Visualizes word embeddings using t-SNE.

    This function takes a trained Word2Vec model and a dictionary mapping words to indices,
    and uses t-SNE to reduce the dimensionality of the embeddings to 2D. It then plots the
    reduced embeddings for the top `num_words` most frequent words in the vocabulary.

    Args:
        model: The trained Word2Vec model.
        word2idx: Dictionary mapping words to indices.
        num_words: Number of words to visualize.

    Returns:
        None
    """
    embeddings = model.u_embeddings.weight.data.cpu().numpy()

    # Select a subset of words for visualization
    words = list(word2idx.keys())[:num_words]
    word_indices = [word2idx[word] for word in words]
    selected_embeddings = embeddings[word_indices]

    # Apply t-SNE for dimensionality reduction
    reduced_embeddings = get_reduced_embeddings(selected_embeddings)

    # Plot the embeddings
    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(
            word,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    # Add title and save the plot to a file
    plt.title("Word Embeddings Visualization with t-SNE")
    plt.savefig("word_embed.png")
    plt.show()
