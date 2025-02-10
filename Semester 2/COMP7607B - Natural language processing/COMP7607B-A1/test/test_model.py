import pytest

from model import SkipGramModel


@pytest.mark.parametrize(
    "vocab_size, embedding_dim, expected",
    [
        (100, 10, (100, 10)),
        (1000, 10, (1000, 10)),
    ],
)
def test_model_embeddings(
    vocab_size,
    embedding_dim,
    expected,
):
    model = SkipGramModel(vocab_size, embedding_dim)
    assert tuple(model.u_embeddings.weight.shape) == expected
    assert tuple(model.v_embeddings.weight.shape) == expected
