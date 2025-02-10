import pytest

from dataset import Word2VecDataset, Word2VecConfig


@pytest.mark.parametrize(
    "text_file, window_size, min_freq, max_vocab_size, expected_size",
    [
        ("test/data/for_dataset.txt", 2, 0, 3, 26),
        ("test/data/for_dataset.txt", 2, 3, 3, 18),
        ("test/data/for_dataset.txt", 2, 3, 1, 6),
        ("test/data/for_dataset.txt", 3, 3, 3, 24),
    ],
)
def test_dataset(
    text_file,
    window_size,
    min_freq,
    max_vocab_size,
    expected_size,
):
    config = Word2VecConfig(window_size, min_freq, max_vocab_size)
    dataset = Word2VecDataset(text_file, config)
    assert len(dataset) == expected_size
