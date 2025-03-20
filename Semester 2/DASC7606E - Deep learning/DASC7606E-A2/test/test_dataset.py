import pytest

from dataset import align_labels_with_tokens


@pytest.mark.parametrize(
    "labels, word_ids, expected",
    [
        # Test case 1: Empty labels and word_ids
        ([], [], []),
        # Test case 2: No special tokens
        ([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        # Test case 3: Special tokens at the beginning
        ([-100, 1, 2, 3, 4], [None, 0, 1, 2, 3], [-100, -100, 1, 2, 3]),
        # Test case 4: Special tokens in the middle
        ([1, 2, -100, 3, 4], [0, 1, None, 2, 3], [1, 2, -100, -100, 3]),
        # Test case 5: Special tokens at the end
        ([1, 2, 3, 4, -100], [0, 1, 2, 3, None], [1, 2, 3, 4, -100]),
        # Test case 6: Tokens inside a word but not at the beginning
        ([1, 2, 3, 4, 5], [0, 1, 1, 2, 3], [1, 2, 2, 3, 4]),
        # Test case 7: Tokens inside a word but not at the beginning with special tokens
        ([1, 2, -100, 3, 4], [0, 1, None, 2, 3], [1, 2, -100, -100, 3]),
    ],
)
def test_align_labels_with_tokens(labels, word_ids, expected):
    """
    Test the align_labels_with_tokens function with various scenarios.

    This function uses parameterized test cases to verify the correct alignment
    of labels with tokenized word IDs, ensuring that special tokens are handled
    appropriately according to the alignment logic.

    Test Cases:
    1. Empty labels and word_ids: Should return an empty list.
    2. No special tokens: Labels should align directly with word IDs.
    3. Special tokens at the beginning: Should result in initial -100 labels.
    4. Special tokens in the middle: Should insert -100 in the aligned labels.
    5. Special tokens at the end: Should append -100 to the aligned labels.
    6. Tokens inside a word but not at the beginning: Should convert B- to I- as needed.
    7. Tokens inside a word but not at the beginning with special tokens: Should handle -100 appropriately.

    Args:
        labels (list): The list of original labels.
        word_ids (list): The list of word IDs including special tokens.
        expected (list): The expected aligned labels.

    Asserts:
        The function verifies that the output of align_labels_with_tokens
        matches the expected aligned labels for each test case.
    """

    assert align_labels_with_tokens(labels, word_ids) == expected
