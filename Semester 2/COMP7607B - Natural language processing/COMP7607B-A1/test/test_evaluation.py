import pytest

from evaluation import get_query_embed
import torch


@pytest.mark.parametrize(
    "w1_embed, w2_embed, w3_embed, expected",
    [
        (
            torch.Tensor([1, 2, 3]),
            torch.Tensor([4, 5, 6]),
            torch.Tensor([7, 8, 9]),
            torch.Tensor([4, 5, 6]),
        ),
        (
            torch.Tensor([3, 2, 1]),
            torch.Tensor([4, 5, 6]),
            torch.Tensor([7, 8, 9]),
            torch.Tensor([6, 5, 4]),
        ),
    ],
)
def test_get_query_embed(
    w1_embed: torch.Tensor,
    w2_embed: torch.Tensor,
    w3_embed: torch.Tensor,
    expected: torch.Tensor,
):
    assert torch.equal(get_query_embed(w1_embed, w2_embed, w3_embed), expected)
