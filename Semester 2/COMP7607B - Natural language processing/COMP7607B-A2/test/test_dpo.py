import pytest
import torch
import torch.nn.functional as F

from trainer import dpo_loss


@pytest.mark.parametrize(
    "ref_probs, probs, beta, expected_loss",
    [
        # Test 1: batch_size=2, seq_len=1, logits=0
        (
            torch.tensor([[-1.0], [-2.0]], dtype=torch.float32),
            torch.tensor([[-0.5], [-1.5]], dtype=torch.float32),
            0.1,
            -F.logsigmoid(torch.tensor(0.0)),
        ),
        # Test 2: batch_size=2, seq_len=1, positive logits
        (
            torch.tensor([[-1.0], [-2.0]], dtype=torch.float32),
            torch.tensor([[-0.4], [-1.6]], dtype=torch.float32),
            0.1,
            -F.logsigmoid(torch.tensor(0.2 * 0.1)),
        ),
        # Test 3: batch_size=4, seq_len=1
        (
            torch.tensor([[-1.0], [-2.0], [-0.5], [-1.5]], dtype=torch.float32),
            torch.tensor([[-0.5], [-1.5], [-0.3], [-1.2]], dtype=torch.float32),
            0.1,
            torch.mean(
                torch.tensor(
                    [
                        -F.logsigmoid(torch.tensor(0.1 * 0.3)),
                        -F.logsigmoid(torch.tensor(0.1 * 0.2)),
                    ],
                    dtype=torch.float32,
                )
            ),
        ),
        # Test 4: batch_size=2, seq_len=2
        (
            torch.tensor([[-1.0, -1.2], [-2.0, -2.2]], dtype=torch.float32),
            torch.tensor([[-0.5, -0.7], [-1.5, -1.7]], dtype=torch.float32),
            0.1,
            -F.logsigmoid(torch.tensor(0.0)),
        ),
        # Test 5: different beta
        (
            torch.tensor([[-1.0], [-2.0]], dtype=torch.float32),
            torch.tensor([[-0.5], [-1.5]], dtype=torch.float32),
            0.2,
            -F.logsigmoid(torch.tensor(0.0 * 0.2)),
        ),
        # Test 6: negative logits
        (
            torch.tensor([[-1.0], [-2.0]], dtype=torch.float32),
            torch.tensor([[-1.5], [-0.5]], dtype=torch.float32),
            0.1,
            -F.logsigmoid(torch.tensor(-2.0 * 0.1)),
        ),
    ],
)
def test_dpo_loss(
    ref_probs: torch.Tensor,
    probs: torch.Tensor,
    beta: float,
    expected_loss: torch.Tensor,
) -> None:
    loss = dpo_loss(ref_probs, probs, beta=beta)
    assert torch.allclose(loss, expected_loss, atol=1e-5)
