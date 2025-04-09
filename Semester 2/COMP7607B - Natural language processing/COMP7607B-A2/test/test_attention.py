import numpy as np
import pytest
import torch

from model.config import LMConfig
from model.model import Attention


@pytest.fixture
def config() -> LMConfig:
    return LMConfig(
        dim=4,
        n_heads=2,
        max_seq_len=10,
        dropout=0.0,  # Disable dropout for deterministic testing
    )


@pytest.fixture
def attention(config: LMConfig) -> Attention:
    attention = Attention(config)

    # fix attention weights
    attention.wq.weight = torch.nn.Parameter(
        torch.tensor(
            [
                [-0.4504, 0.1382, 0.3077, 0.4760],
                [0.3207, -0.2692, -0.0816, -0.3672],
                [-0.2822, -0.2738, 0.3203, -0.2533],
                [0.1325, 0.0396, -0.4821, 0.4144],
            ]
        ),
        requires_grad=True,
    )
    attention.wk.weight = torch.nn.Parameter(
        torch.tensor(
            [
                [0.0326, -0.3946, -0.2646, 0.2665],
                [0.0987, 0.3722, 0.1884, 0.1171],
                [-0.1917, -0.3345, 0.1914, 0.1844],
                [-0.2429, 0.0166, 0.4499, 0.1277],
            ],
            requires_grad=True,
        )
    )
    attention.wv.weight = torch.nn.Parameter(
        torch.tensor(
            [
                [-0.2540, -0.2235, -0.3077, -0.0046],
                [-0.1119, 0.0192, -0.0764, -0.3510],
                [-0.2193, 0.2269, -0.1903, 0.3910],
                [0.1474, 0.0046, 0.3485, 0.4513],
            ]
        ),
        requires_grad=True,
    )
    attention.wo.weight = torch.nn.Parameter(
        torch.tensor(
            [
                [0.4215, -0.0296, -0.0101, 0.1014],
                [-0.3336, -0.1078, 0.0431, -0.3696],
                [0.4910, -0.4030, 0.3858, -0.0447],
                [0.1048, 0.0562, 0.4790, 0.0315],
            ],
            requires_grad=True,
        )
    )
    return attention


@pytest.mark.parametrize(
    "x, pos_cis, expected_output",
    [
        (
            torch.tensor(
                [
                    [
                        [-2.4958, -0.1231, -0.2512, -0.0160],
                        [0.9109, 0.4810, -0.0480, 1.0679],
                        [0.1261, -0.0342, 0.2842, -2.3281],
                        [0.0581, 1.0941, -1.6504, -1.1844],
                    ],
                    [
                        [-0.7906, 0.1733, 0.8797, -1.2594],
                        [-0.9503, 0.1092, -0.5270, 1.0079],
                        [0.4018, 1.9238, -0.7100, -0.5638],
                        [-1.3420, 0.3973, 2.0238, -0.1978],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [1.0000 + 0.0000j],
                    [0.5403 + 0.8415j],
                    [-0.4161 + 0.9093j],
                    [-0.9900 + 0.1411j],
                ]
            ),
            torch.tensor(
                [
                    [
                        [2.4984e-01, -8.3615e-02, 4.7828e-01, 3.4849e-01],
                        [1.0767e-01, -1.1143e-01, 2.9270e-01, 2.2859e-01],
                        [2.6569e-02, 3.5353e-02, 4.5429e-04, 7.0285e-02],
                        [2.8570e-03, 8.8241e-02, -8.1618e-02, 1.4531e-03],
                    ],
                    [
                        [-9.0912e-02, 1.0426e-01, -3.9416e-01, -2.1062e-01],
                        [7.3987e-02, -2.1078e-02, 1.5127e-01, 8.4923e-02],
                        [-4.5539e-02, 9.8852e-02, 1.4697e-02, 9.9672e-02],
                        [-3.7192e-02, 3.2489e-02, -6.9352e-02, 4.5129e-02],
                    ],
                ],
            ),
        ),
    ],
)
def test_attention(
    attention: Attention,
    x: torch.Tensor,
    pos_cis: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    """Test the output shape of the Attention module."""
    with torch.no_grad():
        output, _ = attention(x, pos_cis)
        assert np.allclose(output.numpy(), expected_output.numpy(), rtol=1e-4)
