"""Test the python functions from src/mnist."""

import sys

import numpy as np
import pytest
import torch as th

sys.path.insert(0, "./src/")

from src.mnist import cross_entropy, normalize_batch

def test_cross_entropy() -> None:
    """Test if the cross entropy is implemented correctly."""
    label = th.from_numpy(np.random.randint(0, 1, (64, 10)).astype(np.float64))
    out = th.from_numpy(np.random.randn(64, 10))
    out = out.softmax(dim=-1)
    my_result = cross_entropy(label=label, out=out)
    th_result = th.nn.functional.binary_cross_entropy(
        input=out, target=label, reduction="mean"
    )
    assert th.allclose(my_result, th_result)



norm_testdata = [
    (
        th.linspace(0, 5, 10),
        th.tensor(
            [
                -1.4863,
                -1.1560,
                -0.8257,
                -0.4954,
                -0.1651,
                0.1651,
                0.4954,
                0.8257,
                1.1560,
                1.4863,
            ]
        ),
    ),
    (th.linspace(0, 1, 5), th.tensor([-1.2649, -0.6325, 0.0000, 0.6325, 1.2649])),
]


@pytest.mark.parametrize("inpt, res", norm_testdata)
def test_normalize(inpt, res) -> None:
    """Test the normalization."""
    output = normalize_batch(inpt)
    output = th.round(output, decimals=4)
    assert th.allclose(output, res)
