# SPDX-License-Identifier: Apache-2.0
"""restore_leading_dims: the fp8 linear epilogue must not hand a no-op view to
the caller. A 2-D input's output is returned as-is (RowParallelLinear feeds it
straight to an all-reduce, and a view there costs the lazy bridge two DMA
recipes plus a stranded collective launch); higher-rank inputs get the reshape.
CPU-only."""
import torch

from vllm_gaudi.extension.ops import restore_leading_dims


def test_2d_input_returns_output_itself():
    x = torch.randn(3, 8)
    out = torch.randn(3, 5)
    assert restore_leading_dims(out, x) is out


def test_3d_input_restores_leading_dims():
    x = torch.randn(2, 3, 8)
    out = torch.randn(6, 5)
    r = restore_leading_dims(out, x)
    assert r.shape == (2, 3, 5)
    assert r._base is out  # a view, as before
    torch.testing.assert_close(r.reshape(6, 5), out)


def test_1d_input():
    x = torch.randn(8)
    out = torch.randn(5)
    assert restore_leading_dims(out, x) is out
