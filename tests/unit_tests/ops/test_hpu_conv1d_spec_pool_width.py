# SPDX-License-Identifier: Apache-2.0
"""Wide conv pool (speculative decode) must match the narrow pool's semantics.

Under spec decode the conv cache is allocated width-1+num_spec columns so the
decode path can rewind to the accepted position. Prefill must still leave the
same recent history in the last width-1 columns, and must do it with a
graph-safe whole-row index_copy_ rather than a partial-row assignment.
"""
import pytest
import torch
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn

torch.manual_seed(0)
B, DIM, WIDTH, L, NSPEC = 3, 16, 4, 7, 4
NARROW, WIDE = WIDTH - 1, WIDTH - 1 + NSPEC
SLOTS = 8

x = torch.randn(DIM, B * L) * 0.5
w = torch.randn(DIM, WIDTH) * 0.3
bias = torch.randn(DIM) * 0.1
qsl = torch.arange(B + 1, dtype=torch.int32) * L
cache_idx = torch.tensor([0, 1, 2], dtype=torch.int32)


def _run(has_init):
    base = torch.randn(SLOTS, WIDE, DIM) * 0.4
    narrow_pool = base[:, -NARROW:, :].clone().contiguous()
    wide_pool = base.clone().contiguous()
    hstate = torch.ones(B, dtype=torch.int32) if has_init else None

    out_n = hpu_causal_conv1d_fn(x.clone(),
                                 w,
                                 bias,
                                 conv_states=narrow_pool,
                                 query_start_loc=qsl,
                                 cache_indices=cache_idx,
                                 has_initial_state=hstate,
                                 activation="silu")
    out_w = hpu_causal_conv1d_fn(x.clone(),
                                 w,
                                 bias,
                                 conv_states=wide_pool,
                                 query_start_loc=qsl,
                                 cache_indices=cache_idx,
                                 has_initial_state=hstate,
                                 activation="silu")

    assert (out_n - out_w).abs().max().item() < 1e-5, \
        "convolution output differs between narrow and wide pools"
    assert (narrow_pool[:B] - wide_pool[:B, -NARROW:, :]).abs().max().item() < 1e-5, \
        "wide pool's last width-1 columns differ from the narrow pool"


@pytest.mark.parametrize("has_init", [False, True])
def test_wide_pool_matches_narrow(has_init):
    _run(has_init)
