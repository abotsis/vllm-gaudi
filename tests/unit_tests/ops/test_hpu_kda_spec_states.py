# SPDX-License-Identifier: Apache-2.0
"""Per-position KDA states for MTP speculative decode.

The MTP verify forward scores k+1 candidate tokens per sequence in one pass,
but acceptance is only known after sampling, so the recurrent state must exist
at EVERY candidate position (the next step then loads the accepted one). A
contracted recurrent state cannot be rewound the way the conv cache is
(hpu_causal_conv1d_fn's accepted_offset), so the states have to be produced up
front.

chunk_kda_spec_states claims to do that in ONE call per layer instead of L
sequential kda_decode_step calls -- which matters because decode on this stack
is dispatch-bound, and an L-fold launch increase across 34 KDA layers would
cost more than speculation gains.

Ground truth here is the PRODUCTION decode kernel (kda_decode_step) stepped L
times, so the test pins the new path to what the engine actually computes today.

Run (on a Gaudi node, with vllm_gaudi installed):
  python -m pytest tests/unit_tests/ops/test_hpu_kda_spec_states.py -q
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.hpu_kda_eager import (_spec_masks, chunk_kda_eager, chunk_kda_spec_states, kda_decode_step)

DEV = "hpu" if (hasattr(torch, "hpu") and torch.hpu.is_available()) else "cpu"
H, D = 8, 128  # heads per rank at TP8, KDA head_dim from linear_attn_config


def _inputs(B, L, seed=0, zero_state=False):
    gen = torch.Generator().manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, generator=gen) * 0.5).to(DEV)  # noqa: E731
    q, k, v = mk(B, L, H, D), mk(B, L, H, D), mk(B, L, H, D)
    # g is a log-decay: strictly negative, per channel
    g = -F.softplus(torch.randn(B, L, H, D, generator=gen)).to(DEV)
    beta = torch.rand(B, L, H, generator=gen).to(DEV)
    S0 = (torch.zeros(B, H, D, D) if zero_state else torch.randn(B, H, D, D, generator=gen) * 0.1).to(DEV)
    return q, k, v, g, beta, S0


def _stepped_reference(q, k, v, g, beta, S0):
    """Ground truth: the production single-token kernel, stepped L times."""
    outs, states = [], []
    S = S0.clone()
    for j in range(q.shape[1]):
        out_j, S = kda_decode_step(S, q[:, j], k[:, j], v[:, j], g[:, j], beta[:, j])
        outs.append(out_j.clone())
        states.append(S.clone())
    return torch.stack(outs, 1), torch.stack(states, 2)  # [B,L,H,V], [B,H,L,D,V]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


# L = num_speculative_tokens + 1. L=2 is MTP-1, L=5 is the MTP-4 target.
@pytest.mark.parametrize("L", [2, 3, 5, 8])
@pytest.mark.parametrize("B", [1, 4])
def test_spec_states_match_stepped_decode(L, B):
    q, k, v, g, beta, S0 = _inputs(B, L, seed=L * 10 + B)
    ref_out, ref_states = _stepped_reference(q, k, v, g, beta, S0)
    out, states = chunk_kda_spec_states(q, k, v, g, beta, initial_state=S0)

    assert states.shape == (B, H, L, D, D)
    assert _rel(out, ref_out) < 1e-4, f"outputs diverge: {_rel(out, ref_out)}"
    for j in range(L):
        r = _rel(states[:, :, j], ref_states[:, :, j])
        assert r < 1e-4, f"state at position {j} diverges: {r}"


def test_final_state_matches_stock_chunk_kernel():
    """states[:, :, -1] must equal what chunk_kda_eager returns as final."""
    B, L = 4, 5
    q, k, v, g, beta, S0 = _inputs(B, L, seed=7)
    _, states = chunk_kda_spec_states(q, k, v, g, beta, initial_state=S0)
    _, final = chunk_kda_eager(q,
                               k,
                               v,
                               g=g,
                               beta=beta,
                               chunk_size=L,
                               initial_state=S0,
                               output_final_state=True,
                               use_qk_l2norm_in_kernel=True)
    assert _rel(states[:, :, -1], final) < 1e-4


def test_zero_and_none_initial_state_agree():
    """A fresh sequence (no prior state) must behave like an explicit zero state."""
    B, L = 2, 5
    q, k, v, g, beta, S0 = _inputs(B, L, seed=3, zero_state=True)
    out_z, st_z = chunk_kda_spec_states(q, k, v, g, beta, initial_state=S0)
    out_n, st_n = chunk_kda_spec_states(q, k, v, g, beta, initial_state=None)
    assert _rel(out_n, out_z) < 1e-6
    assert _rel(st_n, st_z) < 1e-6
    ref_out, ref_states = _stepped_reference(q, k, v, g, beta, S0)
    assert _rel(out_z, ref_out) < 1e-4
    assert _rel(st_z, ref_states) < 1e-4


def test_states_are_causal():
    """State j must not depend on tokens after j -- else acceptance leaks future."""
    B, L = 2, 5
    q, k, v, g, beta, S0 = _inputs(B, L, seed=11)
    _, base = chunk_kda_spec_states(q, k, v, g, beta, initial_state=S0)
    # perturb ONLY the last token; states 0..L-2 must be untouched
    v2 = v.clone()
    v2[:, -1] += 10.0
    _, pert = chunk_kda_spec_states(q, k, v2, g, beta, initial_state=S0)
    for j in range(L - 1):
        assert _rel(pert[:, :, j], base[:, :, j]) < 1e-6, f"state {j} saw a future token"
    assert _rel(pert[:, :, -1], base[:, :, -1]) > 1e-3, "last state ignored its own token"


def test_caller_masks_of_wrong_length_are_rejected():
    """A verify batch is not always num_spec+1 long.

    The scheduler can hand the model fewer draft tokens (ngram with a short
    lookup, a partly rejected chain, the tail of a request). The layer caches
    mask buffers per length; if a mismatched set still reached the kernel it
    failed deep in an einsum with "size of tensor a (2) must match tensor b (5)".
    """
    B, L = 2, 2
    q, k, v, g, beta, S0 = _inputs(B, L, seed=5)
    ref_out, ref_states = _stepped_reference(q, k, v, g, beta, S0)

    wrong = _spec_masks(5, q.device)  # masks for L=5, batch is L=2
    out, states = chunk_kda_spec_states(q, k, v, g, beta, initial_state=S0, masks=wrong)
    assert _rel(out, ref_out) < 1e-4
    assert _rel(states[:, :, -1], ref_states[:, :, -1]) < 1e-4
