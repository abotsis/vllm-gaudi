# SPDX-License-Identifier: Apache-2.0
"""Oracle tests: fast HPU KDA chunk kernel vs eager reference.

The eager reference (vllm_gaudi.ops.hpu_kda_eager.chunk_kda_eager) was
validated against the transformers implementation in Phase 1
(tests/hpu/glm5_next_layer_probe.py, probe 4).

Run (on a Gaudi node, with vllm_gaudi installed and the habana backend importable):
  python -m pytest tests/unit_tests/ops/test_hpu_kda.py -x -q -s
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.hpu_kda_eager import chunk_kda_eager, recurrent_kda_eager
from vllm_gaudi.ops.hpu_kda_pytorch import _kda_prefix_states, hpu_chunk_kda

DEV = "hpu" if torch.hpu.is_available() else "cpu"


def _mk(S, T, H, D, seed=0, g_scale=3.0):
    gtor = torch.Generator().manual_seed(seed)
    q = torch.randn(S, T, H, D, generator=gtor)
    k = torch.randn(S, T, H, D, generator=gtor)
    v = torch.randn(S, T, H, D, generator=gtor)
    beta = torch.rand(S, T, H, generator=gtor)
    g = -torch.rand(S, T, H, D, generator=gtor).abs() * g_scale  # in (-3, 0]
    return q, k, v, g, beta


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


@pytest.mark.parametrize("S,T", [(2, 256), (1, 320), (3, 129)])  # multi-chunk, non-multiple, tail-1
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_vs_eager(S, T, dtype):
    H, D, C = 4, 32, 64
    q, k, v, g, beta = _mk(S, T, H, D, seed=42)
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))

    ref_out, ref_state = chunk_kda_eager(q,
                                         k,
                                         v,
                                         g=g,
                                         beta=beta,
                                         chunk_size=C,
                                         initial_state=None,
                                         output_final_state=True,
                                         use_qk_l2norm_in_kernel=True)
    out, state = hpu_chunk_kda(q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state=None,
                               output_final_state=True,
                               use_qk_l2norm_in_kernel=True,
                               chunk_size=C)

    cos_o, cos_s = _cos(out, ref_out), _cos(state, ref_state)
    thr_o, thr_s = (0.999, 0.999) if dtype == torch.float32 else (0.995, 0.995)
    print(f"PASS? S={S} T={T} {str(dtype).split('.')[-1]}: out cos={cos_o:.5f} state cos={cos_s:.5f}")
    assert cos_o >= thr_o, f"out cos={cos_o}"
    assert cos_s >= thr_s, f"state cos={cos_s}"


def test_initial_state_nonzero():
    """Continuation prefill: chunk kernel must consume an initial state."""
    S, T, H, D, C = 2, 128, 4, 32, 64
    q, k, v, g, beta = _mk(S, T, H, D, seed=7)
    init = torch.randn(S, H, D, D) * 0.05

    ref_out, ref_state = chunk_kda_eager(q.float(),
                                         k.float(),
                                         v.float(),
                                         g=g.float(),
                                         beta=beta.float(),
                                         chunk_size=C,
                                         initial_state=init,
                                         output_final_state=True,
                                         use_qk_l2norm_in_kernel=True)
    out, state = hpu_chunk_kda(q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state=init,
                               output_final_state=True,
                               use_qk_l2norm_in_kernel=True,
                               chunk_size=C)
    print(f"init-state: out cos={_cos(out, ref_out):.5f} state cos={_cos(state, ref_state):.5f}")
    assert _cos(out, ref_out) >= 0.999
    assert _cos(state, ref_state) >= 0.999


def test_chunk_decode_handoff():
    """Prefill (fast chunk) -> 1 decode step (eager recurrent) continuity."""
    S, T, H, D, C = 2, 256, 4, 32, 64
    q, k, v, g, beta = _mk(S, T, H, D, seed=11)
    _, state = hpu_chunk_kda(q, k, v, g, beta, None, True, True, C)

    qn = torch.randn(S, 1, H, D)
    kn = torch.randn(S, 1, H, D)
    vn = torch.randn(S, 1, H, D)
    bn = torch.rand(S, 1, H)
    gn = -torch.rand(S, 1, H, D).abs() * 3

    step_out, _ = recurrent_kda_eager(qn,
                                      kn,
                                      vn,
                                      g=gn,
                                      beta=bn,
                                      initial_state=state,
                                      output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)

    # ground truth: full sequence in one eager recurrent pass
    qf = torch.cat([q, qn], 1)
    kf = torch.cat([k, kn], 1)
    vf = torch.cat([v, vn], 1)
    gf = torch.cat([g, gn], 1)
    bf = torch.cat([beta, bn], 1)
    full_out, _ = recurrent_kda_eager(qf,
                                      kf,
                                      vf,
                                      g=gf,
                                      beta=bf,
                                      initial_state=None,
                                      output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)
    cos = _cos(step_out, full_out[:, -1:])
    print(f"handoff token cos={cos:.5f}")
    assert cos >= 0.999


def test_long_decay_stability():
    """Aggressive decay (g up to -5, 8 chunks) must not inf/NaN via the
    difference-based contraction."""
    S, T, H, D, C = 1, 512, 4, 32, 64
    q, k, v, g, beta = _mk(S, T, H, D, seed=13, g_scale=5.0)
    out, state = hpu_chunk_kda(q, k, v, g, beta, None, True, True, C)
    assert torch.isfinite(out.float()).all(), "output has non-finite values"
    assert torch.isfinite(state).all(), "state has non-finite values"
    ref_out, _ = chunk_kda_eager(q,
                                 k,
                                 v,
                                 g=g,
                                 beta=beta,
                                 chunk_size=C,
                                 initial_state=None,
                                 output_final_state=False,
                                 use_qk_l2norm_in_kernel=True)
    cos = _cos(out, ref_out)
    print(f"long-decay cos={cos:.5f}")
    assert cos >= 0.999


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


def test_batched_decode_step():
    """3c: vectorized decode step over N sequences (incl. spec-decode rows)
    vs the eager recurrent oracle run per sequence."""
    from vllm_gaudi.ops.hpu_kda_eager import kda_decode_step as step_ref
    # kda_decode_step (eager module) IS the batched impl; compare against the
    # sequential recurrent scan for correctness at batch+spec shapes.
    N, H, D = 5, 4, 32
    torch.manual_seed(5)
    states = torch.randn(N, H, D, D) * 0.05
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)
    g = -torch.rand(N, H, D).abs() * 5
    beta = torch.rand(N, H)
    out, states2 = step_ref(states.clone(), q, k, v, g, beta)
    # oracle: per-sequence recurrent single step
    for n in range(N):
        o1, s1 = recurrent_kda_eager(q[n:n + 1].unsqueeze(1),
                                     k[n:n + 1].unsqueeze(1),
                                     v[n:n + 1].unsqueeze(1),
                                     g=g[n:n + 1].unsqueeze(1),
                                     beta=beta[n:n + 1].unsqueeze(1),
                                     initial_state=states[n:n + 1],
                                     output_final_state=True,
                                     use_qk_l2norm_in_kernel=True)
        cos_o = _cos(out[n], o1[0, 0])
        cos_s = _cos(states2[n], s1[0])
        assert cos_o >= 0.999, f"seq {n} out cos={cos_o}"
        assert cos_s >= 0.999, f"seq {n} state cos={cos_s}"
    print(f"batched decode step N={N}: all cos >= 0.999")

# ---------------------------------------------------------------------------
# Parallel (Hillis-Steele) chunk scan vs the sequential recurrence reference
# ---------------------------------------------------------------------------

def _scan_seq_reference(m_full, n_t, x0):
    """states BEFORE each chunk + final state, computed by the plain loop."""
    C = m_full.shape[1]
    out_states = []
    state = x0
    for ci in range(C):
        out_states.append(state)
        state = torch.matmul(m_full[:, ci], state) + n_t[:, ci]
    return torch.stack(out_states, 1), state


@pytest.mark.parametrize("S,C,H,D,V", [(1, 48, 8, 128, 128), (2, 25, 4, 64, 128), (1, 1, 4, 32, 32), (1, 2, 8, 16, 16)])
def test_kda_scan_matches_sequential(S, C, H, D, V):
    """Prefix states of the Hillis-Steele operator scan == the sequential loop.

    C == 1 must match bit-exactly (no composition). Deeper chains differ by
    fp32 reordering (tree composition folds +n into the composed map), so the
    check bounds the error relative to the state Frobenius norm — scale-free
    and cancellation-robust.
    """
    gtor = torch.Generator().manual_seed(11)
    m_full = torch.randn(S, C, H, D, D, generator=gtor) * 0.05 + torch.eye(D)
    n_t = torch.randn(S, C, H, D, V, generator=gtor) * 0.3
    x0 = torch.randn(S, H, D, V, generator=gtor)
    got_pre, got_final = _kda_prefix_states(m_full, n_t, x0)
    want_pre, want_final = _scan_seq_reference(m_full, n_t, x0)
    if C == 1:
        # a single chunk: no composition happens, bit-identical
        torch.testing.assert_close(got_pre, want_pre, rtol=0, atol=0)
        torch.testing.assert_close(got_final, want_final, rtol=0, atol=0)
    else:
        # fp32 reordering (tree composition folds +n into the composed map, the
        # loop adds it after); pointwise bounds die on cancellation, so bound
        # the error relative to the state Frobenius norm — scale-free.
        rel_pre = ((got_pre - want_pre).norm() / (want_pre.norm() + 1e-12)).item()
        rel_final = ((got_final - want_final).norm() / (want_final.norm() + 1e-12)).item()
        assert rel_pre < 1e-4, rel_pre
        assert rel_final < 1e-4, rel_final
