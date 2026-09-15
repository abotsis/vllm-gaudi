# SPDX-License-Identifier: Apache-2.0
"""Candidate-slot bookkeeping for MTP speculative decode.

Index layout: [bs, 1 + num_spec+1]. Column 0 is the CANONICAL slot -- the
block-table column where prefill stores the post-prompt state and where each
verify step propagates its loaded state. Column 1+j is private candidate slot
j, holding the state after draft token j. ``num_accepted`` selects the load
column directly: 0 is the fresh sentinel (canonical), j >= 1 resumes from
candidate j-1.

Getting these indices wrong is silent in the short-context regime -- the old
layout (candidates at block-table columns 0..k, load at acc-1) coincided with
the canonical column while the whole sequence fit inside mamba block 0, which
is why every short test passed while any prompt past 640 tokens produced
garbage from its first generated token.

Run (on a Gaudi node, with vllm_gaudi installed):
  python -m pytest tests/unit_tests/ops/test_hpu_kda_spec_slots.py -q
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.models.glm5_next import (_kda_load_slots, _kda_spec_decode, _kda_store_slots)
from vllm_gaudi.ops.hpu_kda_eager import kda_decode_step

H, D = 4, 32
POOL = 32


def _mk(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.5  # noqa: E731
    return (r(n * L, H, D), r(n * L, H, D), r(n * L, H, D), -F.softplus(torch.randn(n * L, H, D, generator=g)),
            torch.rand(n * L, H, generator=g))


def test_load_slot_follows_num_accepted():
    # 3 seqs x [canonical + 5 candidates]
    idx = torch.arange(3 * 6, dtype=torch.int32).view(3, 6)
    acc = torch.tensor([0, 3, 5], dtype=torch.int32)
    got = _kda_load_slots(idx, acc, 3)
    # acc=0 -> canonical (col 0); acc=3 -> candidate 2 (col 3); acc=5 -> col 5
    assert got.tolist() == [0, 9, 17]


def test_load_slot_clamps_and_defaults():
    idx = torch.arange(2 * 4, dtype=torch.int32).view(2, 4)
    # num_accepted beyond the column count must clamp, not wrap into another row
    assert _kda_load_slots(idx, torch.tensor([99, 0], dtype=torch.int32), 2).tolist() == [3, 4]
    # no acceptance info -> canonical column
    assert _kda_load_slots(idx, None, 2).tolist() == [0, 4]
    # 1-D (non-speculative) indices pass through untouched
    flat = torch.tensor([9, 8, 7], dtype=torch.int32)
    assert _kda_load_slots(flat, None, 2).tolist() == [9, 8]
    # single-token store goes to the canonical column
    assert _kda_store_slots(idx, 2).tolist() == [0, 4]


@pytest.mark.parametrize("L", [2, 5])
def test_spec_decode_writes_every_candidate_slot(L):
    """Candidate col 1+j holds the state after token j; canonical col gets the
    loaded (pre-step) state; resume column is num_accepted itself."""
    n = 2
    q, k, v, g, beta = _mk(n, L, seed=L)
    pool = torch.randn(POOL, H, D, D)
    # distinct, non-overlapping columns per sequence: [canonical, cand_0..cand_{L-1}]
    idx = torch.arange(n * (L + 1), dtype=torch.int32).view(n, L + 1)
    acc = torch.tensor([0, L], dtype=torch.int32)  # seq0 fresh (canonical), seq1 candidate L-1

    before = pool.clone()
    out = _kda_spec_decode(q,
                           k,
                           v,
                           g,
                           beta,
                           ssm_state=pool,
                           state_indices=idx,
                           store_indices=idx,
                           num_decodes=n,
                           spec_len=L,
                           num_accepted_tokens=acc,
                           out_dtype=torch.float32)
    assert out.shape == (n * L, H, D)

    # Reference: step the production kernel from the same resume column.
    for s in range(n):
        init = before[int(idx[s, int(acc[s])])].clone()
        # the canonical column now holds the loaded state (propagation)
        assert torch.allclose(pool[int(idx[s, 0])], init, atol=1e-6), \
            f"seq {s} canonical column must hold the loaded state"
        S = init.unsqueeze(0)
        for j in range(L):
            t = s * L + j
            o, S = kda_decode_step(S, q[t:t + 1], k[t:t + 1], v[t:t + 1], g[t:t + 1], beta[t:t + 1])
            wrote = pool[int(idx[s, 1 + j])]
            assert torch.allclose(wrote, S[0], atol=1e-4), f"seq {s} candidate {j} holds the wrong state"
            assert torch.allclose(out[t], o[0], atol=1e-4), f"seq {s} token {j} output wrong"


def test_prefill_handoff_resumes_from_canonical_slot():
    """The #15 regression: after prefill, the resume state lives in the
    CANONICAL column (wherever the block table put it -- column 25 for a 16k
    prompt at 640-token mamba blocks), and candidate slots hold stale junk.
    A fresh request (num_accepted=0) must read canonical, not candidate 0."""
    n, L = 1, 3
    q, k, v, g, beta = _mk(n, L, seed=11)
    pool = torch.randn(POOL, H, D, D)
    CANON = 25
    idx = torch.tensor([[CANON, 1, 2, 3]], dtype=torch.int32)  # canonical + 3 candidates
    post_prompt = pool[CANON].clone()

    out = _kda_spec_decode(q,
                           k,
                           v,
                           g,
                           beta,
                           ssm_state=pool,
                           state_indices=idx,
                           store_indices=idx,
                           num_decodes=n,
                           spec_len=L,
                           num_accepted_tokens=torch.tensor([0], dtype=torch.int32),
                           out_dtype=torch.float32)

    S = post_prompt.unsqueeze(0)
    for j in range(L):
        o, S = kda_decode_step(S, q[j:j + 1], k[j:j + 1], v[j:j + 1], g[j:j + 1], beta[j:j + 1])
        assert torch.allclose(out[j], o[0], atol=1e-4), \
            f"token {j}: fresh request did not resume from the canonical slot"


def test_padded_lanes_do_not_clobber_live_slots():
    """Pad lanes carry -1; remainder() wraps them onto a live slot."""
    n, L = 2, 5
    q, k, v, g, beta = _mk(n, L, seed=99)
    pool = torch.randn(POOL, H, D, D)
    idx = torch.stack([torch.arange(L + 1, dtype=torch.int32),
                       torch.full((L + 1, ), -1, dtype=torch.int32)])  # seq1 is padding
    before = pool.clone()
    _kda_spec_decode(q,
                     k,
                     v,
                     g,
                     beta,
                     ssm_state=pool,
                     state_indices=idx,
                     store_indices=idx,
                     num_decodes=n,
                     spec_len=L,
                     num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
                     out_dtype=torch.float32)
    # -1 wraps to POOL-1; that slot belongs to nobody here and must be untouched
    assert torch.equal(pool[POOL - 1], before[POOL - 1]), "pad lane clobbered a live slot"
    # the real sequence's slots did get written
    assert not torch.equal(pool[0], before[0])


def _lazy_mode() -> bool:
    import os
    return os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"


@pytest.mark.skipif(not _lazy_mode(), reason="HPUGraph capture is lazy-mode-only")
def test_draft_graph_core_matches_eager_path():
    """The graphed draft core must be numerically identical to the eager
    draft path it replaces: same modules (private deep copies), same op
    order, attention bypassed in both. Built from small stand-in modules so
    it runs on one card without the 306 GiB checkpoint; the real-model check
    is acceptance parity at boot (0.819/0.315/0.145/0.038)."""
    import torch.nn as nn
    from vllm_gaudi.models.glm5_next_mtp import _DraftGraphCore

    Hd, V = 64, 97

    class Norm(nn.Module):

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.rand(Hd) + 0.5)

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

    class Proj(nn.Module):

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(Hd, 2 * Hd) * 0.05)

        def forward(self, x):
            return x @ self.w.t(), None

    class Mlp(nn.Module):

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(Hd, Hd) * 0.05)

        def forward(self, x):
            return torch.nn.functional.silu(x @ self.w)

    class Mtp(nn.Module):

        def __init__(self):
            super().__init__()
            self.enorm, self.hnorm = Norm(), Norm()
            self.eh_proj = Proj()
            self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()
            self.mlp = Mlp()
            self.shared_head_norm = Norm()

    torch.manual_seed(0)
    mtp = Mtp().to("hpu")
    embed = nn.Embedding(V, Hd).to("hpu")
    core = _DraftGraphCore(mtp, embed)
    import habana_frameworks.torch as htorch
    graphed = htorch.hpu.wrap_in_hpu_graph(core, disable_tensor_cache=True)

    def eager(ids, pos, h):
        e = torch.where(pos.unsqueeze(-1) == 0, 0, embed(ids))
        x, _ = mtp.eh_proj(torch.cat([mtp.enorm(e), mtp.hnorm(h)], -1))
        r = x
        return mtp.shared_head_norm(r + mtp.mlp(mtp.post_attention_layernorm(r)))

    for bs in (1, 4, 8):
        ids = torch.randint(0, V, (bs, 1), device="hpu")
        pos = torch.randint(0, 50, (bs, 1), device="hpu")
        pos[0, 0] = 0
        h = torch.randn(bs, 1, Hd, device="hpu")
        for rep in range(2):  # capture, then replay
            got = graphed(ids, pos, h).cpu()
            want = eager(ids, pos, h).cpu()
            assert torch.allclose(got, want, atol=1e-5), f"bs={bs} rep={rep}: graph != eager"
    # Ownership contract: the light modules are PRIVATE storage clones
    # (mutating the core must not touch the originals) ...
    with torch.no_grad():
        core.eh_proj.w.add_(1.0)
        core.enorm.w.add_(1.0)
    assert not torch.allclose(core.eh_proj.w.cpu(), mtp.eh_proj.w.cpu())
    assert not torch.allclose(core.enorm.w.cpu(), mtp.enorm.w.cpu())
    # ... while the MoE is SHARED by reference (Synapse's fused-MoE op
    # registers expert weights per recipe by identity; a clone fails to
    # compile with "MOE multiplexer weights were partially registered").
    assert core.mlp is mtp.mlp
