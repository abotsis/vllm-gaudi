# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU-native fast chunked KDA (Kimi Delta Attention) kernel for GLM-5.3-Flash.

Adapts the GDN chunk solver (hpu_gdn_pytorch.py) to KDA's per-(head, channel)
decay. The numerical crux: intra-chunk pair decay is

    decay[i,j,d] = exp(gsum[i,d] - gsum[j,d])   (<= 1 for i > j, g <= 0)

which must be computed from the DIFFERENCE — the factored form
``k*exp(g) @ (k*exp(-g))^T`` overflows (within-chunk cumsum spans reach -320,
so ``exp(+g)`` is inf in fp32). The difference-based contraction is tiled over
the channel dimension to bound the [tc, tc, D_tile] transient.

The UT-transform inverse reuses ``_hpu_solve_lower_triangular_batched``
(Neumann by default, exact forward-substitution via VLLM_GDN_EXACT_SOLVE=1).

Oracle: vllm_gaudi.ops.hpu_kda_eager.chunk_kda_eager (Phase-1 probe validated
against the transformers reference; see tests/unit_tests/ops/test_hpu_kda.py).

Layout: q/k/v [S, T, H, D]; g [S, T, H, D] (log decay, <= 0); beta [S, T, H];
state [S, H, D, D] fp32. All internal math fp32 (default), output cast to the
input dtype.
"""

from __future__ import annotations

import os

import torch
from vllm_gaudi.ops.hpu_gdn_pytorch import _hpu_solve_lower_triangular_batched


def _kda_prefix_states(m_full, n_t, x0):
    """All per-chunk entry states of the affine recurrence, final state.

    Log-depth operator scan (Hillis-Steele). The per-chunk recurrence
        state_i = m_i @ state_{i-1} + n_i
    is an associative scan over affine operators (m_i, n_i); composing
    adjacent blocks with distance-doubling turns the serial C-step dependency
    chain (the prefill critical path: C dependent [D,D] @ [D,V] matmuls) into
    ceil(log2(C)) steps of fully batched matmuls, followed by one batched
    [tc, D] @ [D, V] contraction.

    Numeric note: composition order differs from the sequential loop (tree
    order), so results are equal within fp32 rounding, not bit-exact;
    VLLM_KDA_SCAN=seq restores the previous order.

    Args:
        m_full: [S, C, H, D, D] per-chunk state-map matrices.
        n_t: [S, C, H, D, V] per-chunk additive terms.
        x0: [S, H, D, V] initial state (fp32).

    Returns:
        (states_pre [S, C, H, D, V] state BEFORE each chunk,
         final_state [S, H, D, V] state after the last chunk) — fp32.
    """
    S, C = m_full.shape[0], m_full.shape[1]
    H, D, Vdim = x0.shape[1:]
    device = x0.device
    Md = m_full.clone()
    Nd = n_t.clone()
    d = 1
    while d < C:
        md = Md[:, d:]  # later-window operator (pre-compose copy)
        # N first: N_i = m_i @ N_{i-d} + n_i, then M_i = m_i @ M_{i-d}
        Nd = torch.cat([Nd[:, :d], torch.matmul(md, Nd[:, :-d]) + Nd[:, d:]], dim=1)
        Md = torch.cat([Md[:, :d], torch.matmul(md, Md[:, :-d])], dim=1)
        d *= 2
    eye_dd = torch.eye(D, dtype=torch.float32, device=device)
    # NOTE (HPU-graph): stride-0 `expand` feeding batched matmul miscompiles on
    # the lazy path (CPU parity is clean, device output garbage) — materialize
    # the identity prefix rows instead of expanding.
    eye_pre = eye_dd.reshape(1, 1, D, D).repeat(S, H, 1, 1).reshape(S, 1, H, D, D)
    zero_v = torch.zeros(S, 1, H, D, Vdim, dtype=torch.float32, device=device)
    # states BEFORE each chunk: prefix up to ci-1 (identity for ci=0)
    Mpre = torch.cat([eye_pre, Md[:, :-1]], dim=1)
    Npre = torch.cat([zero_v, Nd[:, :-1]], dim=1)
    states_pre = torch.matmul(Mpre, x0.unsqueeze(1)) + Npre  # [S,C,H,D,V]
    final_state = (torch.matmul(Md[:, -1], x0) + Nd[:, -1]).contiguous()
    return states_pre, final_state


# Channel tile for the difference-based decay contraction: bounds the
# [SC*H, tc, tc, D_TILE] fp32 transient (~SC*H*tc^2*D_TILE*4 bytes).

_NEUMANN_ITERS = int(os.environ.get("VLLM_KDA_NEUMANN_ITERS", "16"))

# Chunk-recurrence scheduler for hpu_chunk_kda. "seq" (default) is the serial
# loop with the validated fp32 accumulation order. "parallel" composes the
# per-chunk affine operators with a log-depth Hillis-Steele scan — prefills
# drop from C dependent matmuls to ceil(log2(C)) batched steps, BUT the scan
# is miscompiled on the HPU lazy path (serving degenerates to token loops;
# CPU parity is clean, CLONE/EXPAND/CAT compose mains traced at capture do not
# reproduce it) and is therefore OFF by default until a device-side repro
# narrows the trace. Opt in with VLLM_KDA_SCAN=parallel for probing.
_KDA_SCAN_PARALLEL = os.environ.get("VLLM_KDA_SCAN", "seq").strip().lower() \
    in ("parallel", "1", "true")

# Rows per exponent reference in _decay_dot. The per-channel log decay is
# bounded below by the model's gate_lower_bound (-5 per token for GLM-5.3),
# so within a block of 16 rows the cumulative decay moves by at most 80 nats
# and e^{+-80} stays inside fp32 (ln(max) = 88.7). Larger blocks overflow in
# the worst case; smaller ones only add matmul calls.
_DECAY_BLOCK = 16
_DECAY_EXP_MAX = 5.0 * _DECAY_BLOCK


def _decay_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    ga: torch.Tensor,
    gb: torch.Tensor,
) -> torch.Tensor:
    """sum_d a[..., i, d] * b[..., j, d] * exp(ga[..., i, d] - gb[..., j, d]).

    All inputs [B, tc, D]; returns [B, tc, tc]. Only entries with j <= i are
    consumed (the callers apply tril), and ga/gb are per-chunk cumulative log
    decays: non-increasing along the row axis, with a bounded per-row step.

    Computed as a matmul on the MME per block of rows: for rows i in a block
    starting at i0, with the reference c = ga[i0],
        exp(ga_i - gb_j) = exp(ga_i - c) * exp(c - gb_j),
    the first factor is in [e^-80, 1] (ga is non-increasing) and the second is
    <= 1 for j < i0, <= e^80 for j inside the block, and only non-causal
    entries (masked by the caller) could exceed it, so it is clamped at e^80
    to keep them finite. Replaces the previous per-D-tile elementwise product
    (a [B, tc, tc, dt] tensor per tile, ~170 M elements per layer at 512
    tokens, ~54% of prefill device time); the matmul does the D contraction.
    """
    B, tc, D = a.shape
    blocks = []
    for i0 in range(0, tc, _DECAY_BLOCK):
        i1 = min(i0 + _DECAY_BLOCK, tc)
        c = ga[:, i0:i0 + 1, :]  # [B, 1, D] reference per channel
        ai = a[:, i0:i1] * (ga[:, i0:i1] - c).exp()  # [B, blk, D], factors in [e^-80, 1]
        bj = b * (c - gb).clamp_(max=_DECAY_EXP_MAX).exp()  # [B, tc, D], causal entries exact
        blocks.append(torch.matmul(ai, bj.transpose(-1, -2)))  # [B, blk, tc]
    return torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]


def hpu_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
    neumann_iters: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fast chunked KDA prefill.

    q/k/v: [S, T, H, D] (fp32 or bf16); g: [S, T, H, D] log decay (<= 0);
    beta: [S, T, H]. Returns (out [S, T, H, D], final_state [S, H, D, D]).
    """
    if neumann_iters is None:
        neumann_iters = _NEUMANN_ITERS
    in_dtype = q.dtype
    device = q.device
    S, T, H, D = k.shape
    Vdim = v.shape[-1]
    tc = min(chunk_size, T)
    num_chunks = (T + tc - 1) // tc
    padded = num_chunks * tc

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    if use_qk_l2norm_in_kernel:
        inv = torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        q = q * inv
        inv = torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        k = k * inv
    q = q * (D**-0.5)

    # pad to chunk multiple (g padded with zeros: no decay in padding, so the
    # per-chunk cumsum stays constant after the last real token)
    if padded > T:
        pad = padded - T
        q = torch.cat([q, q.new_zeros(S, pad, H, D)], 1)
        k = torch.cat([k, k.new_zeros(S, pad, H, D)], 1)
        v = torch.cat([v, v.new_zeros(S, pad, H, Vdim)], 1)
        beta = torch.cat([beta, beta.new_zeros(S, pad, H)], 1)
        g = torch.cat([g, g.new_zeros(S, pad, H, D)], 1)

    # Per-chunk RE-BASED cumsum (matches the eager reference: cumsum within
    # each chunk after chunk-reshape, so e^{g} factors do not vanish on later
    # chunks). [S, C, tc, H, D] -> cumsum over tc.
    g = g.contiguous()
    gsum = g.reshape(S, num_chunks, tc, H, D).cumsum(dim=2).reshape(S, padded, H, D)

    # [S, C, tc, H, *] -> flatten chunk-head batch
    def ch(x):
        return x.reshape(S, num_chunks, tc, *x.shape[2:])

    q_c, k_c, v_c = ch(q), ch(k), ch(v)
    b_c, gs_c = ch(beta), ch(gsum)

    SC = S * num_chunks
    # [SC, H, tc, D] views for the contraction helper
    q_h = q_c.permute(0, 1, 3, 2, 4).reshape(SC * H, tc, D)
    k_h = k_c.permute(0, 1, 3, 2, 4).reshape(SC * H, tc, D)
    gs_h = gs_c.permute(0, 1, 3, 2, 4).reshape(SC * H, tc, D)
    b_h = b_c.permute(0, 1, 3, 2).reshape(SC * H, tc)
    v_h = v_c.permute(0, 1, 3, 2, 4).reshape(SC * H, tc, Vdim)

    eye_tc = torch.eye(tc, dtype=torch.float32, device=device)

    # ---- Phase A: T = (I + strict_lower(b_i * decay-dot))^{-1} -------------
    dot = _decay_dot(k_h, k_h, gs_h, gs_h)  # [SCH, tc, tc]
    a_lower = torch.tril(dot * b_h.unsqueeze(-1), -1)
    lmat = eye_tc.unsqueeze(0) + a_lower
    t_inv = _hpu_solve_lower_triangular_batched(lmat, eye_tc, use_vectorized=True, neumann_iters=neumann_iters)

    # The reference's UT transform zeroes X's diagonal (triu mask
    # diagonal=0), so A = inv(I + strict(b_i * decay-dot)) exactly — same
    # unit-diagonal form as GDN (no self-retention term).
    br = b_h.unsqueeze(-1)
    rhs_u = v_h * br  # v_j * b_j
    rhs_w = k_h * br * gs_h.exp()  # k_j * b_j * e^{gsum_j} (<=1, safe)
    u = torch.bmm(t_inv, rhs_u)  # [SCH, tc, V]
    w = torch.bmm(t_inv, rhs_w)  # [SCH, tc, D]

    # ---- Phase B: sequential chunk recurrence ------------------------------
    u_all = u.reshape(S, num_chunks, H, tc, Vdim)
    w_all = w.reshape(S, num_chunks, H, tc, D)
    q_all = q_h.reshape(S, num_chunks, H, tc, D)
    k_all = k_h.reshape(S, num_chunks, H, tc, D)
    gs_all = gs_h.reshape(S, num_chunks, H, tc, D)

    gs_last = gs_all[..., -1:, :]  # [S, C, H, 1, D]
    delta = (gs_last - gs_all).exp()  # e^{gsum_last - gsum_t} <= 1
    qe = q_all * gs_all.exp()  # q * e^{gsum} <= |q|

    # A_out[i,j] = tril(sum_d q[i,d] k[j,d] decay_ij, diagonal INCLUDED).
    # The eager reference's intra mask keeps j <= i, and its chunk value is
    # reassigned v <- T @ (v*beta) = u, so out = A@u + (qe - A@w) @ S.
    dot_qk = _decay_dot(q_h, k_h, gs_h, gs_h).reshape(S, num_chunks, H, tc, tc)
    a_out = torch.tril(dot_qk, 0)

    u_h5 = u_all
    w_h5 = w_all
    core = torch.matmul(a_out, u_h5)  # [S,C,H,tc,V]
    c_h = qe - torch.matmul(a_out, w_h5)  # [S,C,H,tc,D]

    # Eager: state' = state*e^{g_last} + (k * delta)^T @ (u - w @ state)
    # -- delta rides the OUTER k factor, so it scales BOTH u and w in the
    # contraction. Fold it once into kk = k * delta.
    kk = k_all * delta  # [S,C,H,tc,D]

    n_t = torch.matmul(kk.transpose(-1, -2), u_h5)  # [S,C,H,D,V]
    r = torch.matmul(kk.transpose(-1, -2), w_h5)  # [S,C,H,D,D]
    alpha = gs_last.exp().squeeze(-2)  # [S,C,H,D]
    eye_d = torch.eye(D, dtype=torch.float32, device=device)
    m_full = alpha.unsqueeze(-1) * eye_d - r  # [S,C,H,D,D]

    x0 = (torch.zeros(S, H, D, Vdim, dtype=torch.float32, device=device)
          if initial_state is None else initial_state.float().contiguous())

    if _KDA_SCAN_PARALLEL:
        states_pre, final = _kda_prefix_states(m_full, n_t, x0)
        out = core + torch.matmul(c_h, states_pre)
        final_state = final if output_final_state else None
    else:
        state = x0
        out = torch.empty(S, num_chunks, H, tc, Vdim, dtype=torch.float32, device=device)
        for ci in range(num_chunks):
            out[:, ci] = core[:, ci] + torch.matmul(c_h[:, ci], state)
            state = torch.matmul(m_full[:, ci], state) + n_t[:, ci]
        final_state = state.to(torch.float32).contiguous() if output_final_state else None

    # [S, C, H, tc, V] -> [S, (C tc), H, V]: permute so time dims merge
    out = out.permute(0, 1, 3, 2, 4).reshape(S, padded, H, Vdim)[:, :T].to(in_dtype)
    return out, final_state
