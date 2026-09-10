# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager KDA (Kimi-style Delta Attention) kernels for GLM-5.3-Flash on HPU.

Phase-2 correctness-first implementation: pure PyTorch, runs on any device.
Ported from the transformers 5.16 reference (glm5_next chunk/recurrent KDA);
kept in-tree so vllm-gaudi does not depend on bleeding-edge transformers
internals and so Phase 3 can swap in a fast HPU chunk kernel behind the same
signatures.

Signatures follow the transformers reference so the Phase-1 probe oracle
(tests/hpu/glm5_next_layer_probe.py) validates these directly.

Layout: q/k/v/g: [B, S, H, D]; beta: [B, S, H]; state: [B, H, D, D] fp32.
All math in fp32 internally (states are rounding-sensitive).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def chunk_kda_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked KDA prefill. Port of transformers chunk_kimi_delta_attention."""
    initial_dtype = query.dtype
    q, k, v, beta_, g_ = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)

    batch_size, num_heads, sequence_length, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    scale = 1 / (q.shape[-1]**0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total = sequence_length + pad_size

    q = F.pad(q, (0, 0, 0, pad_size)) * scale
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    g_ = F.pad(g_, (0, 0, 0, pad_size))
    beta_ = F.pad(beta_, (0, pad_size))
    v_beta = v * beta_.unsqueeze(-1)
    k_beta = k * beta_.unsqueeze(-1)

    q, k, v, g_, k_beta, v_beta = (x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                   for x in (q, k, v, g_, k_beta, v_beta))
    beta_ = beta_.reshape(beta_.shape[0], beta_.shape[1], -1, chunk_size)

    g_ = g_.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    decay_mask = (g_.unsqueeze(-2) - g_.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_.exp())

    last_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=v.dtype, device=v.device)
                  if initial_state is None else initial_state.to(v).contiguous())
    out = torch.zeros_like(v)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(total // chunk_size):
        q_i, k_i, v_i, g_i = q[:, :, i], k[:, :, i], v[:, :, i], g_[:, :, i]
        attn_inter = (q_i * g_i.exp()) @ last_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        out[:, :, i] = attn_inter + attn_intra @ v_new
        last_state = (last_state * g_i[:, :, -1].exp().unsqueeze(-1) +
                      (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new)

    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :sequence_length]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, (last_state if output_final_state else None)


def recurrent_kda_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequential KDA scan (reference / debug). S=len may be > 1."""
    initial_dtype = query.dtype
    q, k, v, g_, b = (x.to(torch.float32) for x in (query, key, value, g, beta))
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
    B, S, H, D = k.shape
    V = v.shape[-1]
    q = q / (D**0.5)
    state = (torch.zeros(B, H, D, V, dtype=v.dtype, device=v.device)
             if initial_state is None else initial_state.to(v).contiguous())
    out = torch.zeros(B, S, H, V, dtype=v.dtype, device=v.device)
    for i in range(S):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        g_i = g_[:, i].exp()  # [B,H,D] decay (per channel)
        b_i = b[:, i][..., None]  # [B,H,1]
        state = state * g_i[..., None]  # decay along k-dim
        kv_mem = (state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)
    return out.to(initial_dtype), (state if output_final_state else None)


@torch._dynamo.disable
def kda_decode_step(
    states: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized single-token KDA step over all decode sequences.

    states: [N, H, D, V] fp32 (in/out — updated in place)
    q/k/v:   [N, H, D]
    g:       [N, H, D] (log decay, negative)
    beta:    [N, H]
    Returns (out [N, H, V], states).
    """
    q = q.float()
    k = k.float()
    v = v.float()
    if use_qk_l2norm:
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
    q = q / (q.shape[-1]**0.5)
    g_dec = g.float().exp()  # [N,H,D]
    b = beta.float()[..., None]  # [N,H,1]

    states *= g_dec[..., None]  # decay k-dim
    # Both readouts are batched matvecs over the [D] axis of the [N,H,D,V]
    # state. Writing them as broadcast-multiply + reduce materializes a full
    # [N,H,D,V] fp32 intermediate (2 MB per layer at H=32, D=V=128) purely to
    # sum it away -- twice per layer, x34 KDA layers per decode step. matmul
    # keeps it on the MME with no intermediate. Verified against a float64
    # reference over 200 random trials: max out error 4.9e-08, below fp32 eps
    # (1.19e-07); the state update below is unchanged.
    kv_mem = torch.matmul(k.unsqueeze(-2), states).squeeze(-2)  # [N,H,V]
    delta = (v - kv_mem) * b
    states += k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = torch.matmul(q.unsqueeze(-2), states).squeeze(-2)  # [N,H,V]
    return out, states


# Constant masks, built ONCE per (length, device) and cached.
#
# Under HPU graphs the captured graph holds pointers to the tensors it saw at
# capture time. Rebuilding these on every call lets the captured ones be freed,
# and replay then fails with "Neither storage attached to input tensor, not its
# view". Keeping them alive in this cache is what makes the spec path
# graph-safe; it also skips rebuilding four masks per KDA layer per step.
_SPEC_MASK_CACHE: dict = {}


def _spec_masks(length: int, device):
    key = (length, str(device))
    cached = _SPEC_MASK_CACHE.get(key)
    if cached is None:
        ones = torch.ones(length, length, dtype=torch.bool, device=device)
        cached = (
            torch.eye(length, dtype=torch.float32, device=device),
            torch.triu(ones, 1),  # strictly upper
            torch.triu(ones, 0),  # upper incl. diagonal
            torch.tril(torch.ones(length, length, dtype=torch.float32, device=device)),
        )
        _SPEC_MASK_CACHE[key] = cached
    return cached


def chunk_kda_spec_states(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    masks: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA over q_len = L tokens per sequence, returning the state at EVERY position.

    This is the speculative-decode (MTP) verify path. The target scores k+1
    candidate tokens per sequence in one forward, but which of them are accepted
    is only known after sampling -- so the recurrent state must be available at
    each candidate position, and the next step loads the accepted one. (Unlike
    the conv cache, a contracted recurrent state cannot be rewound; see
    hpu_causal_conv1d_fn's accepted_offset trick, which has no analogue here.)

    ``chunk_kda_eager`` materializes ``last_state`` only at chunk boundaries --
    computing outputs *without* stepping the state is the point of the chunked
    form -- so the per-position states are not a byproduct of it. They are
    recoverable in closed form from the same ``v_new`` it already computes:

        S_j = S_0 * exp(g_j) + sum_{t<=j} (k_t * exp(g_j - g_t)) (x) v_new_t

    which is one causal-masked einsum. That keeps this to ~1 call per layer
    instead of L sequential ``kda_decode_step`` calls -- decode here is
    dispatch-bound, so an L-fold launch increase across 34 KDA layers would
    cost more than speculation gains.

    Shapes: q/k/v/g [B, L, H, D], beta [B, L, H], initial_state [B, H, D, V].
    Returns (out [B, L, H, V], states [B, H, L, D, V]) with states[:, :, j] the
    state after consuming token j.  states[:, :, -1] equals the final state
    ``chunk_kda_eager`` would return for the same inputs.
    """
    initial_dtype = query.dtype
    q, k, v, b, gc = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
    B, H, L, D = k.shape
    V = v.shape[-1]
    q = q * (D**-0.5)
    if initial_state is None:
        S0 = torch.zeros(B, H, D, V, dtype=torch.float32, device=k.device)
    else:
        S0 = initial_state.to(torch.float32).contiguous()

    v_beta = v * b.unsqueeze(-1)
    k_beta = k * b.unsqueeze(-1)
    g_cum = gc.cumsum(dim=-2)  # [B,H,L,D]
    decay = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)).exp()  # [B,H,L,L,D]

    # Guard the caller's masks against a length mismatch rather than failing
    # deep in an einsum: verify batches are not always num_spec+1 long.
    if masks is not None and masks[0].shape[0] != L:
        masks = None
    eye, strict_upper, upper, causal = masks if masks is not None else _spec_masks(L, k.device)
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(upper, 0)

    # attn is strictly lower triangular, hence nilpotent (attn**L == 0), so
    # (I - attn)^-1 = I + attn + attn^2 + ... + attn^(L-1) EXACTLY. Repeated
    # squaring gets there in ceil(log2(L-1)) matmuls -- no solve_triangular
    # (unsupported on HPU) and no python forward-substitution loop like the
    # one chunk_kda_eager runs over chunk_size.
    inv = eye + attn
    p = attn
    for _ in range(max(0, (L - 1).bit_length() - 1)):
        p = p @ p
        inv = inv + inv @ p

    v_new = (inv @ v_beta) - (inv @ (k_beta * g_cum.exp())) @ S0  # [B,H,L,V]

    attn_intra = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(strict_upper, 0)
    out = (q * g_cum.exp()) @ S0 + attn_intra @ v_new  # [B,H,L,V]

    # S_j = S0*exp(g_j) + sum_{t<=j} k_t*exp(g_j-g_t) (x) v_new_t
    w = k.unsqueeze(-3) * decay * causal[None, None, :, :, None]  # [B,H,L,L,D]
    states = (S0.unsqueeze(2) * g_cum.exp().unsqueeze(-1) + torch.einsum("bhjtd,bhtv->bhjdv", w, v_new))

    return out.transpose(1, 2).contiguous().to(initial_dtype), states
