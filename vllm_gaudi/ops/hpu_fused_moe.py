from collections.abc import Callable
from enum import Enum
from functools import partial
import contextlib
import os
from collections import OrderedDict
from typing import Union

from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner as MoERunnerBase, )
import torch
import vllm
import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter, )
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter, )
import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOp
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_clamp_swiglu import clamp_swiglu
from vllm_gaudi.utils import has_quant_config
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

# Map model activation names to the activation strings the Gaudi fused-MoE op's
# MoeActivationMode_t enum accepts. The synapse MoE kernel enumerates "gelu" (and
# silu/selu) but NOT "gelu_tanh"; the Gaudi gelu TPC kernel is itself the tanh
# approximation (tpc_kernels .../gelu_f16.c uses tanh_f16), so HF's
# gelu_pytorch_tanh / gelu_tanh map to the op's "gelu" (the ~1e-3 exact-vs-tanh
# difference does not affect inference). Without this, Gemma4's MoE aborts at
# warmup_graphs with: Activation "gelu_tanh" not found among MoeActivationMode_t.
_MOE_ACTIVATION_ALIASES = {
    "gelu_tanh": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "gelu_new": "gelu",
    "quick_gelu": "gelu",
    # Non-gated squared-ReLU (Nemotron-H). The model's activation config value is
    # "relu2_no_mul"; the Habana MoeActivationMode_t enum spells it "relu2". The
    # no-gate/no-multiply behaviour is selected separately via is_gated=False
    # (see VllmMixtureOfExpertsOpFP8PerChannel), not by this activation name.
    "relu2_no_mul": "relu2",
}


def _normalize_moe_activation(activation):
    act = activation.value if isinstance(activation, Enum) else activation
    return _MOE_ACTIVATION_ALIASES.get(act, act)


# Activation string used by MiniMax-M3 experts. The Habana fused-MoE kernel
# (torch.ops.hpu.mixture_of_experts) only supports silu/gelu, so this
# activation is computed unfused in PyTorch instead (see _unfused_swigluoai_moe).
_SWIGLU_OAI_ACTIVATION = "swigluoai_uninterleave"

# The unfused SwiGLU-OAI dense expert pass materializes [E_local, T, 2*I]
# intermediates. For large prompt buckets (T up to the max batched-token bucket)
# a single such graph exceeds the Synapse compiler's limits and fails to compile
# with ``synStatus 26 [Generic failure]``. Tiling the token dimension bounds the
# size of every fused graph/op while keeping the math and per-tile shapes static,
# which the HPU graph compiler requires. Tunable via env for large-prompt tuning.
_SWIGLU_OAI_TOKEN_TILE = gaudi_envs.VLLM_MINIMAX_M3_MOE_TOKEN_TILE


def _unfused_swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Unfused expert MLP with SwiGLU-OAI activation for MiniMax-M3 on HPU.

    Replaces a single ``moe_op(x, topk_ids, topk_weights, activation=...)`` fused
    call for the ``swigluoai_uninterleave`` activation (unsupported by the Habana
    MoE kernel). Returns the same per-rank (pre-all-reduce) partial as the fused
    op, so the surrounding MoERunner reduction/combine pipeline is unchanged.

    A dense pass is used (every token through every local expert, masked by the
    routing weight) to keep static shapes, which the HPU graph compiler requires
    -- data-dependent gather/indexing would produce dynamic shapes. It is
    vectorized as a sync-free expert-chunked ``bmm`` over the stacked expert
    weights rather than a per-expert Python loop: the loop issues one matmul per
    expert and serializes the HPU graph (~0.3-1.4 tok/s), while the chunked
    ``bmm`` batches experts with static shapes (~11 tok/s). The SwiGLU-OAI
    activation is computed in fp32 for accuracy (matches
    ``minimax_m3_sparse.swiglu_oai``).

    Args:
        layer: the ``FusedMoE``/``RoutedExperts`` layer. Reads the stacked
            per-rank expert weights ``layer.w13_weight`` [E_local, 2*I, H] and
            ``layer.w2_weight`` [E_local, H, I] (the same tensors that seed
            ``moe_op.w13_list``) and ``layer.moe_config.ep_rank`` /
            ``layer.local_num_experts`` for this rank's expert-id offset.
        x: [T, H] flattened activations.
        topk_ids: [T, K] selected (global) expert ids.
        topk_weights: [T, K] routing weights.
    """
    w13 = layer.w13_weight  # [E_local, 2*I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    # First global expert id owned by this (EP) rank; matches the experts_min
    # that process_weights_after_loading passes to VllmMixtureOfExpertsOp.
    if x.numel() == 0:
        return x.new_empty(x.shape[0], layer.w2_weight.shape[-1] if layer.w2_weight.dim() == 3 else x.shape[-1])
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    d = w13.shape[1] // 2
    hidden = x.shape[-1]
    tokens = x.shape[0]

    # Per-(token, local-expert) combine weight, computed sync-free via scatter
    # (no host sync, unlike an ``== expert_id`` comparison inside a Python loop).
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Dense experts via expert-chunked bmm (static shapes, no host sync). Only
    # this rank's local experts contribute; cross-rank combine happens outside.
    # The token dimension is tiled so no single fused graph materializes the full
    # [E_local, T, 2*I] intermediate (which overflows the Synapse compiler and
    # fails with synStatus 26 for large prompt buckets); per-tile shapes stay
    # static and tiles are concatenated back into the original [T, H] output.
    chunk = 16  # dense/packed bmm chunk (also set in the dense branch)
    token_tile = _SWIGLU_OAI_TOKEN_TILE
    if token_tile <= 0 or token_tile >= tokens:
        token_tile = tokens
    out_tiles = []
    for tstart in range(0, tokens, token_tile):
        tend = min(tstart + token_tile, tokens)
        xt = x[tstart:tend]  # [t, H]
        t = xt.shape[0]
        acc = torch.zeros_like(xt)  # [t, H]
        for start in range(0, e_local, chunk):
            end = min(start + chunk, e_local)
            c = end - start
            w13c = w13[start:end].transpose(1, 2)  # [c, H, 2I]
            w2c = w2[start:end].transpose(1, 2)  # [c, I, H]
            xe = xt.unsqueeze(0).expand(c, t, hidden)  # [c, t, H]
            h = torch.bmm(xe, w13c)  # [c, t, 2I]
            # SwiGLU-OAI in fp32 for accuracy: gate=first half, up=second half.
            g = h[..., :d].float()
            u = h[..., d:].float()
            if limit is not None:
                g = g.clamp(max=limit)
                u = u.clamp(min=-limit, max=limit)
            act = (g * torch.sigmoid(alpha * g) * (u + beta)).to(x.dtype)  # [c, t, I]
            y = torch.bmm(act, w2c)  # [c, t, H] per-rank partial
            gate_wc = gate_w[tstart:tend, start:end].t().unsqueeze(-1)  # [c, t, 1]
            acc = acc + (y.float() * gate_wc).sum(0).to(x.dtype)
        out_tiles.append(acc)
    return out_tiles[0] if len(out_tiles) == 1 else torch.cat(out_tiles, dim=0)


# --- Decode-time expert gather (bandwidth reduction) -------------------------
# The dense pass above reads EVERY local expert's weights for every token
# (E_local expert-weight reads) even though each token routes to only top_k
# experts. At small token counts (single-stream / low-concurrency decode) the
# number of DISTINCT local experts the batch touches is at most ``tokens * K``,
# which can be far below E_local. This path gathers only those experts' weights
# via ``index_select`` so only ``G = min(E_local, tokens * K)`` experts stream
# from HBM instead of E_local -- up to ``E_local / G`` fewer expert-weight
# bytes/token, the dominant single-stream decode cost.
#
# Static shapes (HPU graph-compiler requirement): ``G`` is a function of the
# static (bucketed) token count, so every shape is fixed per warmup bucket and
# the decode-bucket graphs compile once during warmup (no per-step recompile).
# Because the number of distinct hit experts is provably ``<= tokens * K <= G``,
# every routed expert is always included -- no token is ever dropped and there
# is NO data-dependent overflow branch. Gathering ids sorted ascending makes the
# per-token expert accumulation order identical to the dense index-order pass
# (padding experts contribute exactly +0.0), so the result is numerically
# identical to ``_unfused_swigluoai_moe`` (bit-exact; validated).
#
# Enabled by default for MiniMax-M3 low-concurrency decode. Set
# VLLM_MINIMAX_M3_MOE_DECODE_GATHER=0 to use the dense expert pass instead.
_MOE_DECODE_GATHER = gaudi_envs.VLLM_MINIMAX_M3_MOE_DECODE_GATHER
# Only gather when the batch is at most this many tokens (bounds gather to the
# low-concurrency decode regime where it wins; above it the dense pass is used).
_MOE_GATHER_MAX_TOKENS = gaudi_envs.VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS


def _gather_swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Expert-gather SwiGLU-OAI MLP: compute only the routed experts.

    Numerically identical to :func:`_unfused_swigluoai_moe` (same per-rank,
    pre-all-reduce partial; same fp32 SwiGLU-OAI math and expert accumulation
    order) but reads only ``G = min(E_local, tokens * K)`` expert weights from
    HBM instead of all ``E_local``. See the module comment above for the
    static-shape / no-drop / bit-exactness argument.
    """
    w13 = layer.w13_weight  # [E_local, 2*I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    d = w13.shape[1] // 2
    hidden = x.shape[-1]
    tokens = x.shape[0]
    k = topk_ids.shape[-1]

    # Static gathered-expert count (>= number of distinct hit experts).
    g = min(e_local, tokens * k)

    # Per-(token, local-expert) combine weight, identical to the dense pass.
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Pick the G local experts to compute (sorted ascending so the per-token
    # accumulation order matches the dense index-order pass -> bit-exact).
    hit = (gate_w > 0).to(torch.float32).sum(0)  # [E_local]
    gather_ids = torch.topk(hit, g, sorted=False).indices  # [G]
    gather_ids, _ = torch.sort(gather_ids)  # ascending ids

    w13_g = w13.index_select(0, gather_ids)  # [G, 2I, H]
    w2_g = w2.index_select(0, gather_ids)  # [G, H, I]
    gate_wg = gate_w.index_select(1, gather_ids)  # [T, G]

    # Expert MLP over the gathered experts (static shapes, no host sync).
    xe = x.unsqueeze(0).expand(g, tokens, hidden)  # [G, T, H]
    h = torch.bmm(xe, w13_g.transpose(1, 2))  # [G, T, 2I]
    gg = h[..., :d].float()
    uu = h[..., d:].float()
    if limit is not None:
        gg = gg.clamp(max=limit)
        uu = uu.clamp(min=-limit, max=limit)
    act = (gg * torch.sigmoid(alpha * gg) * (uu + beta)).to(x.dtype)  # [G, T, I]
    y = torch.bmm(act, w2_g.transpose(1, 2))  # [G, T, H] partial

    # Match the dense pass's accumulation EXACTLY so the result is bit-identical:
    # _unfused_swigluoai_moe sums experts within each 32-expert index chunk in
    # fp32, rounds that partial to bf16, then accumulates across chunks in bf16.
    # Reproduce the same chunk grouping + rounding here (gathered experts are
    # grouped by their original index // chunk; experts absent from a chunk
    # contribute exact +0.0, which never perturbs an fp sum).
    cont = y.float() * gate_wg.t().unsqueeze(-1)  # [G, T, H] fp32
    chunk = 32
    chunk_of = torch.div(gather_ids, chunk, rounding_mode="floor")  # [G]
    acc = x.new_zeros(tokens, hidden)  # [T, H] bf16
    n_chunks = (e_local + chunk - 1) // chunk
    for c in range(n_chunks):
        m = (chunk_of == c).to(cont.dtype)  # [G]
        acc = acc + (cont * m[:, None, None]).sum(0).to(x.dtype)
    return acc


def _swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Dispatch SwiGLU-OAI experts: gather only routed experts at small token
    counts (low-concurrency decode), else the dense pass. Both return the
    identical per-rank partial; the split is purely a bandwidth optimization.
    """
    tokens = x.shape[0]
    k = topk_ids.shape[-1]
    e_local = layer.w13_weight.shape[0]
    if (_MOE_DECODE_GATHER and tokens <= _MOE_GATHER_MAX_TOKENS and tokens * k < e_local):
        return _gather_swigluoai_moe(layer, x, topk_ids, topk_weights, alpha=alpha, beta=beta, limit=limit)
    return _unfused_swigluoai_moe(layer, x, topk_ids, topk_weights, alpha=alpha, beta=beta, limit=limit)


# --- Safe fp8-e4m3fn decode ---------------------------------------------------
# The HPU torch cast float8_e4m3fn -> bf16 corrupts the top exponent bin
# (every |v| >= 256, u8 120-126/248-254, becomes inf/NaN on device; the CPU
# cast is correct). Real checkpoints quantized at amax/448 populate that bin,
# so any torch-cast dequant poisons the weights. Decode via a 256-entry LUT
# (integer gather) instead — exact for every representable e4m3fn value.
_E4M3FN_LUT: dict = {}


def _e4m3fn_lut(device) -> torch.Tensor:
    key = str(device)
    if key not in _E4M3FN_LUT:
        vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)  # 0x7F/0xFF -> 0
        _E4M3FN_LUT[key] = vals.to(device)
    return _E4M3FN_LUT[key]


@torch._dynamo.disable
def fp8_to_float_safe(w: torch.Tensor) -> torch.Tensor:
    """Exact fp8-e4m3fn -> fp32 via LUT (HPU cast bug workaround)."""
    if w.device.type != 'hpu':
        return w.float()
    lut = _e4m3fn_lut(w.device)
    shape = w.shape
    return lut[w.view(torch.uint8).reshape(-1).long()].reshape(shape)


def _dequant_expert_chunk(layer, sel, dtype=None):
    """Dequantize the selected expert weights (fp8) or pass through.

    Handles both HPU fp8 layouts the engine may leave on the layer:
      * block: weight [., N, K], scale [., nBlk, kBlk] (same rank)
      * per-channel (VLLM_HPU_FORCE_CHANNEL_FP8=True default): weight [., N, K]
        fp8 with scale [., N] (one row scale per output channel)
    ``sel`` is a slice or an index tensor; ``dtype`` normalizes the result to
    the activation dtype. Returns (w13, w2) as [c, 2I, H]/[c, H, I].
    """
    if isinstance(sel, list):
        # NB: python-list advanced indexing of an fp8 tensor returns corrupted
        # data under PT_HPU_LAZY_MODE=1 (verified: list-[5] vs slice(5,6)
        # differ by ~1.0); index_select is exact. Slices are unaffected.
        sel_idx = torch.tensor(sel, device=layer.w13_weight.device, dtype=torch.long)
        w13 = layer.w13_weight.index_select(0, sel_idx)
        w2 = layer.w2_weight.index_select(0, sel_idx)
    else:
        w13 = layer.w13_weight[sel]
        w2 = layer.w2_weight[sel]
    if w13.dtype == torch.float8_e4m3fn:
        dt = torch.bfloat16 if dtype is None else dtype
        w13_s = (layer.w13_weight_scale_inv.index_select(0, sel_idx).float()
                 if isinstance(sel, list) else layer.w13_weight_scale_inv[sel].float())
        w2_s = (layer.w2_weight_scale_inv.index_select(0, sel_idx).float()
                if isinstance(sel, list) else layer.w2_weight_scale_inv[sel].float())
        bs = layer.quant_config.weight_block_size

        def _dq(w, ws):
            wf = fp8_to_float_safe(w)  # [., N, K] fp32, exact (LUT decode)
            if ws.dim() == w.dim() and ws.shape[-1] != w.shape[-1]:
                # block scales [., N//bm, K//bk] vs weight [., N, K]
                n, k = w.shape[-2], w.shape[-1]
                bm, bk = bs[0], bs[1]
                wf = wf.reshape(*wf.shape[:-2], n // bm, bm, k // bk, bk)
                ws = ws.reshape(*ws.shape[:-2], n // bm, 1, k // bk, 1)
                return (wf * ws).reshape(*w.shape[:-2], n, k).to(dt)
            if ws.dim() == w.dim():
                # full elementwise scales
                return (wf * ws).to(dt)
            if ws.dim() == w.dim() - 1:
                # per-channel scales [., N] against [., N, K]
                return (wf * ws.unsqueeze(-1)).to(dt)
            raise ValueError(f"unsupported scale layout: w={tuple(w.shape)} s={tuple(ws.shape)}")

        w13 = _dq(w13, w13_s)
        w2 = _dq(w2, w2_s)
    elif dtype is not None and w13.dtype != dtype:
        w13 = w13.to(dtype)
        w2 = w2.to(dtype)
    return w13, w2


# --- Dequantized-expert cache (GLM clamped-SwiGLU decode fast path) ----------
# Decode routes to ~8 of 36 local experts per token per layer; without a cache
# the clamp path re-dequantizes ALL local experts every token (the dominant
# eager-decode cost, ~0.03 tok/s). Cache the dequantized bf16 weights per
# (layer, local expert) in an LRU, filled by the dense prefill pass and hit
# on decode. Memory: each expert holds w13 (2I x H) + w2 (H x I) bf16; the
# default cap of 10 experts/layer ~= 22 GB/rank for GLM-5.3-Flash shapes
# (43 MoE layers) — budget against free HBM; disable with 0.

_EXPERT_CACHE_CAP = int(os.environ.get("VLLM_GLM_MOE_CACHE_EXPERTS", "10"))
_EXPERT_CACHES: dict[int, "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]"] = {}
_EXPERT_PACKED: dict[int, tuple] = {}  # (w13, w2, eids) contiguous [cap, H, 2I]/[cap, I, H]
# Memory-pressure guard: the cache lives OUTSIDE the engine's memory budget
# (weights + KV sized by gpu_memory_utilization). v10 crashed with a fatal
# PT_DEVMEM alloc failure when the cache (~21.6 GB/rank at cap 10) exceeded
# the ~18 GB headroom. Before inserting/evicting/packing we therefore check
# torch.hpu.mem_get_info and self-shrink the cache so device free never
# drops below the floor (which must also cover the dense-path fp32 dequant
# transient, ~3 GB at chunk 16). Alloc failures are caught and degrade to
# dense-only rather than killing the worker.
_MIN_FREE_B = int(os.environ.get("VLLM_GLM_MOE_MIN_FREE_GB", "4")) * (1 << 30)
_CACHE_DEGRADED = False


def _device_free_b() -> float:
    try:
        return float(torch.hpu.mem_get_info()[0])
    except Exception:
        return float("inf")


def _relieve_pressure(layer, want_free_b: float, keep=frozenset()) -> None:
    """Evict LRU non-`keep` experts until free >= want (best effort)."""
    cache = _EXPERT_CACHES.get(id(layer))
    while cache is not None and len(cache) > len(keep & cache.keys()) and _device_free_b() < want_free_b:
        victim = next((e for e in cache if e not in keep), None)
        if victim is None:
            return
        cache.pop(victim)


def _relieve_pressure_global(want_free_b: float, protect_key=None, protect_ids=frozenset()) -> None:
    """Round-robin eviction across ALL layer caches until free >= want
    (dense-path transients are ~3 GB and may need the whole cache to yield).
    `protect_ids` on `protect_key`'s layer are never evicted (this call's
    routed set)."""
    while _device_free_b() < want_free_b:
        progressed = False
        for key, cache in list(_EXPERT_CACHES.items()):
            if not cache:
                continue
            if key == protect_key:
                victim = next((e for e in cache if e not in protect_ids), None)
                if victim is None:
                    continue
                cache.pop(victim)
            else:
                cache.popitem(last=False)
            progressed = True
            if _device_free_b() >= want_free_b:
                return
        if not progressed:
            return


def _expert_cache(layer):
    key = id(layer)
    if key not in _EXPERT_CACHES:
        _EXPERT_CACHES[key] = OrderedDict()
    return _EXPERT_CACHES[key]


def _packed_experts(layer, dtype, keep=frozenset()):
    """Contiguous bmm-ready weight buffer for the cached expert set (rebuilt
    lazily after any cache insertion/eviction; entries are stored already
    transposed [1, H, 2I]/[1, I, H] and contiguous). `keep` entries survive
    pressure relief; if one cannot, raises _CacheDegraded (caller -> dense)."""
    key = id(layer)
    packed = _EXPERT_PACKED.get(key)
    cache = _EXPERT_CACHES[key]
    if packed is None or packed[2] != list(cache.keys()) or packed[0].dtype != dtype:
        _relieve_pressure(layer, _MIN_FREE_B + (1 << 30), keep=keep)
        if any(e not in cache for e in keep):
            raise _CacheDegraded
        try:
            w13 = torch.cat([cache[e][0] for e in cache], 0) if cache else layer.w13_weight[:0].new_zeros(0)
            w2 = torch.cat([cache[e][1] for e in cache], 0) if cache else layer.w2_weight[:0].new_zeros(0)
        except RuntimeError as _alloc_err:  # alloc failure: drop the packed buffer, degrade
            _EXPERT_PACKED.pop(key, None)
            raise _CacheDegraded from _alloc_err
        packed = (w13, w2, list(cache.keys()))
        _EXPERT_PACKED[key] = packed
    return packed


class _CacheDegraded(Exception):
    """Raised internally when cache allocations fail under memory pressure."""


# --- Graph-safe expert cache (HPU-graph decode fast path) -------------------
# The eager LRU cache that preceded this redesign was host-driven: routing is resolved with
# .cpu().tolist() inside the forward and the packed bmm buffer is rebuilt by
# torch.cat after insertions. Under HPU-graph capture (wrap_in_hpu_graph)
# that breaks twice: (a) the host routing / python-set logic executes only at
# capture time, freezing the capture batch's routing into every replay, and
# (b) the packed tensors are (re)allocated inside one bucket's capture, so
# the next bucket replays with stale/empty buffers ("Empty tensor optional
# ... Strided Params Has Value: 0"). Graph-mode runs therefore had to disable
# the cache (VLLM_MINIMAX_M3_MOE_DECODE_GATHER=0 -> dense dequant-all, ~3.6x
# slower).
#
# This redesign keeps the numerics identical and splits the work by context:
#
# * CAPTURED REGION (what replays execute): _capture_gather_silu_clamp_moe
#   -- device-only routing (fp32 gate_w scatter -> topk -> sorted gather ids)
#   plus dequant-in-graph of exactly the gathered G = min(E_local, T*K)
#   experts. It reads ONLY immutable tensors (fp8 weights, scales, LUT)
#   besides the live forward inputs, because wrap_in_hpu_graph FREEZES the
#   values of every non-forward-argument tensor at capture (verified on
#   device: external index_copy_ writes to python-held tensors, registered
#   buffers, even in-graph mul_(1) identity writes are all invisible to
#   replays; only marked user inputs are refreshed). A replay-visible
#   persistent cache would need the buffers threaded through the forward
#   signature -- model files are out of scope here. Re-deriving routing on
#   device each replay is what makes stale-routing impossible, and gathering
#   only G experts keeps the ~E_local/G dequant-bandwidth win that the dense
#   workaround gave up.
#
# * PYTHON CONTEXTS (engaged steps running eagerly, e.g. shapes that overflow
#   max_graphs): persistent per-layer state allocated ONCE outside any
#   capture -- packed_w13 [CAP+1, H, 2I] / packed_w2 [CAP+1, I, H] bf16
#   (row CAP a permanent zero row) and expert_to_slot [E_local] int64 --
#   maintained by moe_graph_cache_maintain (host routing, LUT dequant,
#   index_copy_ into stable slots; LRU eviction) and read by
#   _static_gather_silu_clamp_moe with static-shape device ops. The engine
#   work: anything that runs between capture_begin/capture_end
#   because they would be frozen into the graph and re-executed on every
#   replay, clobbering later updates.
#
# Capacity reuses VLLM_GLM_MOE_CACHE_EXPERTS (allocation is memory-guarded
# against the same free-HBM floor as the eager cache). The cached static
# path engages only while tokens*K <= CAP, which makes "a routed expert has
# no slot" impossible by construction (maintenance caches the whole routed
# set). Misses that somehow still occur map to the zero row and contribute
# exactly +0.0, exactly like the existing out-of-local-range gate-weight
# masking.
#
# Engagement (VLLM_GLM_GRAPH_CACHE, default "auto"): "auto" activates the
# static path whenever HPU-graph intent is latched for this step (the
# runner's per-step use_graphs decision -- the same latch
# causal_conv1d_pytorch reads -- or this module's own set_moe_graphs_active),
# "1" forces it on (eager validation / tests), "0" forces it off. The pure
# eager default keeps the original host-routed LRU path bit-for-bit.
_GRAPH_MOE_STATES: dict[int, "_GraphExpertCache"] = {}
_GRAPH_MOE_LAYERS: dict[int, object] = {}  # id(layer) -> layer (runner hook)
_HPU_GRAPHS_CONFIGURED: bool | None = None  # snapshotted at weight-load time
_moe_graphs_active = False  # own latch (tests / serve wrappers)


def set_moe_graphs_active(active: bool) -> None:
    """Latch HPU-graph intent for the graph-safe MoE expert cache.

    Phase-6c latch pattern (cf. causal_conv1d_pytorch.
    set_conv_pool_hpu_graphs_active, set by the model runner per step). When
    no one calls this, ``_graphs_engaged`` falls back to reading the runner's
    conv latch, which is set from the same site.
    """
    global _moe_graphs_active
    _moe_graphs_active = bool(active)


def _graphs_engaged() -> bool:
    """Whether the graph-safe static expert path may run for this step."""
    mode = os.environ.get("VLLM_GLM_GRAPH_CACHE", "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return False
    if mode in ("1", "on", "true", "yes"):
        return True
    if _moe_graphs_active:
        return True
    try:  # runner-latched graph intent (the latch causal_conv1d reads)
        from vllm_gaudi.ops import causal_conv1d_pytorch as _conv
        return bool(getattr(_conv, "_hpu_graphs_active", False))
    except Exception:
        return False


def _in_hpu_graph_capture() -> bool:
    """True between wrap_in_hpu_graph's capture_begin/capture_end.

    Lazy mode (PT_HPU_LAZY_MODE=1, the only mode wrap_in_hpu_graph captures
    in) exposes NO runtime capture query: is_current_stream_capturing()
    reads a stream flag the lazy capture path never sets, and the capture
    stream context is not reflected in current_stream() (both verified on
    device). We therefore count capture_begin/capture_end ourselves via a
    small patch installed on habana_frameworks' HPUGraph class (installed
    once, lazily); the official queries are kept as secondary signals.
    Fails SAFE: anything unresolved counts as "capturing" (the capturable
    path is always correct; maintenance inside capture is never).
    """
    try:
        if not torch.hpu.is_available():
            return False
        if _HPU_CAPTURE_DEPTH > 0:
            return True
        if hasattr(torch.hpu, "is_current_stream_capturing"):
            return bool(torch.hpu.is_current_stream_capturing())
        return torch.hpu.current_stream() != torch.hpu.default_stream()
    except Exception:
        return False


_HPU_CAPTURE_DEPTH = 0
_CAPTURE_PATCH_LOCK = False


def _install_capture_detector() -> None:
    """Count HPUGraph capture_begin/capture_end calls (idempotent).

    wrap_in_hpu_graph brackets ``orig_fwd`` with HPUGraph().capture_begin()/
    capture_end() (habana_frameworks torch/hpu/graphs.py); counting those
    calls is the only reliable in-lazy-mode signal for "python is running
    inside a graph capture".
    """
    global _CAPTURE_PATCH_LOCK
    if _CAPTURE_PATCH_LOCK:
        return
    _CAPTURE_PATCH_LOCK = True
    try:
        import habana_frameworks.torch.hpu.graphs as _graphs
        cls = _graphs.HPUGraph
        if getattr(cls, "_vllm_gaudi_capture_patched", False):
            return
        orig_begin, orig_end = cls.capture_begin, cls.capture_end

        def _counted_begin(self, *a, **k):
            global _HPU_CAPTURE_DEPTH
            _HPU_CAPTURE_DEPTH += 1
            try:
                return orig_begin(self, *a, **k)
            except Exception:
                _HPU_CAPTURE_DEPTH -= 1
                raise

        def _counted_end(self, *a, **k):
            global _HPU_CAPTURE_DEPTH
            _HPU_CAPTURE_DEPTH -= 1
            return orig_end(self, *a, **k)

        cls.capture_begin = _counted_begin
        cls.capture_end = _counted_end
        cls._vllm_gaudi_capture_patched = True
    except Exception:
        pass


# Install at import time so the FIRST capture (which may happen before any
# forward-side call of _in_hpu_graph_capture) is already counted. Idempotent;
# silently skipped when habana_frameworks is unavailable (CPU-only tests).
_install_capture_detector()


def _hpu_graphs_configured() -> bool:
    """Whether the engine run has HPU graphs enabled at all (sticky).

    Snapshotted once while the vLLM config context is active (weight-load
    time); gates eager-side cache seeding so pure-eager runs never pay the
    extra host-routing sync.
    """
    global _HPU_GRAPHS_CONFIGURED
    if _HPU_GRAPHS_CONFIGURED is None:
        try:
            cfg = get_current_vllm_config_or_none()
            _HPU_GRAPHS_CONFIGURED = bool(cfg is not None and not cfg.model_config.enforce_eager)
        except Exception:
            _HPU_GRAPHS_CONFIGURED = False
    return _HPU_GRAPHS_CONFIGURED


class _GraphExpertCache:
    """Per-layer persistent graph-safe dequantized-expert cache."""

    __slots__ = ("packed_w13", "packed_w2", "expert_to_slot", "zero_row", "cap", "dtype", "e_local", "slot_of",
                 "expert_of_slot", "lru", "free_slots", "layer_ref", "last_topk_ids", "ready")

    def __init__(self, layer, dtype, cap):
        dev = layer.w13_weight.device
        e_local, two_i, h = layer.w13_weight.shape
        i = two_i // 2
        self.e_local = e_local
        self.cap = cap
        self.dtype = dtype
        self.zero_row = cap  # permanent all-zero row (misses / masking)
        # Allocated ONCE, OUTSIDE any capture; mutated in-place only
        # (index_copy_/index_fill_), so their storage addresses -- what a
        # captured graph binds -- never change.
        self.packed_w13 = torch.zeros(cap + 1, h, two_i, device=dev, dtype=dtype)
        self.packed_w2 = torch.zeros(cap + 1, i, h, device=dev, dtype=dtype)
        self.expert_to_slot = torch.full((e_local, ), -1, device=dev, dtype=torch.long)
        self.slot_of: dict[int, int] = {}  # host mirror of expert_to_slot
        self.expert_of_slot: list = [None] * cap
        self.lru: OrderedDict[int, None] = OrderedDict()  # expert -> None
        self.free_slots: list[int] = list(range(cap))
        self.layer_ref = layer
        self.last_topk_ids = None  # stashed routing tensor (graph-internal)
        self.ready = False  # True once maintenance has filled it once


def _graph_expert_cache(layer, dtype, create=False):
    """Get (optionally create) the layer's graph-safe cache state.

    Creation is memory-guarded the same way as the eager cache: never let
    device free drop below the floor, shrinking the capacity (or refusing to
    create) instead of failing the worker. The layer's eager LRU entries are
    dropped at creation time -- once the graph path engages they are dead
    weight and their memory funds the persistent packed buffers.
    """
    key = id(layer)
    state = _GRAPH_MOE_STATES.get(key)
    if state is not None and state.dtype != dtype:
        # dtype changed (e.g. test contexts): rebuild outside capture
        _GRAPH_MOE_STATES.pop(key, None)
        state = None
    if state is None and create and _EXPERT_CACHE_CAP > 0:
        two_i, h = layer.w13_weight.shape[1], layer.w13_weight.shape[2]
        i = two_i // 2
        row_b = (h * two_i + i * h) * torch.empty((), dtype=dtype).element_size()
        cap = _EXPERT_CACHE_CAP
        free = _device_free_b()
        if free - (cap + 1) * row_b < _MIN_FREE_B:
            # try to fund the buffers from the (now unused) eager caches
            _EXPERT_CACHES.pop(key, None)
            _EXPERT_PACKED.pop(key, None)
            _relieve_pressure_global(_MIN_FREE_B + (cap + 1) * row_b)
            free = _device_free_b()
            cap = min(cap, max(int((free - _MIN_FREE_B) // row_b) - 1, 0))
        else:
            _EXPERT_CACHES.pop(key, None)
            _EXPERT_PACKED.pop(key, None)
        if cap > 0:
            try:
                state = _GraphExpertCache(layer, dtype, cap)
                _GRAPH_MOE_STATES[key] = state
            except RuntimeError:  # alloc failure under pressure: no cache
                state = None
    return state


def moe_graph_cache_maintain(layer, topk_ids, dtype):
    """Graph-EXTERNAL cache maintenance. NEVER call while capturing.

    Resolves this batch's routed local experts from a host copy of
    ``topk_ids`` (obtained outside the graph), dequantizes misses with the
    exact LUT path (one batched call, like the eager path), and writes them
    into the persistent packed rows at stable slots with index_copy_
    (in-place, storage-stable) plus the matching expert_to_slot updates
    (index_copy_/index_fill_). Eviction reuses the least-recently-used slot
    when the cache is full; slot->expert bookkeeping is host-side python,
    device state is only ever mutated here, outside capture. After a
    successful call every routed expert of this batch has a slot, so the
    cached static path (used in python contexts) can never miss on this
    routing. (HPU-graph REPLAYS do not read this cache -- wrap_in_hpu_graph
    freezes non-input tensor values at capture -- they recompute routing
    and dequant on device via _capture_gather_silu_clamp_moe.)
    """
    state = _graph_expert_cache(layer, dtype, create=True)
    if state is None or _in_hpu_graph_capture():
        # defensive: never mutate/allocate cache state between
        # capture_begin/capture_end (ops would be frozen into the graph)
        return state if state is not None else None
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local = state.e_local
    state.last_topk_ids = topk_ids
    ids_h = topk_ids.detach().cpu().tolist()
    routed = set()
    for row in ids_h:
        for eid in row:
            lid = int(eid) - experts_min
            if 0 <= lid < e_local:
                routed.add(lid)
    if not routed:
        return state
    for e in routed:  # LRU-touch this batch's hits so eviction can only
        if e in state.lru:  # drop cold experts
            state.lru.move_to_end(e)
    misses = sorted(e for e in routed if e not in state.slot_of)
    if not misses:
        state.ready = True
        return state
    slot_pick: list[int] = []
    evicted: list[int] = []
    for _ in misses:
        if state.free_slots:
            slot_pick.append(state.free_slots.pop())
            continue
        victim = next((e for e in state.lru if e not in routed), None)
        if victim is None:  # cap < routed set: caller must not take the
            break  # static path for this batch (checked via state.cap)
        state.lru.pop(victim)
        slot = state.slot_of.pop(victim)
        state.expert_of_slot[slot] = None
        evicted.append(victim)
        slot_pick.append(slot)
    misses = misses[:len(slot_pick)]
    if misses:
        _relieve_pressure(layer, _MIN_FREE_B)  # room for the fp32 transient
        try:
            mw13, mw2 = _dequant_expert_chunk(layer, misses, dtype=dtype)
        except RuntimeError:
            _relieve_pressure_global(_MIN_FREE_B, protect_key=id(layer), protect_ids=frozenset(routed))
            mw13, mw2 = _dequant_expert_chunk(layer, misses, dtype=dtype)
        dev = state.expert_to_slot.device
        slots_t = torch.tensor(slot_pick, device=dev, dtype=torch.long)
        state.packed_w13.index_copy_(0, slots_t, mw13.transpose(1, 2))
        state.packed_w2.index_copy_(0, slots_t, mw2.transpose(1, 2))
        miss_t = torch.tensor(misses, device=dev, dtype=torch.long)
        state.expert_to_slot.index_copy_(0, miss_t, slots_t)
        if evicted:
            state.expert_to_slot.index_fill_(0, torch.tensor(evicted, device=dev, dtype=torch.long), -1)
        for e, s in zip(misses, slot_pick):
            state.slot_of[e] = s
            state.expert_of_slot[s] = e
            state.lru[e] = None
    # ready ONLY when this batch's whole routed set is cached -- that is the
    # invariant the static path needs (zero-row misses would silently drop
    # expert contributions otherwise).
    if all(e in state.slot_of for e in routed):
        state.ready = True
    return state


def _seed_graph_cache(layer, topk_ids, dtype) -> None:
    """Warm the graph cache during EAGER dense steps (prefill routing).

    Only runs when the engine has HPU graphs configured (pure-eager runs are
    untouched), never inside a capture (allocating persistent buffers there
    is the original crash pattern), and stops once the cache is full --
    steady-state maintenance belongs to the runner hook. Seeding during
    prefill means the first decode capture already sees a hot-set cache
    instead of a cold one.
    """
    if not _hpu_graphs_configured() or _in_hpu_graph_capture():
        return
    state = _graph_expert_cache(layer, dtype, create=True)
    if state is None or len(state.lru) >= state.cap:
        return
    with contextlib.suppress(Exception):
        moe_graph_cache_maintain(layer, topk_ids, dtype)


def _dequant_expert_chunk_t(layer, sel_idx, dtype=None):
    """Tensor-indexed, fully device-side variant of _dequant_expert_chunk.

    Produces bit-identical rows ([c, H, 2I]/[c, I, H], contiguous) but takes a
    DEVICE index tensor, so it is capturable in HPU graphs: every op is
    static-shape and the only non-input tensors it reads (fp8 weights,
    scales, LUT) are immutable constants, which graphs may safely freeze.
    """
    w13 = layer.w13_weight.index_select(0, sel_idx)  # [c, 2I, H]
    w2 = layer.w2_weight.index_select(0, sel_idx)  # [c, H, I]
    if w13.dtype == torch.float8_e4m3fn:
        dt = torch.bfloat16 if dtype is None else dtype
        w13_s = layer.w13_weight_scale_inv.index_select(0, sel_idx).float()
        w2_s = layer.w2_weight_scale_inv.index_select(0, sel_idx).float()
        bs = layer.quant_config.weight_block_size

        def _dq(w, ws):
            wf = fp8_to_float_safe(w)  # [., N, K] fp32, exact (LUT decode)
            if ws.dim() == w.dim() and ws.shape[-1] != w.shape[-1]:
                n, k = w.shape[-2], w.shape[-1]
                bm, bk = bs[0], bs[1]
                wf = wf.reshape(*wf.shape[:-2], n // bm, bm, k // bk, bk)
                ws = ws.reshape(*ws.shape[:-2], n // bm, 1, k // bk, 1)
                return (wf * ws).reshape(*w.shape[:-2], n, k).to(dt)
            if ws.dim() == w.dim():
                return (wf * ws).to(dt)
            if ws.dim() == w.dim() - 1:
                return (wf * ws.unsqueeze(-1)).to(dt)
            raise ValueError(f"unsupported scale layout: w={tuple(w.shape)} s={tuple(ws.shape)}")

        w13 = _dq(w13, w13_s)
        w2 = _dq(w2, w2_s)
    elif dtype is not None and w13.dtype != dtype:
        w13 = w13.to(dtype)
        w2 = w2.to(dtype)
    return w13.transpose(1, 2).contiguous(), w2.transpose(1, 2).contiguous()


def _capture_gather_silu_clamp_moe(layer, x, topk_ids, topk_weights, *, limit):
    """Capturable routed-expert compute: gather + dequant-in-graph.

    Why not the persistent packed cache here: wrap_in_hpu_graph freezes the
    VALUES of every non-forward-argument tensor at capture (verified on
    device: external index_copy_ writes to python-held tensors, registered
    buffers, and even in-graph mul_(1) identity writes are all invisible to
    replays; only marked user inputs are copied per replay). A cache read
    inside the graph would therefore forever serve the capture-time rows.
    The fp8 weights/scales/LUT, by contrast, are immutable -- freezing them
    is exactly correct -- so this path gathers the routed G experts with
    device ops and dequantizes them in-graph every replay. That keeps the
    ~E_local/G dequant-bandwidth win over the dense fallback (the 3.6x
    regression) while being correct for ARBITRARY replayed routing.

    Numerically identical to the eager gather path: same gate weights (fp32
    scatter), same ascending expert order, same dequant math, same chunk-32
    bf16 accumulation.
    """
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local = layer.w13_weight.shape[0]
    hidden = x.shape[-1]
    tokens = x.shape[0]
    k = topk_ids.shape[-1]

    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    g = min(e_local, tokens * k)
    hit = (gate_w > 0).to(torch.float32).sum(0)  # [E_local]
    gather_ids = torch.sort(torch.topk(hit, g, sorted=False).indices).values  # [G]
    gw_g = gate_w.index_select(1, gather_ids).t().unsqueeze(-1)  # [G, T, 1]

    acc = torch.zeros_like(x)
    chunk = 32  # same chunking as the eager gather branch
    for cs in range(0, g, chunk):
        ce = min(cs + chunk, g)
        c = ce - cs
        w13c, w2c = _dequant_expert_chunk_t(layer, gather_ids[cs:ce], dtype=x.dtype)
        xe = x.unsqueeze(0).expand(c, tokens, hidden)
        h = torch.bmm(xe, w13c)
        act = clamp_swiglu(h, limit)
        y = torch.bmm(act, w2c)
        wc = gw_g[cs:ce]
        acc = acc + (y.float() * wc).sum(0).to(x.dtype)
    return acc


def _static_gather_silu_clamp_moe(layer, x, topk_ids, topk_weights, state, *, limit):
    """Graph-safe routed-expert compute (capturable; numerically identical
    to the eager gather path).

    Everything here is static-shape device ops: NO host reads (.cpu()/
    .item()/.tolist()), NO python-set routing logic, NO allocations of
    persistent state. The expert->slot mapping is resolved at REPLAY time
    from the live expert_to_slot table, and weight rows come from the
    persistent packed buffers via index_select -- so cache updates performed
    outside the graph between replays are picked up without re-capture.
    Uncached / out-of-range experts land on the permanent zero row and
    contribute exactly +0.0, matching the eager path's gate-weight masking.
    Accumulation mirrors the eager gather branch exactly (ascending expert
    order, chunk-32 bf16 partials, fp32 weighted per-chunk sums).
    """
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local = state.e_local
    hidden = x.shape[-1]
    tokens = x.shape[0]
    k = topk_ids.shape[-1]
    state.last_topk_ids = topk_ids  # pure python ref; hooks read it later

    # Per-(token, local-expert) combine weight, identical to the eager paths.
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Static gathered-expert count G = min(E_local, T*K) >= distinct hit
    # experts: every routed expert is always included, padding rows carry
    # gate weight 0 and contribute exact +0.0 (same argument as the eager
    # gather path). All values recomputed at replay from live routing.
    g = min(e_local, tokens * k)
    hit = (gate_w > 0).to(torch.float32).sum(0)  # [E_local]
    gather_ids = torch.sort(torch.topk(hit, g, sorted=False).indices).values  # [G]

    # Expert -> slot via the DEVICE table (dynamic at replay); misses and
    # out-of-range ids land on the permanent zero row.
    slots = state.expert_to_slot.index_select(0, gather_ids)  # [G]
    safe_slots = torch.where(slots >= 0, slots, torch.full_like(slots, state.zero_row))
    w13 = state.packed_w13.index_select(0, safe_slots)  # [G, H, 2I]
    w2 = state.packed_w2.index_select(0, safe_slots)  # [G, I, H]
    gw_g = gate_w.index_select(1, gather_ids).t().unsqueeze(-1)  # [G, T, 1]

    acc = torch.zeros_like(x)
    chunk = 32  # same chunking as the eager gather branch
    for cs in range(0, g, chunk):
        ce = min(cs + chunk, g)
        c = ce - cs
        xe = x.unsqueeze(0).expand(c, tokens, hidden)
        h = torch.bmm(xe, w13[cs:ce])
        act = clamp_swiglu(h, limit)
        y = torch.bmm(act, w2[cs:ce])
        wc = gw_g[cs:ce]
        acc = acc + (y.float() * wc).sum(0).to(x.dtype)
    return acc


def _silu_clamp_expert_act(h: torch.Tensor, d: int, limit: float) -> torch.Tensor:
    """GLM clamped SwiGLU (transformers Glm5Next semantics): clamp gate BEFORE
    silu, clamp up symmetrically; computed in fp32."""
    g = h[..., :d].float().clamp(max=limit)
    u = h[..., d:].float().clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(g) * u).to(h.dtype)


@torch._dynamo.disable
def _silu_clamp_moe(layer, x, topk_ids, topk_weights, *, limit: float) -> torch.Tensor:
    """Dense expert MLP with GLM clamped-SwiGLU for silu+swiglu_limit models.

    ``@torch._dynamo.disable``: this path mixes dynamic-ish index ops with
    chunked bmm; under compiled decode regions it trips Synapse
    Sections-validation failures. Running it eager is correct and the outer
    compiled region simply splits around it.

    The Habana fused MoE op only supports plain silu/gelu and silently drops
    ``swiglu_limit``; without the clamp the expert outputs are unbounded and
    generation degenerates (observed as verbatim repetition). This path mirrors
    ``_unfused_swigluoai_moe`` (dense, expert-chunked bmm, token-tiled) but
    dequantizes block-fp8 expert chunks on the fly — persistent bf16 copies
    would not fit (43 MoE layers x 1.8 GB/rank). Below the gather token
    threshold only the routed experts are dequantized (decode bandwidth).
    """
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local = layer.w13_weight.shape[0]
    hidden = x.shape[-1]
    tokens = x.shape[0]
    k = topk_ids.shape[-1]

    local_ids = topk_ids - experts_min
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    chunk = 32
    # Decode-gather regime: only the routed experts, served from the LRU
    # dequant cache packed into one contiguous buffer. Eager context only
    # (dynamo.disable): routing is resolved host-side to keep the HPU op
    # count ~8/token. NB: torch.nonzero on device is dynamic-shape and trips
    # Synapse sections validation — host-side set() is used instead.
    if _MOE_DECODE_GATHER and tokens <= _MOE_GATHER_MAX_TOKENS and tokens * k < e_local:
        if _graphs_engaged():
            # HPU-graph intent latched for this step: use a graph-safe routed path.
            # Between capture_begin/capture_end only the capturable gather+dequant
            # path may run (zero host reads; the persistent cache is NOT read
            # there because wrap_in_hpu_graph freezes non-input tensor values at
            # capture -- replays would forever serve capture-time rows). In eager
            # python contexts with graphs latched, maintenance runs first (host
            # routing + LUT dequant + index_copy_ into stable slots, all outside
            # capture) and the cached static path -- bit-identical to both the
            # captured path and the eager gather path -- serves the step.
            state = _GRAPH_MOE_STATES.get(id(layer))
            if state is not None and state.dtype == x.dtype:
                state.last_topk_ids = topk_ids  # python ref; hook maintains from it
            if _in_hpu_graph_capture():
                return _capture_gather_silu_clamp_moe(layer, x, topk_ids, topk_weights, limit=limit)
            try:
                state = moe_graph_cache_maintain(layer, topk_ids, x.dtype)
            except Exception:
                state = None
            # tokens*k <= cap guarantees maintenance can cache the whole routed
            # set, making the cached static path exactly the eager gather path.
            if (state is not None and state.dtype == x.dtype and state.ready and state.cap >= tokens * k):
                return _static_gather_silu_clamp_moe(layer, x, topk_ids, topk_weights, state, limit=limit)
            # cache unusable (cold / capacity / pressure): fall through to the
            # original eager branch below -- this is a python context, so its
            # host routing is safe and bit-for-bit the default behavior.
        try:
            # lazy-mode boundary: the host-routing below (D2H + python set ops)
            # must not be fused into the surrounding lazy graph; mark_step
            # flushes cleanly (no-op in eager mode).
            import habana_frameworks.torch.core as _htcore
            _htcore.mark_step()
            ids_h = topk_ids.detach().cpu().tolist()
            wts_h = topk_weights.detach().float().cpu().tolist()
            gw_rows: list[list[float]] = []  # per-token weights over gather_list
            for row_ids, row_w in zip(ids_h, wts_h):
                row: dict[int, float] = {}
                for eid, wv in zip(row_ids, row_w):
                    lid = eid - experts_min
                    if 0 <= lid < e_local:
                        row[lid] = row.get(lid, 0.0) + float(wv)
                gw_rows.append(row)
            routed = set()
            for row in gw_rows:
                routed.update(row.keys())
            if not routed:
                return x.new_zeros(tokens, hidden)
            gather_list = sorted(routed)
            cache = _expert_cache(layer)
            for e in gather_list:  # LRU-touch this token's hits so eviction
                if e in cache:  # can only remove cold experts
                    cache.move_to_end(e)
            misses = [e for e in gather_list if e not in cache or cache[e][0].dtype != x.dtype]
            if misses:  # one batched dequant for all misses (per-miss op cost /8)
                _relieve_pressure(layer, _MIN_FREE_B)  # room for the fp32 transient
                try:
                    mw13, mw2 = _dequant_expert_chunk(layer, misses, dtype=x.dtype)
                except RuntimeError:
                    _relieve_pressure_global(_MIN_FREE_B, protect_key=id(layer), protect_ids=frozenset(gather_list))
                    mw13, mw2 = _dequant_expert_chunk(layer, misses, dtype=x.dtype)
                for i, e in enumerate(misses):
                    cache[e] = (mw13[i:i + 1].transpose(1, 2).contiguous(), mw2[i:i + 1].transpose(1, 2).contiguous())
                while (_EXPERT_CACHE_CAP if not _CACHE_DEGRADED else 0) > 0 and \
                        len(cache) > (_EXPERT_CACHE_CAP if not _CACHE_DEGRADED else 0):
                    cache.popitem(last=False)
            if any(e not in cache for e in gather_list):
                # cap smaller than the routed set: cannot serve from cache this
                # call — fall through to the dense path for correctness.
                pass
            else:
                pw13, pw2, cached_eids = _packed_experts(layer, dtype=x.dtype, keep=frozenset(gather_list))
                slot = torch.tensor([cached_eids.index(e) for e in gather_list], device=x.device, dtype=torch.long)
                w13 = pw13.index_select(0, slot)  # [g, H, 2I]
                w2 = pw2.index_select(0, slot)  # [g, I, H]
                g_count = len(gather_list)
                gw_g = torch.tensor([[row.get(e, 0.0) for e in gather_list] for row in gw_rows],
                                    device=x.device,
                                    dtype=torch.float32).t().unsqueeze(-1)  # [g, T, 1]
                acc = torch.zeros_like(x)
                for cs in range(0, g_count, chunk):
                    ce = min(cs + chunk, g_count)
                    c = ce - cs
                    xe = x.unsqueeze(0).expand(c, tokens, hidden)
                    h = torch.bmm(xe, w13[cs:ce])
                    act = clamp_swiglu(h, limit)
                    y = torch.bmm(act, w2[cs:ce])
                    wc = gw_g[cs:ce]
                    acc = acc + (y.float() * wc).sum(0).to(x.dtype)
                return acc
        except (_CacheDegraded, ValueError, KeyError):
            # cache/packing inconsistency under pressure: dense path below
            pass
            # (fall through: _CacheDegraded or cap < routed set -> dense below)

    # Dense prefill path: expert-chunked bmm, token-tiled, dequant per chunk.
    # chunk 16 halves the fp32 dequant transient (~3 GB) vs 32, so the dense
    # fallback stays servable under cache pressure.
    chunk = 16
    _seed_graph_cache(layer, topk_ids, x.dtype)  # warm graph cache (prefill
    # routing) when HPU graphs are configured; no-op in pure-eager runs
    _relieve_pressure_global(_MIN_FREE_B + (3 << 30), protect_key=id(layer))
    token_tile = _SWIGLU_OAI_TOKEN_TILE
    if token_tile <= 0 or token_tile >= tokens:
        token_tile = tokens
    # Expert chunks OUTER, token tiles INNER. The dequantized chunk does not
    # depend on the tile, so the original (tile-outer) nesting re-dequantized
    # every expert once per tile: at 8192 tokens / tile 512 that is 16 dequants
    # of the same 16 experts, ~13 GB of bf16 rows plus their fp32 transients,
    # all accumulated into ONE lazy recipe (nothing flushes until the MoE
    # all-reduce). Hoisting it makes that 1 dequant, and the per-tile
    # Each tile still sums expert chunks in increasing `start` order, so the
    # accumulation order -- and the result -- is bit-identical to before.
    #
    # Two things that look like obvious follow-ups here are NOT safe, both
    # measured against the repro at 8192 tokens:
    #   * an htcore.mark_step() inside the nest, to bound the recipe further --
    #     _silu_clamp_moe runs inside the wrap_in_hpu_graph-wrapped model
    #     forward, and flushing there dies with "ValidateSyncInputTensors
    #     tensor_data is empty", even on the bypass_hpu_graphs branch;
    #   * `del w13c, w2c` after the tile loop, to release the chunk early --
    #     same failure. The recipe is still deferred at that point, so dropping
    #     the last python reference takes the storage with it.
    # Hoisting the dequant is what shrinks the recipe; neither of those is
    # needed on top of it.
    tiles = [(ts, min(ts + token_tile, tokens)) for ts in range(0, tokens, token_tile)]
    accs = [torch.zeros_like(x[ts:te]) for ts, te in tiles]
    for start in range(0, e_local, chunk):
        end = min(start + chunk, e_local)
        c = end - start
        w13c, w2c = _dequant_expert_chunk(layer, slice(start, end), dtype=x.dtype)
        w13c = w13c.transpose(1, 2)  # [c, H, 2I]
        w2c = w2c.transpose(1, 2)  # [c, I, H]
        for i, (tstart, tend) in enumerate(tiles):
            xt = x[tstart:tend]
            t = xt.shape[0]
            xe = xt.unsqueeze(0).expand(c, t, hidden)
            h = torch.bmm(xe, w13c)
            act = clamp_swiglu(h, limit)
            y = torch.bmm(act, w2c)
            gate_wc = gate_w[tstart:tend, start:end].t().unsqueeze(-1)
            accs[i] = accs[i] + (y.float() * gate_wc).sum(0).to(x.dtype)
    return accs[0] if len(accs) == 1 else torch.cat(accs, dim=0)


def model_has_quant_config() -> bool:
    """Whether the active model runs with a MoE quantization config.

    After upstream PR #41184 the ``layer`` reaching ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer carries ``vllm_config`` (that
    field belonged to the old top-level ``FusedMoE``). The model config must
    therefore be resolved from the global vLLM config instead of off ``layer``.

    This MUST be called at build time (e.g. in ``__init__`` /
    ``process_weights_after_loading``), where the vLLM config context is set:
    the result is static per run, so callers cache it and read the cached flag
    on the forward hot path. ``get_current_vllm_config_or_none`` is used so the
    function degrades to ``False`` instead of raising if no context is active.

    Returns:
        ``True`` when the active model config declares a MoE quant config.
    """
    vllm_config = get_current_vllm_config_or_none()
    model_config = vllm_config.model_config if vllm_config is not None else None
    return model_config is not None and has_quant_config(model_config)


def select_experts_from_routed(layer, hidden_states: torch.Tensor,
                               router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to experts for the grouped/custom-routing monolithic path.

    After upstream PR #41184 the ``layer`` passed to ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer owns a ``.router`` object (the
    router moved onto ``MoERunner``). ``RoutedExperts`` does, however, carry all
    the routing parameters, so we reproduce upstream's behaviour via the
    standalone ``select_experts`` helper. It is imported lazily because
    ``experts.cpu_moe`` pulls CPU custom ops from ``vllm._custom_ops`` at
    module import time.

    Args:
        layer: The ``RoutedExperts`` instance holding the routing parameters.
        hidden_states: Flattened input activations.
        router_logits: Gate logits for the current tokens.

    Returns:
        A ``(topk_weights, topk_ids)`` tuple.
    """
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import select_experts

    # Models hand the runner a placeholder ``router_logits`` (== hidden_states)
    # and expect the runner to overwrite it with the gate output. If that step
    # is skipped the placeholder reaches expert selection and only fails later,
    # deep inside the routing kernels, as an opaque broadcast error.
    num_experts = layer.moe_config.num_logical_experts
    if router_logits.size(-1) != num_experts:
        raise RuntimeError(f"Router logits have {router_logits.size(-1)} columns but the layer routes to "
                           f"{num_experts} experts; the MoE gate was not applied to the hidden states.")

    return select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=layer.top_k,
        use_grouped_topk=layer.use_grouped_topk,
        renormalize=layer.renormalize,
        topk_group=layer.topk_group,
        num_expert_group=layer.num_expert_group,
        custom_routing_function=layer.custom_routing_function,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=layer.routed_scaling_factor,
        e_score_correction_bias=layer.e_score_correction_bias,
    )


@UnquantizedFusedMoEMethod.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if (vllm_config is not None and vllm_config.model_config is not None
                and vllm_config.model_config.hf_config is not None):
            self.model_type = vllm_config.model_config.hf_config.model_type

        # SwiGLU-OAI activation params (MiniMax-M3). Cached here (config context
        # active) and read on the forward hot path when the fused kernel cannot
        # provide the activation. Falls back to the MiniMax-M3 defaults.
        tc = None
        if vllm_config is not None and vllm_config.model_config is not None:
            tc = getattr(vllm_config.model_config, "hf_text_config", None)
            if tc is None:
                tc = vllm_config.model_config.hf_config
        self.swiglu_alpha = float(getattr(tc, "swiglu_alpha", 1.702))
        self.swiglu_beta = float(getattr(tc, "swiglu_beta", 1.0))
        _limit = getattr(tc, "swiglu_limit", 7.0)
        self.swiglu_limit = None if _limit is None else float(_limit)

    def _select_monolithic(self) -> Callable:
        """Overriding base method"""
        return self.apply_monolithic

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts
        has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        bias = has_bias if has_bias is True else None

        is_bf16 = getattr(layer, "w13_weight", None) is not None and layer.w13_weight.dtype == torch.bfloat16

        is_unquantized = not self.has_moe_quant_config

        cache_weight_lists = bool(is_bf16 and is_unquantized)

        # Pass cache flag into moe_op (requires ops.py __init__ signature update)
        layer.moe_op = VllmMixtureOfExpertsOp(layer.global_num_experts, num_experts, experts_min, experts_max, bias,
                                              dispatch_fn)

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

        # Build cache once AFTER weights/bias are set (BF16 + unquantized only)
        if cache_weight_lists and hasattr(layer.moe_op, "_cache_weight_lists"):
            layer.moe_op._cache_weight_lists()

        # Graph-safe expert cache: register clamp-path MoE layers for the
        # lazy maintenance path and snapshot the
        # run's HPU-graph configuration while the vLLM config context is
        # active here. Pure-eager runs stay unregistered (no seeding cost).
        global _HPU_GRAPHS_CONFIGURED
        if (self.swiglu_limit is not None and getattr(layer, "activation", "silu") == "silu"
                and not gaudi_envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT):
            try:
                cfg = get_current_vllm_config_or_none()
                if cfg is not None and _HPU_GRAPHS_CONFIGURED is None:
                    _HPU_GRAPHS_CONFIGURED = not cfg.model_config.enforce_eager
            except Exception:
                pass
            if _hpu_graphs_configured():
                _GRAPH_MOE_LAYERS[id(layer)] = layer

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type == "gpt_oss":
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel compiles for bf16 (x.dtype) router
        # weights and int64 routing tables. The grouped-topk / custom-routing
        # helper returns float32 weights and int32 ids; the regular-topk path
        # above already normalized them, but the grouped path previously left
        # them unconverted -> the bf16 MoE kernel graph received a float32
        # router_weights tensor and failed to compile (synStatus 26). Normalize
        # for every routing path so the kernel inputs are dtype-consistent.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # Sequence-parallel MoE without data parallelism (TP>1 + EP with the
            # allgather_reducescatter backend => use_sequence_parallel_moe). The
            # model block chunks tokens by tp_size before the experts and
            # all-gathers afterwards, expecting `self.experts` to be
            # token-neutral. Upstream MoERunner._maybe_combine (mirrored in
            # patched_fused_moe_forward) reduce-scatters the expert output over
            # the EP group whenever is_sequence_parallel is set, regardless of
            # dp_size. The paired dispatch all-gather is set up via dispatch_fn
            # only for dp_size > 1, so at dp_size == 1 the combine is unpaired
            # and halves the token count -> the block's final reshape fails.
            # All-gather x / topk over the EP group here to restore the
            # dispatch/combine symmetry (dp_metadata is None at dp_size == 1, so
            # dispatch_tensor allocates its own EP-sized output).
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused
            # (dense), or gather-only routed experts at low-conc decode.
            output = _swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        elif (activation == "silu" and self.swiglu_limit is not None
              and not gaudi_envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT):
            # silu + swiglu_limit (GLM-5.x): the Habana fused op drops the
            # clamp; run the clamped-SwiGLU expert path instead.
            output = _silu_clamp_moe(layer, x, topk_ids, topk_weights, limit=float(self.swiglu_limit))
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)

    def forward_oot(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type is not None and self.model_type in ["gpt_oss"]:
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # See apply_monolithic: the bf16 HPU MoE kernel needs int64 routing
        # tables and x.dtype router weights. Normalize for every routing path
        # (grouped-topk / custom routing returns int32 + float32) so the kernel
        # graph receives dtype-consistent inputs and compiles.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # See apply_monolithic: at dp_size == 1 with sequence-parallel MoE
            # (TP>1 + EP), MoERunner._maybe_combine reduce-scatters the expert
            # output over the EP group but no paired dispatch all-gather runs
            # (dispatch_fn is wired only for dp_size > 1). Restore symmetry by
            # all-gathering the inputs over the EP group so the combine leaves
            # the token count unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        if self.model_type in ["gpt_oss"]:
            gpt_oss_out = layer.moe_op(
                x,
                topk_ids.to(torch.int64),
                topk_weights.to(x.dtype),
                permuted_weights=True,
                activation=_normalize_moe_activation(layer.activation),
            )
            if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
                return gpt_oss_out.view(*(gpt_oss_out.size(0), *input_shape[1:]))
            return gpt_oss_out.view(*input_shape)

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused
            # (dense), or gather-only routed experts at low-conc decode.
            output = _swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        elif (activation == "silu" and self.swiglu_limit is not None
              and not gaudi_envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT):
            # silu + swiglu_limit (GLM-5.x): clamped-SwiGLU expert path.
            output = _silu_clamp_moe(layer, x, topk_ids, topk_weights, limit=float(self.swiglu_limit))
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


def _launch_shared_experts_overlap(runner: MoERunnerBase, shared_experts_input: torch.Tensor | None) -> bool:
    """Start the shared-expert multi-stream overlap, tolerating either upstream API.

    Upstream keeps flip-flopping on how the aux-stream launch is spelled: vllm#51838
    added ``SharedExperts.maybe_forward_async`` plus a ``shared_experts_overlapping``
    kwarg on ``MoERunner._apply_quant_method``, vllm#52024 reverted to
    ``MoERunner._maybe_sync_shared_experts_stream``, and vllm#52033 re-landed the
    #51838 shape. Feature-detect so the next flip does not break the HPU fast path.

    On Gaudi every shape is a no-op — upstream gates the multi-stream path on
    ``current_platform.is_cuda_alike()`` — so falling through to "no overlap" on an
    unrecognised future shape stays functionally correct.

    Args:
        runner: The upstream ``MoERunner`` (``self`` of the patched forward).
        shared_experts_input: Shared-expert input, or None when there are none.

    Returns:
        True iff the launch was asynchronous, meaning the caller must pass
        ``shared_experts_overlapping=True`` down to ``_apply_quant_method``.
    """
    sync_stream = getattr(runner, "_maybe_sync_shared_experts_stream", None)
    if sync_stream is not None:
        # vllm#52024 shape: runner-side sync, overlap decided internally.
        sync_stream(shared_experts_input)
        return False

    shared_experts = getattr(runner, "_shared_experts", None)
    if shared_experts is None or shared_experts_input is None:
        return False

    # vllm#51838 / vllm#52033 shape: launch on the aux stream, await it later.
    forward_async = getattr(shared_experts, "maybe_forward_async", None)
    if forward_async is None:
        return False
    return bool(forward_async(shared_experts_input))


def patched_fused_moe_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Patched forward that avoids graph breaks from ForwardContext lookups
    and dynamo per-layer string guards.

    Instead of calling _forward_impl (which uses _sequence_parallel_context
    and _maybe_dispatch — both of which access ForwardContext and cause
    torch.compile graph breaks), for dp_size==1 we inline the quant-config
    init, gate application, _apply_quant_method and _maybe_combine directly.
    This also bypasses self.layer_name (a per-layer string) so dynamo no
    longer emits per-layer string guards that trigger recompilation.

    After upstream PR #41184 (FusedMoE/MoERunner inversion), `self` IS the
    MoERunner: the expert weights and quant_method live on
    self.routed_experts, quant init moved to
    routed_experts._ensure_moe_quant_config_init(), and _apply_quant_method
    no longer takes a `layer` argument. The post-forward reduction sequence
    mirrors upstream MoERunner.forward so we stay in sync with the shared/
    fused output combination logic.
    """
    hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
    # Upstream _maybe_pad_hidden_states returns a 3-tuple: the (possibly padded)
    # hidden_states plus separate pre-transform and post-transform truncation
    # sizes (either may be None). Mirror MoERunner.forward, which keeps them
    # distinct — pre_xform strips fused_output before the routed-output
    # transform, post_xform strips the final output after all-reduce.
    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (self._maybe_pad_hidden_states(
        shared_experts_input, hidden_states))

    if self.moe_config.dp_size == 1:
        # Bypass _forward_impl entirely for dp_size==1 to eliminate
        # graph breaks from _sequence_parallel_context() (which calls
        # get_forward_context()), skip the no-op _maybe_dispatch(), and
        # avoid double gate / stream-sync calls that _forward_impl
        # would redundantly repeat.
        if self.moe_config.pcp_size > 1:
            raise RuntimeError("dp_size==1 fast path does not support pcp_size > 1")
        # Mirror MoERunner._forward_impl's pre-dispatch setup: the quant config
        # is initialized on routed_experts (which the runner holds directly), so
        # unlike the old FusedMoE-layer-based init we do NOT need the layer here.
        self.routed_experts._ensure_moe_quant_config_init()
        # Start the shared-expert aux stream before routed dispatch, mirroring
        # upstream MoERunner._forward_impl. Version-tolerant: see
        # _launch_shared_experts_overlap.
        shared_experts_overlapping = _launch_shared_experts_overlap(self, shared_experts_input)
        # Apply the gate if the runner holds it (mirrors _forward_impl).
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = torch.nn.functional.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)
        # Core MoERunner._apply_quant_method takes no layer argument — it reads
        # everything it needs off the runner (self.routed_experts / self.router).
        # Call it exactly as upstream _forward_impl does. Only pass
        # shared_experts_overlapping when an async launch actually happened: that
        # is the only case in which the kwarg is guaranteed to exist upstream.
        extra_quant_kwargs = {"shared_experts_overlapping": True} if shared_experts_overlapping else {}
        shared_output, fused_hidden = self._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
            input_ids=input_ids,
            **extra_quant_kwargs,
        )
        result = self._maybe_combine(shared_output, fused_hidden)
    else:
        # Upstream PR #41184 dropped MoERunner._trtllm_mxfp4_unpadded_dim(); the
        # TRT-LLM MXFP4 unpadded hidden dim now comes from moe_config and is only
        # non-zero when the quant method produces unpadded output. Mirror
        # MoERunner.forward exactly so we stay in sync with the custom-op signature.
        hidden_dim_unpadded = (self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0)
        result = self._forward_entry(hidden_states, router_logits, shared_experts_input, input_ids,
                                     self._encode_layer_name(), hidden_dim_unpadded)

    # Mirror upstream MoERunner.forward post-_forward_entry pipeline.
    if isinstance(result, tuple):
        shared_output, fused_output = result
    else:
        shared_output, fused_output = None, result

    # Trim padding from fused_output before the routed-output transform, matching
    # upstream MoERunner.forward (latent MoE with shared experts).
    if og_hidden_dim_pre_xform is not None:
        fused_output = fused_output[..., :og_hidden_dim_pre_xform]

    # Mirror upstream MoERunner.forward's two mutually-exclusive all-reduce
    # points, threaded by _fused_output_is_reduced: reduce a latent routed
    # output before its (possibly non-linear) transform, then reduce shared
    # output to match, else all-reduce the combined sum in the final step.
    fused_output_is_reduced = self._fused_output_is_reduced
    fused_output, fused_output_is_reduced = self._maybe_reduce_routed_output_before_transform(
        fused_output, fused_output_is_reduced)

    shared_output = self._maybe_reduce_shared_expert_output(shared_output, fused_output_is_reduced)
    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
    fused_output = self.apply_routed_output_transform(fused_output)

    combined = (shared_output + fused_output) if shared_output is not None else fused_output

    combined = self._maybe_reduce_final_output(combined, og_hidden_dim_post_xform, fused_output_is_reduced)
    return self._maybe_add_zero_expert_output(combined)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    This implementation uses a standard Python loop, which is compatible with
    graph compilation modes that do not support dynamic shapes resulting from
    operations like `torch.where`.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping a global expert index to its local index. Contains -1 for
            experts that are not assigned to the current rank.

    Returns:
        str: A string mapping from local to global index,
        ordered by global index.
            (e.g., "0->5, 1->12, 2->23")
    """
    mappings = []
    # A standard loop over a tensor with a known shape is statically analyzable.
    # `enumerate` provides the global_index (the position in the tensor) and
    # `local_index_tensor` (the value at that position).
    for global_index, local_index_tensor in enumerate(expert_map):
        local_index = local_index_tensor.item()
        # We only build strings for valid experts (those not marked as -1).
        if local_index != -1:
            mappings.append(f"{local_index}->{global_index}")

    return ", ".join(mappings)


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing parameters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    eplb_state: EplbLayerState | None = None,
    # zero expert parameters
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
    # Deepseek V4 vision routing bias parameters
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. ZeroExpertRouter - if zero_expert_type is not None
    3. GroupedTopKRouter - if use_grouped_topk is True
    4. CustomRoutingRouter - if custom_routing_function is not None
    5. FusedTopKBiasRouter - if e_score_correction_bias is not None
    6. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights

    Grouped topk arguments:
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)
        shared_expert_weight: Weight of the fused shared-expert slot (upstream
            vLLM d039c171+). Accepted for signature parity; unused on HPU since
            the fused-shared-expert path is not taken (default 1.0 no-op).

    Grouped topk and fused topk bias arguments:
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores

    Custom routing arguments:
        custom_routing_function: Optional custom routing function

    EPLB arguments:
        eplb_state: EPLB (Expert Parallelism Load Balancing) state

    Zero expert arguments:
        zero_expert_type: Type of zero expert (e.g. identity). If not None,
            creates a ZeroExpertRouter.
        num_logical_experts: Number of real (non-zero) experts. Required when
            zero_expert_type is not None.

    Hash Indices Table:
        hash_indices_table: Used to map input_ids to experts, needed for
            Deepseek V4

    Vision routing bias arguments (upstream vLLM PR 54566+):
        bias_vl: Vision routing bias for image tokens (Deepseek V4).
        image_sentinel_lo: First of five consecutive in-vocab image sentinel
            ids (0 = vision routing disabled). Both are forwarded to
            FusedTopKBiasRouter, which owns the vision-bias routing path.

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    if zero_expert_type is not None:
        assert num_logical_experts is not None, "num_logical_experts is required when zero_expert_type is set"
        assert e_score_correction_bias is not None, "e_score_correction_bias is required when zero_expert_type is set"
        return ZeroExpertRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            num_logical_experts=num_logical_experts,
            zero_expert_type=zero_expert_type,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )

    if use_grouped_topk:
        assert custom_routing_function is None
        if num_expert_group is None or topk_group is None:
            raise ValueError("num_expert_group and topk_group must be provided when use_grouped_topk is True")
        grouped_topk_router = GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        return grouped_topk_router

    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
        )

    assert scoring_func in ["sigmoid", "softmax", "sqrtsoftplus"]

    if e_score_correction_bias is not None or hash_indices_table is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            hash_indices_table=hash_indices_table,
            num_fused_shared_experts=num_fused_shared_experts,
            shared_expert_weight=shared_expert_weight,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )


# Apply patches
_orig_default_moe_runner_forward = MoERunnerBase.forward

# When enabled, bypasses the opaque torch.ops.vllm.moe_forward_shared custom
# op wrapper so that torch.ops.hpu.mixture_of_experts is captured directly in
# compiled Synapse graphs instead of running eagerly.
# Set HPU_FUSED_MOE=0 to disable and fall back to the original path.
_MOE_COMPILE = os.getenv("HPU_FUSED_MOE", "1") == "1"


def _patched_default_moe_runner_forward(self, *args, **kwargs):
    if _MOE_COMPILE:
        return patched_fused_moe_forward(self, *args, **kwargs)
    return _orig_default_moe_runner_forward(self, *args, **kwargs)


MoERunnerBase.forward = _patched_default_moe_runner_forward

# Note: after upstream PR #41184, FusedMoE is a factory function (not an
# nn.Module subclass), so there is no FusedMoE.__init__ to patch. The
# dp_size==1 fast path in patched_fused_moe_forward operates directly on the
# MoERunner (self.routed_experts / self.gate), so no stashed layer reference
# is required.

vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = get_compressed_expert_map
vllm.model_executor.layers.fused_moe.router.router_factory.create_fused_moe_router = create_fused_moe_router
vllm.model_executor.layers.fused_moe.layer.create_fused_moe_router = create_fused_moe_router

# Enable non-gated (is_act_and_mul=False) MoE on HPU.
#
# vLLM's FusedMoEConfig.__post_init__ ends with a guard that raises
# NotImplementedError for non-gated activations on any platform that is not
# CUDA/XPU/ROCm -- that guard lives in vLLM Python core and has no HPU branch,
# so it fires even though the Habana fused-MoE kernel now handles non-gated
# experts directly. Relax it: run the original __post_init__ and swallow only
# that guard. We identify the guard by CONFIG STATE (a non-gated activation),
# not by the exception message -- the message is prose upstream can reword at
# will, whereas ``is_act_and_mul`` is a stable public property. The guard is the
# final statement of __post_init__, so all other configuration has already
# completed by the time it fires -- swallowing it leaves a fully-initialized
# config. Drop this once vLLM core adds HPU to the supported-platform list.
from vllm.model_executor.layers.fused_moe import config as _vllm_moe_config  # noqa: E402

_orig_moe_config_post_init = _vllm_moe_config.FusedMoEConfig.__post_init__


def _patched_moe_config_post_init(self):
    try:
        _orig_moe_config_post_init(self)
    except NotImplementedError:
        # Swallow ONLY vLLM core's non-gated-activation platform guard, which on
        # HPU (is_cuda_alike()/is_xpu() both False) fires for every non-gated
        # activation. Identify it by config state, not the exception message: a
        # non-gated config (is_act_and_mul False) is the exact and only case the
        # guard targets. Any NotImplementedError from a gated config is unrelated
        # -- re-raise it untouched.
        if self.is_act_and_mul:
            raise


_vllm_moe_config.FusedMoEConfig.__post_init__ = _patched_moe_config_post_init
