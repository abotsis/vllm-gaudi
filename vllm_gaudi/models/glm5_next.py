# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (glm5_next) on HPU — text model, Phase 2.

Hybrid architecture: 33x KDA linear-attention layers + 11x DeepSeek-style
sparse-MLA layers (run DENSE on HPU in this phase — exact for ctx <= index_topk
(2048), approximation beyond; same trade the plugin makes for DeepSeek-V3.2),
manifold-constrained hyper-connections (mHC, 4 streams) on every layer,
288-expert MoE with sigmoid noaux_tc routing + shared expert, MTP layer 45
(weights loaded, execution deferred to Phase 4), FP8 128x128 blockwise
checkpoint (dense MLPs + experts stay FP8 via HPUFp8 paths).

Templates: upstream deepseek_v2 (MLA wrapper + loader), qwen3_next (hybrid
skeleton), plugin qwen3_5.py (HPU GDN forward pattern), deepseek_v4/xpu (mHC).
KDA core kernels: vllm_gaudi.ops.hpu_kda_eager (Phase-2 eager; Phase 3 adds
the fast chunk kernel).

Registered as plugin-first architecture:
  "Glm5NextForConditionalGeneration" -> HpuGlm5NextForConditionalGeneration

Vision tower is NOT built in this phase; vision weights are skipped by the
loader (text-only serving).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

# Make the plugin self-sufficient for MLA backend selection: upstream's
# ModelConfig.use_mla consults an allowlist of model types
# (model_arch_config_convertor.is_deepseek_mla) that does not know glm5_next,
# so without this wrap a fresh venv would fall back to the non-MLA attention
# path and crash at the HPU MLA wrapper (rope-dim-0 / latent layout).
# Wrapped here (at plugin import) so the repo carries the fix; drop when
# upstream adds glm5_next to the list.
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm_gaudi.extension.logger import logger as init_logger
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SwigluStepAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import MHCPreOp
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator, )

from vllm_gaudi.v1.worker.layer_diagnostic import boundary as _diag_layer_boundary
from vllm_gaudi.v1.worker.layer_diagnostic import attention_hooks as _diag_attention_hooks
from vllm_gaudi.v1.worker.layer_diagnostic import (kda_boundary as _diag_kda, kda_sequence_boundary as _diag_kda_seq,
                                                   kda_pool_rows as _diag_kda_pool)
from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_kda_eager import (chunk_kda_eager, chunk_kda_spec_states, kda_decode_step)
from vllm_gaudi.ops.hpu_kda_pytorch import hpu_chunk_kda
from vllm_gaudi.ops.hpu_kda_conv import hpu_kda_conv_update

import os

logger = init_logger()

_KDA_BIND_REPORTED = False
# VLLM_GLM_KDA_EAGER=1 forces the (slower) reference eager chunk kernel —
# useful for numerics debugging; default is the fast HPU kernel.
_KDA_FAST_CHUNK = os.environ.get("VLLM_GLM_KDA_EAGER", "0").strip().lower() not in ("1", "true")


def _glm_kda_chunk(q, k, v, g, beta, initial_state, chunk_size):
    if _KDA_FAST_CHUNK:
        return hpu_chunk_kda(q,
                             k,
                             v,
                             g,
                             beta,
                             initial_state=initial_state,
                             output_final_state=True,
                             use_qk_l2norm_in_kernel=True,
                             chunk_size=chunk_size)
    return chunk_kda_eager(q,
                           k,
                           v,
                           g=g,
                           beta=beta,
                           chunk_size=chunk_size,
                           initial_state=initial_state,
                           output_final_state=True,
                           use_qk_l2norm_in_kernel=True)


# ----------------------------------------------------------------------------
# KDA linear-attention layer
# ----------------------------------------------------------------------------


def _kda_load_slots(state_indices, num_accepted_tokens, n):
    """Slot holding the INCOMING recurrent state for each decode sequence.

    Under speculative decode the indices are [bs, 1 + num_spec+1]: column 0 is
    the CANONICAL slot (the block-table column prefill stored the post-prompt
    state to) and column 1+j is private candidate slot j, holding the state
    after draft token j of the previous verify step. ``num_accepted`` selects
    directly: 0 is the fresh sentinel (resume from canonical -- a request that
    just prefilled, chunk-continued, or ran a draftless step), and j >= 1
    resumes from candidate j-1. Without speculation the indices are 1-D and
    there is exactly one slot.
    """
    if state_indices.dim() == 1:
        return state_indices[:n]
    slots = state_indices.shape[1]
    if num_accepted_tokens is None:
        # Canonical slot. Materialised: a strided column view is not replay-safe.
        return state_indices[:n, 0].contiguous()
    col = torch.clamp(num_accepted_tokens[:n].long(), 0, slots - 1)
    return state_indices[:n].gather(1, col.view(-1, 1)).squeeze(1).contiguous()


def _kda_conv_slots(state_indices, n):
    """1-D conv-cache slot per sequence (candidate slots are SSM-only)."""
    if state_indices.dim() == 1:
        return state_indices[:n]
    return state_indices[:n, 0].contiguous()


def _kda_store_slots(store_indices, n):
    """Destination slot for a single-token (non-speculative) decode step."""
    if store_indices.dim() == 1:
        return store_indices[:n]
    return store_indices[:n, 0].contiguous()


def _kda_spec_decode(q,
                     k,
                     v,
                     g,
                     beta,
                     *,
                     ssm_state,
                     state_indices,
                     store_indices,
                     num_decodes,
                     spec_len,
                     num_accepted_tokens,
                     out_dtype,
                     masks=None):
    """KDA over a draft chain: score all spec_len tokens, keep every state.

    Returns core attention output [num_decodes*spec_len, H, V] and writes the
    state after token j into candidate slot j for every sequence.
    """
    n, L = num_decodes, spec_len
    H, D = q.shape[-2], q.shape[-1]
    V = v.shape[-1]
    pool = ssm_state.shape[0]

    load_slot = _kda_load_slots(state_indices, num_accepted_tokens, n)
    init = ssm_state.index_select(0, torch.remainder(load_slot, pool).long())

    # Decode buckets are rectangular: every sequence contributes exactly
    # spec_len tokens, laid out sequence-major.
    nl = n * L
    if nl != q.shape[0]:
        raise ValueError(f"speculative KDA expects a rectangular decode batch: {n} sequences x {L} tokens "
                         f"!= {q.shape[0]} rows. A ragged spec batch is a bucketing bug.")
    out, states = chunk_kda_spec_states(q[:nl].view(n, L, H, D),
                                        k[:nl].view(n, L, H, D),
                                        v[:nl].view(n, L, H, V),
                                        g[:nl].view(n, L, H, D),
                                        beta[:nl].view(n, L, H),
                                        initial_state=init,
                                        masks=masks)

    if store_indices.dim() == 2:
        # Columns: [canonical, cand_0 .. cand_{k-1}]. Candidate j's state goes
        # to its private slot, and the LOADED state (the canonical state for
        # the tokens computed so far) is propagated to the canonical column in
        # the same scatter -- keeping the block-table column the non-spec path,
        # the conv cache, and a preemption-resume all key on up to date, at the
        # cost of zero extra kernel launches.
        tgt = torch.cat([store_indices[:n, 0:1], store_indices[:n, 1:1 + L]], dim=1).reshape(-1)
        # [n,H,L,D,V] -> [n,L,H,D,V]; prepend init as the canonical row.
        flat_new = torch.cat([init.unsqueeze(1), states.permute(0, 2, 1, 3, 4).to(init.dtype)], dim=1)
        flat_new = flat_new.reshape(n * (L + 1), H, D, V)
    else:
        tgt = store_indices.view(-1, 1).expand(-1, L)[:n, :L].reshape(-1)
        # [n,H,L,D,V] -> [n,L,H,D,V] -> [n*L,H,D,V], matching flat_idx order
        flat_new = states.permute(0, 2, 1, 3, 4).reshape(nl, H, D, V)
    flat_idx = torch.remainder(tgt, pool).long()
    # Padded lanes carry -1, which remainder() wraps onto a live slot; make
    # their write-back an identity, as the single-token path does.
    cur = ssm_state.index_select(0, flat_idx)
    flat_new = torch.where((tgt < 0).view(-1, 1, 1, 1), cur, flat_new.to(cur.dtype))
    _kda_scatter_states(ssm_state, flat_idx, flat_new)
    return out.reshape(nl, H, V).to(out_dtype)


@torch._dynamo.disable
def _kda_scatter_states(ssm_state, flat_idx, flat_new):
    """index_copy_ in eager -- HPU torch.compile drops aliased index_copy_."""
    ssm_state.index_copy_(0, flat_idx, flat_new)


@torch._dynamo.disable
def _kda_save_state(final_state, ssm_state, state_indices, guard_padding=False):
    """index_copy_ in eager — HPU torch.compile drops aliased index_copy_.

    With `guard_padding`, rows whose state index is -1 (padding in a prefill
    bucket wider than the batch) write back their current value instead:
    remainder() would otherwise wrap -1 onto the pool's last slot and clobber
    a live sequence's state. The decode path passes real rows only and keeps
    its recipe unchanged."""
    if final_state is None or ssm_state is None or state_indices is None:
        return
    safe = torch.remainder(state_indices, ssm_state.shape[0]).long()
    new = final_state.to(ssm_state.dtype)
    if guard_padding:
        cur = ssm_state.index_select(0, safe)
        new = torch.where((state_indices < 0).view(-1, *([1] * (new.dim() - 1))), cur, new)
    ssm_state.index_copy_(0, safe, new)


class HpuGlm5NextKdaAttention(GatedDeltaNetAttention):
    """GLM-5.3 KDA layer (per-channel forget gate, Kimi delta rule).

    Module structure matches raw checkpoint names. q/k/v are merged into one
    ColumnParallel GEMM (stacked loading). Small per-channel tensors
    (dt_bias, A_log, conv weights) are kept REPLICATED full-size and sliced
    per TP rank in forward (avoids custom TP loaders; sizes are KBs).
    """

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        GatedDeltaNetAttention.__init__(self, config, vllm_config=vllm_config, prefix=prefix)
        _spec = getattr(vllm_config, "speculative_config", None)
        if _spec is not None and vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("GLM KDA speculative state publication requires prefix caching to be disabled.")
        # Query length of a speculative verify step (num_spec + 1); 1 when off.
        self._spec_verify_len = (_spec.num_speculative_tokens + 1) if _spec is not None else 1
        self._spec_masks_by_len = {}

        lac = config.linear_attn_config
        self.num_heads_total = lac["num_heads"]
        self.head_dim = lac["head_dim"]
        self.conv_kernel_size = lac["short_conv_kernel_size"]
        self.lower_bound = lac.get("gate_lower_bound", -5.0)
        self.qkv_dim = self.num_heads_total * self.head_dim
        self.num_heads = divide(self.num_heads_total, self.tp_size)
        self.qkv_dim_local = divide(self.qkv_dim, self.tp_size)
        self.kda_eps = config.rms_norm_eps

        quant_config = vllm_config.quant_config
        self.qkv_proj = MergedColumnParallelLinear(config.hidden_size, [self.qkv_dim] * 3,
                                                   bias=False,
                                                   quant_config=quant_config,
                                                   prefix=f"{prefix}.qkv_proj")
        # forget gate: f_a (replicated low-rank head_dim) -> f_b (sharded channels)
        self.f_a_proj = ReplicatedLinear(config.hidden_size,
                                         self.head_dim,
                                         bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.f_a_proj")
        self.f_b_proj = ColumnParallelLinear(self.head_dim,
                                             self.qkv_dim,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.f_b_proj")
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads_total), requires_grad=False)
        # beta gate per head
        self.b_proj = ColumnParallelLinear(config.hidden_size,
                                           self.num_heads_total,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.b_proj")
        # output gate: g_a (replicated) -> g_b (sharded channels)
        self.g_a_proj = ReplicatedLinear(config.hidden_size,
                                         self.head_dim,
                                         bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.g_a_proj")
        self.g_b_proj = ColumnParallelLinear(self.head_dim,
                                             self.qkv_dim,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.g_b_proj")
        # depthwise conv weights (replicated full-size; sliced in forward)
        for name_ in ("q_conv1d", "k_conv1d", "v_conv1d"):
            self.register_buffer(f"{name_}_weight",
                                 torch.zeros(self.qkv_dim, 1, self.conv_kernel_size),
                                 persistent=True)
        self.o_norm = nn.Parameter(torch.ones(self.head_dim), requires_grad=False)
        self.o_proj = RowParallelLinear(self.qkv_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.kv_cache: list[torch.Tensor] = []
        self.cache_group_idx = None
        self.mamba_chunk_size = 64  # eager chunk kernel chunk size

        self._tp_slice = slice(self.tp_rank * self.qkv_dim_local, (self.tp_rank + 1) * self.qkv_dim_local)
        self._head_slice = slice(self.tp_rank * self.num_heads, (self.tp_rank + 1) * self.num_heads)

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"duplicate static forward context prefix {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # -- MambaBase interfaces -------------------------------------------
    def get_state_dtype(self):
        return MambaStateDtypeCalculator.kda_state_dtype(self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self):
        return MambaStateShapeCalculator.kda_state_shape(self.tp_size,
                                                         self.num_heads_total,
                                                         self.head_dim,
                                                         conv_kernel_size=self.conv_kernel_size,
                                                         num_spec=self.num_spec)

    def _resolve_state_indices(self, attn_metadata, field="load_indices_tensor"):
        """Per-group state slot ids from ``field``.

        load_indices_tensor  = block holding the last already-COMPUTED token,
                               i.e. where the incoming recurrent state lives.
        store_indices_tensor = block holding the last SCHEDULED token, i.e.
                               where this step's final state must land.

        The runner builds them from different block offsets
        (block_idx_last_computed_token vs block_idx_last_scheduled_token), so
        they diverge as soon as the scheduled chunk crosses a mamba block
        boundary -- routine under mamba cache mode 'align', which vLLM enables
        automatically whenever prefix caching is on. Writing the final state
        back to the LOAD slot leaves the store slot holding a state that never
        saw this chunk, and the next step reads that stale slot: the first
        generated token is still right (its logits come from prefill) and
        everything after it is garbage. hpu_mamba_mixer2.py stores to
        store_indices_tensor for exactly this reason.
        """
        indices = getattr(attn_metadata, field, None)
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            # .contiguous(): under HPU graphs a strided view of the group
            # tensor replays as "Neither storage attached to input tensor,
            # not its view". Speculative decode makes this 3-D
            # ([groups, bs, candidate slots]), so the squeeze leaves a view
            # whose base is not itself a graph input.
            indices = indices.index_select(0, cg.view(1)).squeeze(0).contiguous()
        return indices

    def _extract_metadata(self, num_tokens):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, None, 0, 0, 0, 0, None, None)
        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)
        store_indices = self._resolve_state_indices(attn_metadata, "store_indices_tensor")
        if store_indices is None:
            store_indices = state_indices
        conv_state = self.kv_cache[0] if self.kv_cache else None
        ssm_state = self.kv_cache[1] if self.kv_cache else None
        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        num_accepted_tokens = getattr(attn_metadata, "num_accepted_tokens", None)

        if not is_prompt:
            num_decodes = int(getattr(attn_metadata, "num_gdn_decodes", 0))
            if num_decodes == 0:
                num_decodes = (state_indices.shape[0] if state_indices is not None else
                               (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        if is_prompt and not _KDA_BIND_REPORTED:
            # One-shot: hpu_chunk_kda takes initial_state as an OPTIONAL, so a
            # missing KV cache or state index surfaces as the opaque
            # "Empty tensor optional" from inside the fused kernel rather than
            # as a Python error. Say which input is actually absent.
            globals()["_KDA_BIND_REPORTED"] = True
            logger.warning(
                "[GLM] KDA prefill inputs: kv_cache=%s ssm_state=%s conv_state=%s "
                "state_indices=%s store_indices=%s", "none" if not self.kv_cache else f"len={len(self.kv_cache)}",
                None if ssm_state is None else tuple(ssm_state.shape),
                None if conv_state is None else tuple(conv_state.shape),
                None if state_indices is None else tuple(state_indices.shape),
                None if store_indices is None else tuple(store_indices.shape))
        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None and ssm_state is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            initial_state = ssm_state[state_indices].contiguous()
            if has_initial_state is not None:
                mask = has_initial_state.bool().view(-1, 1, 1, 1).to(initial_state.dtype)
                initial_state = initial_state * mask
        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_accepted_tokens, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state,
                store_indices)

    # -- load-time derived constants ------------------------------------
    def _build_spec_masks(self) -> None:
        """Materialise the verify-step constant masks ONCE, after weight load.

        A module-level cache is not enough: these are HPU tensors created inside
        a lazy region, and reusing one across mark_steps fails with
        "ValidateSyncInputTensors tensor_data is empty". Registering them as
        buffers here (the same reason _build_clamp_aux builds its aux tensors at
        load time rather than on first forward) gives them module-owned storage.
        """
        # The verify step does NOT always carry num_spec+1 tokens: the scheduler
        # can schedule fewer draft tokens (ngram with a short lookup, a partly
        # rejected chain, the tail of a request). Build a mask set for every
        # length that can occur, or the kernel gets [5,5] masks for an L=2 batch
        # ("size of tensor a (2) must match tensor b (5)").
        dev = next(self.parameters()).device
        self._spec_masks_by_len = {}
        for L in range(2, max(self._spec_verify_len, 1) + 1):
            ones = torch.ones(L, L, dtype=torch.bool, device=dev)
            names = (f"_spec_eye_{L}", f"_spec_striu_{L}", f"_spec_triu_{L}", f"_spec_tril_{L}")
            vals = (torch.eye(L, dtype=torch.float32, device=dev), torch.triu(ones, 1), torch.triu(ones, 0),
                    torch.tril(torch.ones(L, L, dtype=torch.float32, device=dev)))
            for n, v in zip(names, vals):
                self.register_buffer(n, v, persistent=False)
            self._spec_masks_by_len[L] = tuple(getattr(self, n) for n in names)

    def build_decode_constants(self) -> None:
        """Materialize TP-sliced / dtype-cast constants ONCE after weight load.

        Every one of these is a pure function of loaded parameters, but the
        forward path recomputed them on each call: per KDA layer per step that
        is 4 narrows, 4 casts, an exp and a 3-way cat -- and there are 34 KDA
        layers, so ~400 redundant device ops on every decode step. Decode is
        dispatch-bound (measured: 458 cached-graph dispatches and ~6.8k device
        idle gaps per step against 6.3 ms of actual GEMM), so removing
        dispatches is the point, not removing FLOPs.

        Built at load rather than on first forward: under wrap_in_hpu_graph the
        first forward is the capture, and tensors created there are frozen into
        the graph rather than being valid inputs -- the same rule the fused-MoE
        aux buffers follow.
        """
        with torch.no_grad():
            self.register_buffer("_dt_bias_local", self.dt_bias[self._tp_slice].float().contiguous(), persistent=False)
            self.register_buffer("_A_local",
                                 self.A_log[self._head_slice].float().exp().view(1, -1, 1).contiguous(),
                                 persistent=False)
            self.register_buffer("_o_norm_f32", self.o_norm.float().contiguous(), persistent=False)
            self.register_buffer("_conv_w_local",
                                 torch.cat([
                                     self.q_conv1d_weight[self._tp_slice],
                                     self.k_conv1d_weight[self._tp_slice],
                                     self.v_conv1d_weight[self._tp_slice],
                                 ],
                                           dim=0).squeeze(1).contiguous(),
                                 persistent=False)
        self._build_spec_masks()
        logger.debug("[GLM] KDA decode constants built (conv_w=%s A=%s spec_len=%s)", tuple(self._conv_w_local.shape),
                     tuple(self._A_local.shape), self._spec_verify_len)

    # -- forward --------------------------------------------------------
    def _forget_gate(self, x: torch.Tensor) -> torch.Tensor:
        """g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)) ∈
        [lower_bound, 0]; per (head, channel). [T, H_local, D]"""
        fg = self.f_b_proj(self.f_a_proj(x)[0])[0]
        dt_b = getattr(self, "_dt_bias_local", None)
        A = getattr(self, "_A_local", None)
        if dt_b is None:  # constants not built (non-standard load path)
            dt_b = self.dt_bias[self._tp_slice].float()
            A = self.A_log[self._head_slice].float().exp().view(1, -1, 1)
        fg = fg.float() + dt_b
        return self.lower_bound * torch.sigmoid(A * fg.view(-1, self.num_heads, self.head_dim))

    def _gated_o_norm(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = core.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.kda_eps)
        _on = getattr(self, "_o_norm_f32", None)
        x = x * (self.o_norm.float() if _on is None else _on)
        x = x * torch.sigmoid(gate.float())
        return x.to(core.dtype)

    def _conv_weight_local(self) -> torch.Tensor:
        w = getattr(self, "_conv_w_local", None)
        if w is not None:
            return w
        return torch.cat([
            self.q_conv1d_weight[self._tp_slice],
            self.k_conv1d_weight[self._tp_slice],
            self.v_conv1d_weight[self._tp_slice],
        ],
                         dim=0).squeeze(1)  # [3*local, KW]

    def forward_orig(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = x.shape[0]

        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_accepted_tokens, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state,
         store_indices) = self._extract_metadata(num_tokens)

        mixed_qkv, _ = self.qkv_proj(x)
        g = self._forget_gate(x)  # [T, H, D] in (-lb, 0)
        beta = torch.sigmoid(self.b_proj(x)[0].float())  # [T, H]
        _diag_kda(self, "forget_gate", g)
        _diag_kda(self, "beta", beta)
        conv_w = self._conv_weight_local()

        H, D = self.num_heads, self.head_dim
        core = torch.zeros(num_tokens, H, D, dtype=x.dtype, device=x.device)

        if conv_state is not None and is_prompt:
            import habana_frameworks.torch.core as _htc
            _htc.mark_step()  # flush compiled-region outputs before the conv
            # state write binds them in the lazy bridge (storage attachment)
            if padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens:
                token_mask = padding_mask_flat.view(-1, 1).to(mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask
                g = g * token_mask.view(-1, 1, 1).to(g.dtype)
                beta = beta * token_mask.view(-1, 1).to(beta.dtype)

            qkv = hpu_causal_conv1d_fn(
                x=mixed_qkv.transpose(0, 1).contiguous(),
                weight=conv_w,
                bias=None,
                conv_states=conv_state,
                query_start_loc=query_start_loc,
                activation="silu",
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                store_cache_indices=store_indices,
                block_size_to_align=mamba_block_size,
                is_prompt=True,
            ).transpose(0, 1)
            if padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens:
                qkv = qkv * token_mask
            _diag_kda(self, "conv_out", qkv)
            _diag_kda_seq(self, "ssm_state_in", initial_state)

            T = num_tokens
            q = qkv[:, :self.qkv_dim_local].view(1, T, H, D)
            k = qkv[:, self.qkv_dim_local:2 * self.qkv_dim_local].view(1, T, H, D)
            v = qkv[:, 2 * self.qkv_dim_local:].view(1, T, H, D)
            gg, bb = g.view(1, T, H, D), beta.view(1, T, H)
            init = initial_state
            if prefill_num_seqs > 0 and prefill_seq_len > 0 and prefill_num_seqs * prefill_seq_len == T:
                B_, S_ = prefill_num_seqs, prefill_seq_len
                q, k, v = (t.view(B_, S_, H, D) for t in (q, k, v))
                gg, bb = gg.view(B_, S_, H, D), bb.view(B_, S_, H)
            else:
                B_, S_ = 1, T

            core_b, final_state = _glm_kda_chunk(q, k, v, gg, bb, init, self.mamba_chunk_size)
            core = core_b.reshape(T, H, D)
            _diag_kda_seq(self, "ssm_state_out", final_state)
            _kda_save_state(final_state, ssm_state, store_indices, guard_padding=prefill_num_seqs > 1)

        elif conv_state is not None:
            # The decode-side mark_step was a "twin" of the prefill boundary,
            # but the prefill one exists to bind conv-pool storage before the
            # advanced-index pool WRITE (hpu__slice_insert under graph
            # capture); the decode path updates conv state through
            # hpu_causal_conv1d_update instead. Each mark_step is a graph
            # segment boundary, and at 34 KDA layers that was ~16% of the 218
            # enqueues per decode step.
            # Tokens per sequence this step: 1 for ordinary decode, num_spec+1
            # while verifying a draft chain.
            spec_len = (num_tokens // num_decodes) if num_decodes > 0 else 1
            if state_indices is not None and state_indices.dim() == 2:
                qkv = hpu_kda_conv_update(
                    x=mixed_qkv,
                    pool=conv_state,
                    weight=conv_w,
                    load_indices=state_indices[:num_decodes],
                    store_indices=store_indices[:num_decodes],
                    accepted=num_accepted_tokens[:num_decodes],
                    query_start_loc=query_start_loc[:num_decodes + 1],
                    max_query_len=spec_len,
                    activation="silu",
                )
            else:
                conv_slots = _kda_conv_slots(state_indices, num_decodes) if state_indices is not None else None
                _diag_kda_pool(self, "conv_pool_in", conv_state, conv_slots)
                qkv = hpu_causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_w,
                    bias=None,
                    activation="silu",
                    conv_state_indices=conv_slots,
                    query_start_loc=(query_start_loc[:num_decodes + 1] if query_start_loc is not None else None),
                )
                _diag_kda_pool(self, "conv_pool_out", conv_state, conv_slots)
            _diag_kda(self, "conv_out", qkv)
            T = num_tokens
            q = qkv[:, :self.qkv_dim_local].view(T, H, D)
            k = qkv[:, self.qkv_dim_local:2 * self.qkv_dim_local].view(T, H, D)
            v = qkv[:, 2 * self.qkv_dim_local:].view(T, H, D)

            if num_decodes > 0 and ssm_state is not None and spec_len > 1:
                # --- speculative verify: q_len = num_spec+1 tokens/sequence ---
                # kda_decode_step consumes exactly one token per sequence, so it
                # cannot score a draft chain. Run the whole chain in ONE call
                # and keep the state at EVERY candidate position: acceptance is
                # only known after sampling, and unlike the conv cache a
                # contracted recurrent state cannot be rewound. Candidate slot j
                # (indices column 1+j) holds the state after token j; the next
                # step resumes from column num_accepted, where 0 is the fresh
                # sentinel pointing at the canonical block-table slot.
                core[:] = _kda_spec_decode(q,
                                           k,
                                           v,
                                           g.view(T, H, D),
                                           beta.view(T, H),
                                           ssm_state=ssm_state,
                                           state_indices=state_indices,
                                           store_indices=store_indices,
                                           num_decodes=num_decodes,
                                           spec_len=spec_len,
                                           num_accepted_tokens=num_accepted_tokens,
                                           out_dtype=core.dtype,
                                           masks=self._spec_masks_by_len.get(spec_len))
            elif num_decodes > 0 and ssm_state is not None:
                n = num_decodes
                states = ssm_state.index_select(
                    0,
                    torch.remainder(_kda_load_slots(state_indices, num_accepted_tokens, n), ssm_state.shape[0]).long())
                # kda_decode_step updates `states` in place; the snapshot below is
                # ordered before it in the lazy graph, so it holds the loaded state.
                _diag_kda(self, "ssm_state_in", states)
                out_n, states_new = kda_decode_step(states, q[:n], k[:n], v[:n],
                                                    g.view(T, H, D)[:n],
                                                    beta.view(T, H)[:n])
                _diag_kda(self, "kda_out", out_n)
                # Padded decode lanes carry the sentinel -1, and the
                # remainder() above wraps -1 onto the LAST REAL slot (N-1) --
                # so an unguarded write-back clobbers whichever live sequence
                # owns that slot. n is the padded bucket size (the runner
                # never sets num_gdn_decodes, so the getattr default of 0
                # always falls through to state_indices.shape[0]), meaning pad
                # lanes are present at every bucket above the batch size.
                # Make their write-back an identity: store back exactly what
                # they read. Prefill needs no such guard while prompt buckets
                # stay bs=1.
                _store = _kda_store_slots(store_indices, n)
                pad = (_store < 0).view(-1, *([1] * (states_new.dim() - 1)))
                states_new = torch.where(pad, states, states_new.to(states.dtype))
                _diag_kda(self, "ssm_state_out", states_new)
                _kda_save_state(states_new, ssm_state, _store)
                core[:n] = out_n.to(core.dtype)
            # This tail exists for a MIXED batch, where rows past the decode
            # sequences are extra (prefill-ish) tokens with no incoming state.
            # Under speculation T == num_decodes * spec_len, so `T > num_decodes`
            # is true on every verify step -- but _kda_spec_decode has already
            # written all T rows, and re-running them through the chunk kernel
            # with initial_state=None hands the fused op an unbound optional
            # ("Empty tensor optional").
            if num_decodes < T and spec_len == 1:
                extra, _ = _glm_kda_chunk(q[num_decodes:].unsqueeze(0), k[num_decodes:].unsqueeze(0),
                                          v[num_decodes:].unsqueeze(0),
                                          g.view(T, H, D)[num_decodes:].unsqueeze(0),
                                          beta.view(T, H)[num_decodes:].unsqueeze(0), None, max(1, T - num_decodes))
                core[num_decodes:] = extra.squeeze(0).to(core.dtype)

        _diag_kda(self, "core", core)
        gate = self.g_b_proj(self.g_a_proj(x)[0])[0].view(-1, H, D)
        _diag_kda(self, "gate", gate)
        out = self._gated_o_norm(core, gate).view(num_tokens, -1)
        out, _ = self.o_proj(out)
        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_orig(hidden_states)


# ----------------------------------------------------------------------------
# Sparse-MLA layer (dense on HPU this phase; indexer off)
# ----------------------------------------------------------------------------


class _GlmFusedQkvAProj(MergedColumnParallelLinear):
    """Fused [q_a | kv_a(+rope)] projection, replicated across TP
    (disable_tp), matching upstream DeepseekV2FusedQkvAProjLinear. Checkpoint
    weights stack onto shards 0/1 via _GLM5_HF_TO_VLLM_MAPPER."""

    def __init__(self, config, quant_config, prefix: str):
        super().__init__(config.hidden_size, [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
                         bias=False,
                         quant_config=quant_config,
                         disable_tp=True,
                         prefix=prefix)


class HpuGlm5NextSparseAttention(nn.Module):
    """DeepSeek-style MLA with GLM-5.3 dims: q_lora 1536, kv_lora 512,
    qk_nope 256, qk_rope 0 (NoPE), v 256, 64 heads. Upstream
    MultiHeadLatentAttentionWrapper; HPU OOT impl plugs in automatically."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.num_heads_total = config.num_attention_heads
        self.num_heads = divide(self.num_heads_total, tp_size)
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 0 (NoPE)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = _GlmFusedQkvAProj(config, quant_config, prefix)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                             self.num_heads_total * self.qk_head_dim,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_b_proj")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank,
                                              self.num_heads_total * (self.qk_nope_head_dim + self.v_head_dim),
                                              bias=False,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.num_heads_total * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,  # NoPE
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,  # dense this phase (Phase 5: indexer)
            is_sparse=False,
            topk_indices_buffer=None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(self.hidden_size,
                                                        self.num_heads,
                                                        self.scaling,
                                                        self.qk_nope_head_dim,
                                                        self.qk_rope_head_dim,
                                                        self.v_head_dim,
                                                        self.q_lora_rank,
                                                        self.kv_lora_rank,
                                                        mla_modules,
                                                        vllm_config.cache_config,
                                                        quant_config,
                                                        prefix,
                                                        skip_topk=False)

    def forward(self, positions, hidden_states):
        return self.mla_attn(positions, hidden_states, None)


# ----------------------------------------------------------------------------
# MLP / MoE
# ----------------------------------------------------------------------------


class HpuGlm5NextMLP(nn.Module):
    """Dense MLP with GLM clamped SwiGLU (swiglu_limit)."""

    def __init__(self,
                 config,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        quant_config = vllm_config.quant_config
        self.swiglu_limit = config.swiglu_limit
        self.intermediate_size = (intermediate_size if intermediate_size is not None else config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [self.intermediate_size] * 2,
                                                       bias=False,
                                                       quant_config=quant_config,
                                                       prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(self.intermediate_size,
                                           config.hidden_size,
                                           bias=False,
                                           reduce_results=reduce_results,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act = SwigluStepAndMul(self.swiglu_limit)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)[0]
        x = self.act(gate_up)
        out = self.down_proj(x)[0]
        return out


class HpuGlm5NextMoE(nn.Module):
    """288-expert MoE: sigmoid noaux_tc routing + shared expert.

    Routed experts run the clamped-SwiGLU path (ops/hpu_fused_moe.py
    ``_silu_clamp_moe``): the Habana fused MoE op supports only plain
    silu/gelu and silently drops ``swiglu_limit``. Set
    VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT=1 to restore the fused op."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._prefix = prefix
        quant_config = vllm_config.quant_config
        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.gate")
        self.gate_e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts), requires_grad=False)
        # Fold the shared expert into the FusedMoE call (the DeepSeek-V2
        # pattern): computing it separately costs a SECOND all-reduce per MoE
        # layer (~a third of the per-step collective count) plus its extra
        # graph-recipe boundaries. Folded, the runner sums shared + routed and
        # issues a single all-reduce ("_maybe_reduce_final_output"), so the
        # shared MLP must NOT reduce on its own -- hence reduce_results=False.
        self._fold_shared = True
        self.shared_experts = HpuGlm5NextMLP(config,
                                             vllm_config,
                                             prefix=f"{prefix}.shared_experts",
                                             intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                                             reduce_results=not self._fold_shared)
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts if self._fold_shared else None,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            e_score_correction_bias=self.gate_e_score_correction_bias,
            router_logits_dtype=torch.float32,
            # GLM clamped SwiGLU: the Habana fused MoE op drops the limit, so
            # the plugin routes silu+swiglu_limit layers to the clamped expert
            # path (ops/hpu_fused_moe.py _silu_clamp_moe).
            swiglu_limit=config.swiglu_limit,
        )

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])
        router_logits = self.gate(x.to(torch.float32))[0]
        # Routing evidence for the batch-shape diagnostic: fp32 gate output per
        # token plus the (constant) selection bias, so top-k and its margin can
        # be recomputed offline (n_group=1 => plain top-k of sigmoid(x)+bias).
        _diag_layer_boundary(self._prefix + ".router_logits", router_logits)
        _diag_layer_boundary(self._prefix + ".router_bias",
                             self.gate_e_score_correction_bias.view(1, -1),
                             row_indices=[0])
        out = self.experts(hidden_states=x, router_logits=router_logits)
        return out.view(shape)


# ----------------------------------------------------------------------------
# The TPC mHC fused-kernel experiment was retired: its per-launch cost ate the
# MME win it targeted (multi-output custom ops don't split recipes), so the
# torch-native mHC path below is the production path.

# mHC (manifold-constrained hyper-connections)
# ----------------------------------------------------------------------------


def _mhc_pre_hpu_enabled() -> bool:
    """Env gate for the bf16-mix mHC pre path (VLLM_GLM_MHC_MIX_DTYPE).

    "bf16" (default): mix GEMM runs as a bf16 MME GEMM — the fp32 GEMV
    (N = (2+hc)*hc ~= 24 lanes over hc_mult*hidden reduce) is the decode
    critical path at batch 1. Gate math (sigmoid/softmax/sinkhorn, sqrsum)
    stays fp32. "fp32" restores the exact upstream numerics.
    """
    return os.environ.get("VLLM_GLM_MHC_MIX_DTYPE", "bf16").strip().lower() == "bf16"


def _mhc_pre_hpu(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                 sinkhorn_repeat):
    """mhc_pre_torch with the mix GEMM on the bf16 MME path.

    Mirrors vllm.model_executor.kernels.mhc.torch.mhc_pre_torch; the ONLY
    difference is ``mixes = matmul(x_bf16, fn_bf16.t()).float()`` (bf16 MME
    with fp32 accumulation, single bf16 rounding of the 24-lane output)
    instead of an fp32 GEMV. ``residual`` is already bf16 (lost nothing:
    upstream casts the same bf16 to fp32 for the GEMV); all gate math keeps
    fp32 precision. Output dtypes/signatures match mhc_pre_torch.
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_hidden_size)
    mixes = torch.matmul(x, fn.to(torch.bfloat16).t()).float()
    sqrsum = x.float().square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (mixes[:, 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]).view(num_tokens, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


class HpuGlm5NextHyperConnection(nn.Module):
    """One mHC site (attn or ffn). fp32 params matching the checkpoint.

    The pre-block dispatches to :func:`_mhc_pre_hpu` on HPU (bf16 MME mix
    GEMM, fp32 gate math; the upstream fp32-GEMV mix is the decode critical
    path at batch 1) and to upstream's ``MHCPreOp`` elsewhere.
    """

    def __init__(self, config, prefix: str = ""):
        super().__init__()
        hc = config.hc_mult
        mix = (2 + hc) * hc
        hc_dim = hc * config.hidden_size
        self.fn = nn.Parameter(torch.zeros(mix, hc_dim, dtype=torch.float32), requires_grad=False)
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=False)
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.sinkhorn_repeat = config.hc_sinkhorn_iters
        self.mhc_pre = MHCPreOp()

    def forward(self, residual: torch.Tensor):
        """residual [T, hc, hidden] -> (post_mix [T, hc, 1],
        comb [T, hc, hc], layer_input [T, hidden])"""
        if residual.device.type == "hpu" and _mhc_pre_hpu_enabled():
            return _mhc_pre_hpu(
                residual,
                self.fn,
                self.scale,
                self.base,
                self.rms_eps,
                self.hc_eps,
                self.hc_eps,
                2.0,
                self.sinkhorn_repeat,
            )
        return self.mhc_pre(
            residual=residual,
            fn=self.fn,
            hc_scale=self.scale,
            hc_base=self.base,
            rms_eps=self.rms_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=self.sinkhorn_repeat,
        )


def _mhc_post(x, residual, post_mix, comb_mix):
    return _mhc_post_torch(x, residual, post_mix, comb_mix)


def _mhc_post_torch(x, residual, post_mix, comb_mix):
    """residual[..., j, :] = sum_i comb[i, j] * residual[..., i, :] +
    post_mix[j] * x  (fp32 math) — mirrors vllm mhc_post_torch.

    NB: written einsum-free. The einsum lowering on the HPU compile path
    materializes broken zero/one-width sections (the [8,4,1,4]->[8,4,256]
    reshape crash); explicit permute+broadcast+sum compiles cleanly."""
    mixed = (comb_mix.float().permute(0, 2, 1).unsqueeze(-1) * residual.float().unsqueeze(1)).sum(dim=2)
    post = post_mix.float().squeeze(-1).unsqueeze(-1) * x.float().unsqueeze(1)
    return (mixed + post).to(residual.dtype)


# ----------------------------------------------------------------------------
# Decoder layer
# ----------------------------------------------------------------------------


class HpuGlm5NextDecoderLayer(nn.Module):

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._prefix = prefix
        self.layer_type = config.layer_types[extract_layer_index(prefix)]
        if self.layer_type == "linear_attention":
            self.self_attn = HpuGlm5NextKdaAttention(config, vllm_config, prefix=f"{prefix}.self_attn")
            self._is_linear = True
        else:
            self.self_attn = HpuGlm5NextSparseAttention(config, vllm_config, prefix=f"{prefix}.self_attn")
            self._is_linear = False

        self.attn_hc = HpuGlm5NextHyperConnection(config)
        self.ffn_hc = HpuGlm5NextHyperConnection(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        self._diag_split_layer0 = layer_idx == 0
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = HpuGlm5NextMoE(config, vllm_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = HpuGlm5NextMLP(config, vllm_config, prefix=f"{prefix}.mlp")

    def forward(self, positions, hidden_streams):
        # hidden_streams: [T, hc, hidden]
        post_a, comb_a, x = self.attn_hc(hidden_streams)
        if self._diag_split_layer0:
            # All three outputs are token-leading: [T, hc, 1], [T, hc, hc], [T, hidden].
            _diag_layer_boundary(self._prefix + ".attention_mhc_pre_input", x)
            _diag_layer_boundary(self._prefix + ".attention_mhc_pre_post_a", post_a)
            _diag_layer_boundary(self._prefix + ".attention_mhc_pre_comb_a", comb_a)
        x = self.input_layernorm(x)
        if self._diag_split_layer0:
            _diag_layer_boundary(self._prefix + ".attention_norm_input", x)
        with _diag_attention_hooks(self):
            if self._is_linear:
                attn_out = self.self_attn(hidden_states=x)
            else:
                attn_out = self.self_attn(positions=positions, hidden_states=x)
        if self._diag_split_layer0:
            _diag_layer_boundary(self._prefix + ".attention_raw_output", attn_out)
        residual = _mhc_post(attn_out, hidden_streams, post_a, comb_a)
        _diag_layer_boundary(self._prefix + ".attention_mhc_post", residual)

        post_f, comb_f, x = self.ffn_hc(residual)
        x = self.post_attention_layernorm(x)
        ffn_out = self.mlp(x)
        residual = _mhc_post(ffn_out, residual, post_f, comb_f)
        _diag_layer_boundary(self._prefix + ".ffn_mhc_post", residual)
        return residual


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


# torch.compile opt-in. vLLM only compiles model classes carrying this
# decorator -- CompilationMode alone does nothing. Without it, PT_HPU_LAZY_MODE=0
# yields pure eager with NO graphs at all (the runner's wrap_in_hpu_graph is
# lazy-only), which is far slower than lazy+HPU-graphs. Matches the placement
# used by minimax_m3 / gpt_bigcode: the inner model, not the *ForCausalLM
# wrapper.
# dynamic_arg_dims is explicit because this forward has no type annotations,
# so the decorator's auto-detection finds nothing ("No dynamic dimensions found
# in the forward method"). Dim 0 is the token axis for all three tensor args.
@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0,
    "positions": 0,
    "inputs_embeds": 0,
})
class HpuGlm5NextModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.hc_mult = config.hc_mult

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   quant_config=quant_config,
                                                   prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HpuGlm5NextDecoderLayer(config, vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden = (inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids))
        orig_dim = hidden.dim()
        if orig_dim > 2:
            # HPU warmup/bucketed paths can pass [B, 1, L, H] — flatten to
            # token-major [T, H] (all sublayers are token-major here)
            hidden = hidden.reshape(-1, hidden.size(-1))
        # mHC streams init: replicate embed across hc streams, then final
        # unweighted-mean collapse (Glm5NextTextHyperHead) before the norm
        streams = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        _diag_layer_boundary("initial_streams", streams)
        for layer in self.layers[self.start_layer:self.end_layer]:
            streams = layer(positions=positions, hidden_streams=streams)
        hidden = streams.mean(dim=1)
        # The MTP draft head applies its own hnorm to whatever hidden state it
        # is handed, so in principle it wants the PRE-final-norm residual.
        # Measured, routing the pre-norm residual to the draft (and moving the
        # final norm into compute_logits) did not move acceptance, so the
        # simple layout stays: norm here, both consumers get the post-norm
        # hidden state.
        hidden = self.norm(hidden)
        _diag_layer_boundary("final_norm", hidden)
        return hidden


# ----------------------------------------------------------------------------
# MTP (nextn) layer 45 — weights loaded, execution deferred to Phase 4
# ----------------------------------------------------------------------------


class HpuGlm5NextMTP(nn.Module):

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "mtp"):
        super().__init__()
        # eh_proj merges concatenated [e, h] (2*hidden) -> hidden
        self.eh_proj = ReplicatedLinear(config.hidden_size * 2,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=vllm_config.quant_config,
                                        prefix=f"{prefix}.eh_proj")
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HpuGlm5NextSparseAttention(config, vllm_config, prefix=f"{prefix}.self_attn")
        self.mlp = HpuGlm5NextMoE(config, vllm_config, prefix=f"{prefix}.mlp")
        self.shared_head_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------

_GLM5_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        # MTP layer 45 first (before the generic strip)
        "model.language_model.layers.45.": "mtp.",
        "model.language_model.": "model.",
    },
    orig_to_new_substr={
        "hc_attn_fn": "attn_hc.fn",
        "hc_attn_base": "attn_hc.base",
        "hc_attn_scale": "attn_hc.scale",
        "hc_ffn_fn": "ffn_hc.fn",
        "hc_ffn_base": "ffn_hc.base",
        "hc_ffn_scale": "ffn_hc.scale",
        "mlp.gate.e_score_correction_bias": "mlp.gate_e_score_correction_bias",
        "shared_head.norm": "shared_head_norm",
        # Weights for structures the HPU model does not build (vision tower,
        # sparse-attention indexer, rotaries): mapped to None, which drops them
        # before the loader sees them (vllm#53106 removed the loader's
        # skip_prefixes/skip_substrs kwargs; upstream gaudi migrated gpt_bigcode
        # and starcoder2 the same way, PR #1763).
        "visual": None,
        "vision_tower": None,
        "self_attn.indexer": None,
        "index_kpool_compress": None,
        "rotary": None,
        "self_attn.q_conv1d.weight": "self_attn.q_conv1d_weight",
        "self_attn.k_conv1d.weight": "self_attn.k_conv1d_weight",
        "self_attn.v_conv1d.weight": "self_attn.v_conv1d_weight",
        "self_attn.o_norm.weight": "self_attn.o_norm",
        # NOTE: fp8 blockwise scale stays "weight_scale_inv" — this vLLM
        # registers weight_scale_inv params for block quant (fp8.py:351,482)
        # q_a_proj / kv_a_proj_with_mqa stack onto fused_qkv_a_proj (0/1)
        # via orig_to_new_stacked — no rename needed.
        # (removed) "weight_scale_inv" -> "weight_scale": WRONG for plain fp8
        # block quant on this stack — params are *_weight_scale_inv.
    },
    orig_to_new_stacked={
        "self_attn.q_proj": ("self_attn.qkv_proj", 0),
        "self_attn.k_proj": ("self_attn.qkv_proj", 1),
        "self_attn.v_proj": ("self_attn.qkv_proj", 2),
        "self_attn.q_a_proj": ("self_attn.fused_qkv_a_proj", 0),
        "self_attn.kv_a_proj_with_mqa": ("self_attn.fused_qkv_a_proj", 1),
        "mlp.gate_proj": ("mlp.gate_up_proj", 0),
        "mlp.up_proj": ("mlp.gate_up_proj", 1),
        "mlp.shared_experts.gate_proj": ("mlp.shared_experts.gate_up_proj", 0),
        "mlp.shared_experts.up_proj": ("mlp.shared_experts.gate_up_proj", 1),
    },
)


class HpuGlm5NextForConditionalGeneration(
        nn.Module,
        HasInnerState,
        IsHybrid,
):
    # The KDA prefill (hpu_chunk_kda) and conv (hpu_causal_conv1d_fn) take
    # [B, S] batches with per-sequence padding masks, state indices and init
    # states, so the runner may merge concurrent prompts into one prefill
    # batch (VLLM_PROMPT_BS_BUCKET_MAX > 1); see _can_merge_prefill_contents.
    supports_batched_mamba_prefill = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        self.model = HpuGlm5NextModel(vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model")
        # The MTP block must carry a layer index in its prefix: vLLM's
        # extract_layer_index() asserts a layer name holds exactly one integer,
        # so a bare "mtp" prefix fails with "layer name mtp.self_attn.attn
        # should only contain one integer" once the block is registered for a
        # KV cache. Index num_hidden_layers (45) is free -- the target owns
        # 0..44 -- and matches the checkpoint's own layers.45 naming. The
        # python attribute stays `mtp`, so weight loading is unaffected (the
        # loader walks attribute paths, not this prefix).
        self.mtp_prefix = f"model.layers.{text_config.num_hidden_layers}"
        self.mtp = (HpuGlm5NextMTP(text_config, vllm_config, prefix=self.mtp_prefix) if getattr(
            text_config, "num_nextn_predict_layers", 0) > 0 else None)
        if self.mtp is not None:
            # Whether the MTP block actually executes decides whether it may
            # keep its KV cache. Its attention wrapper registered itself in the
            # static forward context at init.
            _spec = getattr(vllm_config, "speculative_config", None)
            _mtp_runs = (_spec is not None and getattr(_spec, "method", None) == "mtp")
            if not _mtp_runs:
                # Weight-load-only: leave the registration in place and the
                # cache manager would allocate KV for a layer that never
                # executes (and choke on its non-standard prefix). Purge every
                # mtp.* entry from the context.
                static_ctx = vllm_config.compilation_config.static_forward_context
                for ctx_name in [k for k in static_ctx if k.startswith(self.mtp_prefix + ".")]:
                    del static_ctx[ctx_name]
            else:
                # Speculative decode: the block runs every step as the draft
                # head (HpuEagleProposer.load_model binds it), so it needs a
                # real KV cache -- keep the registration.
                logger.warning("[GLM] MTP execution enabled; keeping mtp.* in the static "
                               "forward context so the draft attention gets a KV cache")
        self.lm_head = ParallelLMHead(text_config.vocab_size,
                                      text_config.hidden_size,
                                      quant_config=vllm_config.quant_config,
                                      prefix=f"{prefix}.lm_head" if prefix else "lm_head")
        self.logits_processor = LogitsProcessor(text_config.vocab_size)

    # -- hybrid/mamba interfaces (runner + hybrid cache manager) --------
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple:
        hf = vllm_config.model_config.hf_text_config
        lac = hf.linear_attn_config
        num_spec = (vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0)
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            lac["num_heads"],
            lac["head_dim"],
            conv_kernel_size=lac["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids=input_ids,
                          positions=positions,
                          intermediate_tensors=intermediate_tensors,
                          inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The mapper also drops the vision-tower / indexer / rotary weights the
        # HPU model does not build (mapped to None above), so AutoWeightsLoader
        # never sees them.
        weights = _GLM5_HF_TO_VLLM_MAPPER.apply(weights)
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        # Derived constants must be built AFTER the parameters hold real
        # values, and BEFORE the first forward (which is the HPU-graph
        # capture). See build_decode_constants.
        # Duck-typed rather than isinstance: an isinstance check silently
        # matched nothing on the server path and the constants were never
        # built, leaving every forward on the recompute fallback.
        _n = 0
        for m in self.modules():
            if hasattr(m, "build_decode_constants"):
                m.build_decode_constants()
                _n += 1
        logger.warning("[GLM] load_weights done: built decode constants on %d KDA layers "
                       "(%d modules scanned)", _n, sum(1 for _ in self.modules()))
        return loaded
