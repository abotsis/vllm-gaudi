# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash MTP (nextn layer 45) draft head for HPU speculative decode.

Structure follows the checkpoint exactly (38 distinct module paths under
``model.language_model.layers.45.``): enorm / hnorm / eh_proj, a plain
pre-norm block (input_layernorm -> MLA self_attn -> post_attention_layernorm
-> 288-expert MoE), and ``shared_head.norm``. Note what is NOT there:

  * no hyper-connections -- unlike every one of the 45 target decoder layers,
    which run attn_hc/ffn_hc. The MTP block is a plain residual block.
  * no ``shared_head.head`` -- the head is SHARED with the target's lm_head,
    so the draft cannot own a private copy.

The forward matches the reference MTP formulation (see upstream
``glm4_moe_mtp.py:105-127``, the same DeepSeek-derived design):

    h = eh_proj(cat[ enorm(embed(tok)), hnorm(target_hidden) ])

with position 0 embeddings masked to zero (nothing precedes them to predict
from).

**Weight sharing.** This module owns no parameters. The target
``HpuGlm5NextForConditionalGeneration`` already builds and loads layer 45 into
``self.mtp`` on every run today (6.98 GiB of the checkpoint, 0.87 GiB/rank at
TP8), and it owns the embedding and lm_head the draft must share anyway.
Loading a second copy the usual way would mean a second full pass over a
306 GiB checkpoint at every boot plus duplicate device memory, for weights
already resident. So this wraps the target's modules by reference; see
``HpuEagleProposer.load_model``.
"""

import os

import contextlib
import torch
import torch.nn as nn

# The draft's self-attention is bypassed by default: measured, it corrupts
# the draft's own predictions rather than helping (see forward()). Set
# VLLM_GLM_MTP_DRAFT_ATTN=1 to restore it.
_DRAFT_ATTN = os.environ.get("VLLM_GLM_MTP_DRAFT_ATTN", "0") == "1"

# Diagnostic: scale the draft's attention contribution. Re-read per forward so a
# sweep costs one request instead of a 12-minute reboot; only meaningful with the
# draft UNGRAPHED, since graph capture traces the branch once and replays it.
# (The draft is never HPU-graphed -- see the _shares_target_modules branch in
# hpu_model_runner's load_model.)
#
# Non-negative values scale the attention output. Two negative sentinels turn the
# file into an in-process A/B for the position-0 acceptance gap, which previously
# could only be compared across separate boots and therefore across ~6% of
# restart-to-restart variance:
#
#   -1  BYPASS   -- do not call self_attn at all. This is the control: it should
#                   reproduce the shipped VLLM_GLM_MTP_DRAFT_ATTN=0 default.
#   -2  PADWRITE -- call self_attn and discard its output (as scale 0 does), but
#                   redirect its K/V writes to the pad slot. Scale 0 is
#                   arithmetically identical to BYPASS yet scores much lower, so
#                   the damage is the call's side effect rather than its output;
#                   this separates "the KV write" from "anything else the call
#                   touches". If PADWRITE recovers and 0 does not, it is the write.
_ATTN_SCALE_FILE = os.environ.get("VLLM_GLM_MTP_ATTN_SCALE_FILE", "")
_BYPASS = -1.0
_PADWRITE = -2.0
# The runner's padded-lane slot. Writes land here instead of at a real position.
_PAD_SLOT_ID = -1
_attn_scale_cache = {"mtime": None, "value": 1.0}


def _attn_scale() -> float:
    if not _ATTN_SCALE_FILE:
        return 1.0
    try:
        st = os.stat(_ATTN_SCALE_FILE)
    except OSError:
        return 1.0
    if _attn_scale_cache["mtime"] != st.st_mtime:
        try:
            with open(_ATTN_SCALE_FILE) as f:
                _attn_scale_cache["value"] = float(f.read().strip() or 1.0)
        except Exception:
            _attn_scale_cache["value"] = 1.0
        _attn_scale_cache["mtime"] = st.st_mtime
    return _attn_scale_cache["value"]


def _pad_slot_metadata(attn_metadata):
    """Copy of ``attn_metadata`` whose slot_mapping points only at the pad slot.

    Used by the PADWRITE probe: the attention call runs unchanged but its
    reshape_and_cache lands somewhere no draft step will ever read back.
    """
    import dataclasses
    sm = getattr(attn_metadata, "slot_mapping", None)
    if sm is None or not dataclasses.is_dataclass(attn_metadata):
        return attn_metadata
    return dataclasses.replace(attn_metadata, slot_mapping=torch.full_like(sm, _PAD_SLOT_ID))


# Graphed draft: the draft's whole tensor path (embed -> eh_proj -> norms ->
# MoE -> head norm) replays as ONE HPU graph per bucket shape instead of running
# eager. Eager, each draft forward is hundreds of host launches, and on this
# model the host launch path is the bottleneck (93 ms host vs 17 ms device per
# step) -- MTP-4 pays that four times per step. Set VLLM_GLM_MTP_DRAFT_GRAPH=0
# to fall back to the eager draft (same numerics, no graph).
_DRAFT_GRAPH = os.environ.get("VLLM_GLM_MTP_DRAFT_GRAPH", "1") == "1"
# Experimental local opt-in: graph only the tensor work around eager attention.
_SPLIT_GRAPH = os.environ.get("VLLM_GLM_MTP_SPLIT_GRAPH", "0") == "1"


def _private_copy(module: nn.Module) -> nn.Module:
    """Shallow-copy a module tree with every parameter/buffer storage cloned.

    Plain-attribute tensors (the fp8 path stores processed weight VIEWS as
    ordinary attributes after load) are cloned too, so nothing in the copy
    aliases the original's device memory. Tensors that are views of a
    parameter in the same module get re-derived where possible; otherwise
    they are cloned independently (correct, marginally more memory).
    """
    seen: dict[int, torch.Tensor] = {}

    def _clone(t: torch.Tensor) -> torch.Tensor:
        key = t.data_ptr()
        if key in seen and seen[key].shape == t.shape and seen[key].stride() == t.stride():
            return seen[key]
        c = t.detach().clone()
        seen[key] = c
        return c

    def _copy_module(m: nn.Module) -> nn.Module:
        # object.__new__ + explicit __dict__ transplant: copy.copy() routes
        # through Module.__getattr__ for attributes the HPU bridge adds
        # lazily, and dies on the first one that does not exist yet.
        new = object.__new__(type(m))
        new.__dict__ = dict(m.__dict__)
        # The HPU bridge installs naming hooks on every module at attach time
        # (torch_overwrites: custom_name + pre/post forward hooks, guarded by a
        # names_hook flag). A dict transplant carries the FLAG and the hook
        # registries but not a working custom_name binding, so the bridge
        # skips re-registration and the forward hook then fails. Strip that
        # state; attaching the copy to its parent re-registers it cleanly.
        for k in ("names_hook", "custom_name"):
            new.__dict__.pop(k, None)
        new._forward_pre_hooks = type(m._forward_pre_hooks)()
        new._forward_hooks = type(m._forward_hooks)()
        new._forward_hooks_with_kwargs = type(getattr(m, "_forward_hooks_with_kwargs", {}))()
        new._forward_hooks_always_called = type(getattr(m, "_forward_hooks_always_called", {}))()
        new._forward_pre_hooks_with_kwargs = type(getattr(m, "_forward_pre_hooks_with_kwargs", {}))()
        new._parameters = {}
        new._buffers = {}
        new._modules = {}
        for name, prm in m._parameters.items():
            if prm is None:
                new._parameters[name] = None
                continue
            c = _clone(prm.data)
            np_ = nn.Parameter(c, requires_grad=False)
            # carry vLLM parameter metadata attributes over (loader-only, but
            # some forward paths read e.g. .weight_loader-adjacent flags)
            for k, v in vars(prm).items():
                if not k.startswith("_") and not hasattr(np_, k):
                    with contextlib.suppress(Exception):
                        setattr(np_, k, v)
            new._parameters[name] = np_
        for name, buf in m._buffers.items():
            new._buffers[name] = None if buf is None else _clone(buf)
        for name, child in m._modules.items():
            new._modules[name] = None if child is None else _copy_module(child)
        # plain-attribute tensors (post-load fp8 views live here)
        for k, v in list(vars(m).items()):
            if k.startswith("_") or k in ("training", ):
                continue
            if torch.is_tensor(v):
                setattr(new, k, _clone(v))
        return new

    return _copy_module(module)


class _DraftGraphCore(nn.Module):
    """The pure-tensor draft path, built to be captured as an HPU graph.

    Owns PRIVATE copies of every weight it touches, cloned on device from the
    target's already-loaded modules (no checkpoint re-read, ~0.9 GiB/rank):
    capturing the target's storage in a second graph is what fails at replay
    ("Neither storage attached to input tensor, not its view"), and the copies
    make the two graphs independent by construction rather than by trust.

    Everything shape-dynamic or Python-dynamic stays OUTSIDE: forward context,
    attention (bypassed on the shipped default, so no metadata enters here),
    the diagnostic scale file, and the argmax -- which stays with the proposer
    so the graph's output contract is just a hidden-state tensor.
    """

    def __init__(self, mtp: nn.Module, embed_tokens: nn.Module):
        super().__init__()
        # Clone STORAGE, not objects. deepcopy fails on vLLM's parameter
        # subclasses (custom __new__), and rebuilding from constructors would
        # miss fp8 process_weights_after_loading's rewritten layouts/views. So:
        # shallow-copy the module tree (every class, attribute and view
        # relationship preserved) and replace each tensor's storage with a
        # private clone -- exactly what the graph needs and nothing else.
        self.embed_tokens = _private_copy(embed_tokens)
        self.enorm = _private_copy(mtp.enorm)
        self.hnorm = _private_copy(mtp.hnorm)
        self.eh_proj = _private_copy(mtp.eh_proj)
        self.input_layernorm = _private_copy(mtp.input_layernorm)
        self.post_attention_layernorm = _private_copy(mtp.post_attention_layernorm)
        # The MoE is SHARED by reference, not cloned. Synapse's fused-MoE op
        # registers expert weights in a per-recipe metadata DB keyed on storage
        # identity; a cloned MoE inside a second graph failed to compile with
        # "MOE multiplexer weights were partially registered" (the op's
        # registration and the clone's storage disagreed). The target's graph
        # only READS these weights, and so does this one, so sharing is
        # semantically safe; the graph-capture double-registration question is
        # answered empirically by this boot rather than assumed either way.
        self.mlp = mtp.mlp
        self.shared_head_norm = _private_copy(mtp.shared_head_norm)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        x, _ = self.eh_proj(torch.cat([self.enorm(inputs_embeds), self.hnorm(hidden_states)], dim=-1))
        residual = x
        # input_layernorm feeds only the (bypassed) attention; skipped here.
        x = self.post_attention_layernorm(residual)
        x = self.mlp(x)
        return self.shared_head_norm(residual + x)


class _DraftPreAttentionGraphCore(nn.Module):
    """Private light weights; attention and its cache remain outside capture."""

    def __init__(self, mtp: nn.Module, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = _private_copy(embed_tokens)
        self.enorm = _private_copy(mtp.enorm)
        self.hnorm = _private_copy(mtp.hnorm)
        self.eh_proj = _private_copy(mtp.eh_proj)
        self.input_layernorm = _private_copy(mtp.input_layernorm)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed_tokens(input_ids)
        embedding = torch.where(positions.unsqueeze(-1) == 0, 0, embedding)
        residual, _ = self.eh_proj(torch.cat([self.enorm(embedding), self.hnorm(hidden_states)], dim=-1))
        return residual, self.input_layernorm(residual)


class _DraftPostAttentionGraphCore(nn.Module):
    """Private norms with the same shared MoE ownership as the bypass core."""

    def __init__(self, mtp: nn.Module):
        super().__init__()
        self.post_attention_layernorm = _private_copy(mtp.post_attention_layernorm)
        self.shared_head_norm = _private_copy(mtp.shared_head_norm)
        self.mlp = mtp.mlp

    def forward(self, residual: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        states = residual + attention
        return self.shared_head_norm(states + self.mlp(self.post_attention_layernorm(states)))


class HpuGlm5NextMTPModel(nn.Module):
    """Draft head over the target's already-loaded MTP (layer 45) modules."""

    def __init__(self, target: nn.Module):
        super().__init__()
        mtp = getattr(target, "mtp", None)
        if mtp is None:
            raise ValueError("target model has no .mtp module -- GLM-5.3 MTP speculative decode needs "
                             "num_nextn_predict_layers > 0 in the model config and the layer-45 weights "
                             "present in the checkpoint.")
        # Marks that every parameter below is the TARGET's. The runner must not
        # wrap this in a second HPU graph: the target model is already wrapped,
        # and capturing the same storage in two graphs makes replay fail with
        # "Neither storage attached to input tensor, not its view".
        self._shares_target_modules = True
        self.requires_prompt_cache_fill = _DRAFT_ATTN and not bool(_ATTN_SCALE_FILE)
        # Held by reference on purpose: no parameters are copied or re-read.
        self.mtp = mtp
        self._target = target
        self.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self.logits_processor = target.logits_processor
        # Cores are built from the target's already-loaded, processed weights.
        # Attention itself always uses the original modules and forward context.
        self.pre_attention_graph_core = None
        self.post_attention_graph_core = None
        if _SPLIT_GRAPH and _DRAFT_GRAPH and _DRAFT_ATTN:
            import habana_frameworks.torch as htorch
            self.pre_attention_graph_core = htorch.hpu.wrap_in_hpu_graph(_DraftPreAttentionGraphCore(
                mtp, self.embed_tokens),
                                                                         disable_tensor_cache=True,
                                                                         dry_run=False)
            self.post_attention_graph_core = htorch.hpu.wrap_in_hpu_graph(_DraftPostAttentionGraphCore(mtp),
                                                                          disable_tensor_cache=True,
                                                                          dry_run=False)
        self.graph_core = None
        if _DRAFT_GRAPH and not _DRAFT_ATTN:
            import habana_frameworks.torch as htorch
            core = _DraftGraphCore(mtp, target.model.embed_tokens)
            # dry_run=False is load-bearing. wrap_in_hpu_graph enables dry-run
            # capture by default when disable_tensor_cache=True, and dry run
            # "does not evaluate graphs while capturing ... may cause accuracy
            # issues if the input shapes of the current graph depend on the
            # outputs of previous graphs" (its own docstring). The draft chain
            # is exactly that: each replay consumes the previous replay's
            # output. Measured: with dry run, the first true replay after a
            # capture returned deterministic garbage (max|d| 3.766 vs eager,
            # argmax agreement 0.000, identical across two boots) while every
            # other call was bit-exact. The target model is unaffected only
            # because its inputs are always fresh runner buffers.
            self.graph_core = htorch.hpu.wrap_in_hpu_graph(core, disable_tensor_cache=True, dry_run=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step. Returns hidden states for the shared head.

        This head is deliberately NOT wrapped in HpuModelAdapter/HPU graphs
        (it shares the target's already-captured weights), so unlike every
        other model here it has to install its own forward context -- the
        proposer hands ``attn_metadata`` straight to us and the MLA layer
        below reads it from the context.
        """
        if self.graph_core is not None and hidden_states.shape[1] == 1:
            # Graph replay -- DECODE-shaped calls only ([bs, 1, H]): those are
            # the 4-per-step draft forwards where all the launches are, and
            # they bucket to a handful of shapes. The prefill-shaped call
            # ([1, prompt_len, H], once per request) would capture a fresh
            # recipe per distinct prompt length -- compile cost per novel
            # length plus the first-execution hazard each time; measured as
            # code-corpus p1 0.75 vs 0.94 eager. It stays on the eager path.
            # No forward context: nothing inside the core reads it.
            # Materialise the inputs before EVERY replay. An HPU-graph replay
            # is opaque to lazy mode's dependency tracking: it launches against
            # its input buffers as they are, and does not wait for pending
            # lazy ops that write them. Draft positions 1..3 feed each replay
            # the previous position's output after eager ops (compute_logits,
            # argmax, the proposer's bookkeeping) -- exactly such pending
            # writes. Measured with an in-engine parity probe: the first replay
            # consuming a chain-produced input diverged from eager to the
            # digit (max|d| 3.766, argmax agreement 0.000, identical across
            # A mark_step here drains those writes first.
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()
            # mark_step drains the LAZY queue; the device may still be
            # executing what it drained. A partial recovery (code p1 0.74 ->
            # 0.84, not the 0.94 eager reference) says the replay still races
            # device-side completion of its input producers. Full sync.
            torch.hpu.synchronize()
            out = self.graph_core(input_ids, positions, hidden_states)
            # Materialise the replay output BEFORE the eager consumers
            # (compute_logits' TP-gathered lm_head, argmax) touch it. Without
            # this, the eager ops fused onto the replay's memory and emitted
            # out-of-range draft ids: code acceptance fell 0.94 -> 0.75 and a
            # logprobs request carried a junk id into the tokenizer
            # (OverflowError) -- the same lazy-mode genre as the sampler
            # bugs, at the graph/eager seam this time.
            htcore.mark_step()
            return out
        use_split = (self.pre_attention_graph_core is not None and self.post_attention_graph_core is not None
                     and hidden_states.ndim == 3 and hidden_states.shape[1] == 1 and inputs_embeds is None
                     and not _ATTN_SCALE_FILE)
        forward_impl = self._forward_split if use_split else self._forward
        if attn_metadata is not None:
            if _DRAFT_ATTN and _ATTN_SCALE_FILE and _attn_scale() == _PADWRITE:
                attn_metadata = _pad_slot_metadata(attn_metadata)
            from vllm.config import get_current_vllm_config
            from vllm.forward_context import set_forward_context
            with set_forward_context(attn_metadata, get_current_vllm_config()):
                return forward_impl(input_ids, positions, hidden_states, inputs_embeds)
        return forward_impl(input_ids, positions, hidden_states, inputs_embeds)

    def _forward_split(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import habana_frameworks.torch.core as htcore

        batch = hidden_states.shape[0]
        if input_ids.numel() != batch or positions.numel() != batch:
            raise ValueError("MTP split graph requires one input id and position per decode lane.")
        input_ids = input_ids.reshape(batch, 1)
        positions = positions.reshape(batch, 1)
        htcore.mark_step()
        torch.hpu.synchronize()
        residual, attention_input = self.pre_attention_graph_core(input_ids, positions, hidden_states)
        htcore.mark_step()
        torch.hpu.synchronize()
        attention = self.mtp.self_attn(positions=positions, hidden_states=attention_input)
        if attention.numel() != residual.numel():
            raise ValueError("MTP attention output must preserve the input token and hidden dimensions.")
        attention = attention.reshape_as(residual)
        htcore.mark_step()
        torch.hpu.synchronize()
        output = self.post_attention_graph_core(residual, attention)
        htcore.mark_step()
        torch.hpu.synchronize()
        return output

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Position 0 has no predecessor to predict from; zero it rather than
        # feed the raw embedding (matches glm4_moe_mtp.py:115).
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        x, _ = self.mtp.eh_proj(torch.cat([self.mtp.enorm(inputs_embeds), self.mtp.hnorm(hidden_states)], dim=-1))

        # Plain pre-norm residual block -- no hyper-connections here.
        # (KV aliasing with the target was checked across all 45 layers and
        # refuted -- see GLM53_MTP_HANDOFF.md 5.2; the probe is gone.)
        residual = x
        x = self.mtp.input_layernorm(x)
        # v5: the draft consumes token t to predict t+1; test whether its RoPE
        # should sit at t+1 rather than at t.
        if not _DRAFT_ATTN:
            # ---------------------------------------------------------------
            # The draft's self-attention is BYPASSED by default. Measured, on
            # identical prompts (per-position acceptance off /metrics):
            #
            #   attention on   verbatim 0.62  code 0.51  prose 0.50
            #   attention off  verbatim 0.98  code 0.94  prose 0.65
            #
            # It is not merely useless, it is actively harmful -- attending over
            # a KV path that yields garbage rather than nothing, which is why
            # removing it helps instead of being neutral. The underlying bug is
            # in the draft's KV cache/slot wiring and is NOT fixed here; this
            # skips the corrupted contribution.
            #
            # Safe by construction: draft quality affects THROUGHPUT ONLY.
            # Rejection sampling emits the target's tokens whatever the draft
            # proposes, so a weaker draft can never change the output.
            #
            # Cost of the workaround: later draft positions cannot see the
            # tokens drafted before them, so the chain stays weak
            # (0.82/0.32/0.15/0.04 against CUDA's 0.92/0.75/0.59/0.44). Fixing
            # the KV wiring is what would recover positions 1-3, and prose,
            # which needs attention context most (0.65 vs CUDA's 0.92).
            #
            # Set VLLM_GLM_MTP_DRAFT_ATTN=1 to restore attention.
            # ---------------------------------------------------------------
            pass
        else:
            a = _attn_scale()
            if a == _BYPASS:
                # Same code path as the DRAFT_ATTN=0 branch above, but reachable
                # without a reboot, so it can be compared against a scale of 0 in
                # the same process rather than across two boots.
                pass
            else:
                x = self.mtp.self_attn(positions=positions, hidden_states=x)
                if x.numel() != residual.numel():
                    raise ValueError("MTP attention output must preserve the input token and hidden dimensions.")
                # MLA flattens token axes; restore them before the residual add.
                x = x.reshape_as(residual)
                # PADWRITE discards the output exactly as scale 0 does; only the
                # write destination differs, which is the variable under test.
                scale = 0.0 if a == _PADWRITE else a
                residual = residual + (x if scale == 1.0 else x * scale)

        x = self.mtp.post_attention_layernorm(residual)
        x = self.mtp.mlp(x)
        residual = residual + x

        return self.mtp.shared_head_norm(residual)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, None)

    def load_weights(self, weights):
        # Nothing to load: every parameter is the target's, already populated.
        return set()


class HpuGlm5NextMTPForCausalLM(nn.Module):
    """Registration stand-in for the ``Glm5NextMTPModel`` architecture.

    ``SpeculativeConfig`` builds a draft ``ModelConfig`` and resolves its
    ``architectures`` through the model registry, so the name has to exist.
    But nothing constructs this: GLM-5.3's MTP head is bound directly to the
    target's already-loaded layer-45 modules in
    ``HpuEagleProposer.load_model``, which never calls ``get_model()``. If this
    ever *is* constructed, something has fallen back to the generic draft
    loader and would silently re-read a 306 GiB checkpoint -- so fail loudly.
    """

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs):
        raise NotImplementedError(
            "The Glm5NextMTPModel arch is not constructed directly -- the GLM-5.3 MTP draft head is "
            "bound to the target model's layer-45 modules by "
            "HpuEagleProposer.load_model(). Reaching this means the generic draft-model "
            "loader ran instead, which would re-read the whole checkpoint.")
