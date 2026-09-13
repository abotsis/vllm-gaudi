# SPDX-License-Identifier: Apache-2.0
"""Bounded layer snapshots during an explicitly graph-bypassed synchronous forward."""
import os
import stat
from contextlib import contextmanager

import torch

MAX_BYTES = 64 * 1024 * 1024
_collector = None


def directory(runner):
    if getattr(runner, "_diag_layer_disabled_at_boot", True):
        return None
    path = runner._diag_sampler_directory()
    if path is None:
        return None
    try:
        info = os.lstat(os.path.join(path, "LAYER_ENABLED"))
        if (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600):
            return path
    except OSError:
        pass
    return None


def cpu(tensor):
    return tensor.detach().to(device="cpu", non_blocking=False, copy=True)


def prepare(runner, path, context, tokens, positions, metadata, logits_indices, mark_step, rank, baseline_graphs):
    """Independent admission; validate exact history before installing any collector."""
    ordinal = getattr(runner, "_diag_layer_calls", 0)
    if ordinal >= 9 or getattr(runner, "_diag_layer_bytes", 0) >= MAX_BYTES:
        return None
    request_ids, logits_requests = context
    if runner.use_merged_prefill or tokens.ndim != 2 or positions.shape != tokens.shape:
        raise ValueError("Layer diagnostic requires rectangular unmerged inputs")
    if not metadata.is_prompt and tokens.shape[1] != 1:
        raise ValueError("Layer diagnostic requires NSPEC0 decode")
    admitted = getattr(runner, "_diag_layer_admitted", [])
    seen = getattr(runner, "_diag_layer_seen", set())
    runner._diag_layer_admitted, runner._diag_layer_seen = admitted, seen
    records = []
    for row, rid in enumerate(request_ids):
        if rid not in logits_requests:
            continue
        request = runner.requests[rid]
        position = len(request.output_token_ids)
        if position >= 3 or (rid, position) in seen:
            continue
        if rid not in admitted:
            if position != 0 or len(admitted) >= 3:
                continue
            admitted.append(rid)
        if len(seen) >= 9:
            break
        seen.add((rid, position))
        records.append(
            dict(request_id=rid,
                 row=row,
                 output_position=position,
                 output_token_ids=list(request.output_token_ids),
                 prompt_token_ids=list(request.prompt_token_ids),
                 num_computed_tokens=request.num_computed_tokens,
                 input_batch_index=runner.input_batch.req_id_to_index[rid]))
    if not records:
        return None
    runner._diag_layer_calls = ordinal + 1
    mark_step()
    mapping = cpu(logits_indices).reshape(-1)
    if metadata.is_prompt:
        if len(logits_requests) != mapping.numel() or len(set(logits_requests)) != len(logits_requests):
            raise ValueError("Ambiguous prefill mapping")
    elif mapping.numel() < len(request_ids):
        raise ValueError("Incomplete decode mapping")
    for record in records:
        row = record["row"]
        index = logits_requests.index(record["request_id"]) if metadata.is_prompt else row
        flat = int(mapping[index])
        mapped_row, column = divmod(flat, tokens.shape[1])
        if mapped_row != row:
            raise ValueError("Request-to-token row mismatch")
        token, position = int(cpu(tokens[row, column])), int(cpu(positions[row, column]))
        history = record["output_token_ids"] or record["prompt_token_ids"]
        if position != len(record["prompt_token_ids"]) + record["output_position"] - 1:
            raise ValueError("Absolute position mismatch")
        if not history or token != history[-1]:
            raise ValueError("Exact token history mismatch")
        record.update(flat_logit_index=flat, token=token, absolute_position=position)
    return dict(runner=runner,
                path=path,
                mark_step=mark_step,
                indices=[r["flat_logit_index"] for r in records],
                payload=dict(schema=1,
                             rank=rank,
                             ordinal=ordinal,
                             records=records,
                             input_shape=tuple(tokens.shape),
                             request_ids=list(request_ids),
                             logits_requests=list(logits_requests),
                             phase="prefill" if metadata.is_prompt else "decode",
                             baseline_graph_eligible=bool(baseline_graphs),
                             diagnostic_bypass=True,
                             boundaries=[],
                             truncated=False))


def boundary(name, tensor, *, row_indices=None):
    """No tensor operations unless the one-forward collector is installed."""
    ticket = _collector
    if ticket is None:
        return
    runner = ticket["runner"]
    selected = ticket["indices"] if row_indices is None else row_indices
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] == 0:
        unavailable(name, "Not a nonempty token-leading tensor")
        return
    if not selected or min(selected) < 0 or max(selected) >= tensor.shape[0]:
        unavailable(name, "Selected token index outside leading dimension")
        return
    size = len(selected) * (tensor.numel() // tensor.shape[0]) * tensor.element_size()
    used = getattr(runner, "_diag_layer_bytes", 0)
    if used + size > MAX_BYTES:
        ticket["payload"]["truncated"] = True
        return
    runner._diag_layer_bytes = used + size
    indices = torch.tensor(selected, dtype=torch.long, device=tensor.device)
    owned = tensor.index_select(0, indices).detach().clone()
    ticket["payload"]["boundaries"].append(
        dict(name=name, shape=tuple(tensor.shape), dtype=str(tensor.dtype), rows=owned, row_indices=list(selected)))


def unavailable(name, reason):
    """Record a deliberately uninstrumented boundary without touching tensors."""
    ticket = _collector
    if ticket is None:
        return
    ticket["payload"].setdefault("unavailable_boundaries", []).append(dict(name=name, reason=reason))


def flush(ticket):
    ticket["mark_step"]()
    payload = ticket["payload"]
    for item in payload["boundaries"]:
        item["rows"] = cpu(item["rows"])
    dfd = os.open(ticket["path"], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(dfd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Layer diagnostic directory changed")
        fd = os.open(f"layer-rank{payload['rank']}-{payload['ordinal']:02d}.pt",
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600,
                     dir_fd=dfd)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
    finally:
        os.close(dfd)


def token_boundary(name, tensor):
    """Strict flattened rectangular token contract for attention-only probes."""
    if _collector is None:
        return
    shape = _collector["payload"]["input_shape"]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != shape[0] * shape[1]:
        unavailable(name, "Token-leading dimension differs from rectangular input; prefix/cache layout not mapped")
        return
    boundary(name, tensor)


def attention_boundary(owner, name, tensor):
    if _collector is None or owner is not _collector.get("attention_impl"):
        return
    token_boundary(_collector["attention_prefix"] + "." + name, tensor)


def attention_metadata(owner, metadata):
    """Only metadata with a proven token or rectangular request axis is selected."""
    if _collector is None or owner is not _collector.get("attention_impl"):
        return
    prefix = _collector["attention_prefix"]
    _collector["payload"].setdefault("attention_metadata", []).append(
        dict(name=prefix, is_prompt=bool(metadata.is_prompt), block_size=metadata.block_size))
    slots = metadata.slot_mapping
    shape = _collector["payload"]["input_shape"]
    if isinstance(slots, torch.Tensor) and tuple(slots.shape) == tuple(shape):
        token_boundary(prefix + ".slot_mapping", slots.flatten())
    else:
        unavailable(prefix + ".slot_mapping", "Expected rectangular token slot mapping")
    lengths = metadata.seq_lens_tensor
    if isinstance(lengths, torch.Tensor) and tuple(lengths.shape) == (shape[0], ):
        rows = [index // shape[1] for index in _collector["indices"]]
        boundary(prefix + ".seq_lens", lengths, row_indices=rows)
    else:
        unavailable(prefix + ".seq_lens", "No proven rectangular request length mapping")
    for field in ("block_list", "block_mapping", "block_groups", "attn_bias"):
        unavailable(prefix + "." + field, "Block-space/bias request mapping not proven; not copied")
    unavailable(prefix + ".cache_history", "Historical block-to-request mapping not proven; no KV pool copied")


def cache_written(module, inputs, output):
    """Plain VLLMKVCache writes index_copy_(0, slots, input); no quantized-layout assumptions."""
    if _collector is None:
        return
    prefix = _collector["attention_prefix"] + ".cache_written_rows"
    if type(module).__module__ != "vllm_gaudi.extension.utils" or type(module).__name__ != "VLLMKVCache":
        unavailable(prefix, "Quantized/wrapped cache operation: physical layout not proven")
        return
    if len(inputs) != 3:
        unavailable(prefix, "Cache call signature not proven")
        return
    value, cache, slots = inputs
    shape = _collector["payload"]["input_shape"]
    count = shape[0] * shape[1]
    if (not isinstance(slots, torch.Tensor) or slots.ndim != 1 or slots.numel() != count
            or slots.dtype not in (torch.int32, torch.int64) or value.ndim != 2 or value.shape[0] != count
            or cache.ndim != 2 or cache.shape[1:] != value.shape[1:]):
        unavailable(prefix, "Cache/slot token dimensions not proven")
        return
    # Copy only selected physical addresses to host; never the slot table or KV pool.
    index = torch.tensor(_collector["indices"], device=slots.device, dtype=torch.long)
    addresses = slots.index_select(0, index)
    _collector["mark_step"]()
    addresses = cpu(addresses).tolist()
    boundary(prefix, cache, row_indices=addresses)


# -- KDA (linear attention) layer-0 interior ----------------------------------
# GLM-5.3-Flash layer 0 is a KDA layer, so the MLA hooks above never fire for it.
# The seed of the serial-vs-co-batched delta sits inside this block; these
# capture its intermediates with the same owner-identity gate as the MLA path.
KDA_MODULES = ("qkv_proj", "f_a_proj", "f_b_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj")


def kda_boundary(owner, name, tensor):
    """Token-leading KDA intermediates of the selected layer-0 instance only."""
    if _collector is None or owner is not _collector.get("kda_owner"):
        return
    token_boundary(_collector["kda_prefix"] + "." + name, tensor)


def _sequence_rows():
    shape = _collector["payload"]["input_shape"]
    return sorted({index // shape[1] for index in _collector["indices"]})


def kda_sequence_boundary(owner, name, tensor):
    """Per-sequence KDA tensors: leading dim is the rectangular batch axis, not tokens."""
    if _collector is None or owner is not _collector.get("kda_owner"):
        return
    prefix = _collector["kda_prefix"] + "." + name
    shape = _collector["payload"]["input_shape"]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != shape[0]:
        unavailable(prefix, "Leading dimension is not the rectangular batch axis")
        return
    boundary(prefix, tensor, row_indices=_sequence_rows())


def kda_pool_rows(owner, name, pool, slots):
    """Selected sequences' physical pool rows; ``slots`` is the 1-D per-lane slot tensor."""
    if _collector is None or owner is not _collector.get("kda_owner"):
        return
    prefix = _collector["kda_prefix"] + "." + name
    lanes = _sequence_rows()
    if (not isinstance(slots, torch.Tensor) or slots.ndim != 1 or not isinstance(pool, torch.Tensor) or pool.ndim < 2
            or max(lanes) >= slots.numel()):
        unavailable(prefix, "Lane-to-slot mapping not proven")
        return
    index = torch.tensor(lanes, device=slots.device, dtype=torch.long)
    _collector["mark_step"]()
    addresses = cpu(slots.index_select(0, index)).tolist()
    if min(addresses) < 0 or max(addresses) >= pool.shape[0]:
        unavailable(prefix, "Selected lane maps to a pad or out-of-range slot")
        return
    boundary(prefix, pool, row_indices=addresses)


@contextmanager
def kda_hooks(layer):
    """Projection outputs of the actual layer-0 KDA instance, inside the selected forward."""
    ticket = _collector
    attn = getattr(layer, "self_attn", None)
    prefix = layer._prefix + ".kda"
    if (ticket is None or attn is None or type(attn).__name__ != "HpuGlm5NextKdaAttention"
            or any(not isinstance(getattr(attn, name, None), torch.nn.Module) for name in KDA_MODULES)):
        unavailable(prefix, "Exact layer-0 KDA module identity unavailable")
        yield
        return
    handles = []
    called = set()
    ticket["kda_owner"] = attn
    ticket["kda_prefix"] = prefix
    ticket["payload"]["attention_modules"] = dict(kind="kda",
                                                  owner=layer._prefix + ".self_attn",
                                                  owner_type=type(attn).__name__)

    def capture(name, value):
        called.add(name)
        if isinstance(value, tuple):
            value = value[0] if value else None
        token_boundary(prefix + "." + name, value)

    def output_hook(name):

        def hook(module, inputs, output):
            capture(name, output)

        return hook

    def input_hook(module, inputs):
        capture("pre_o_proj", inputs[0] if inputs else None)

    try:
        for name in KDA_MODULES:
            handles.append(getattr(attn, name).register_forward_hook(output_hook(name)))
        handles.append(attn.o_proj.register_forward_pre_hook(input_hook))
        yield
        for name in sorted((set(KDA_MODULES) | {"pre_o_proj"}) - called):
            unavailable(prefix + "." + name, "Module not called on this path")
    finally:
        for handle in reversed(handles):
            handle.remove()
        ticket.pop("kda_owner", None)
        ticket.pop("kda_prefix", None)


@contextmanager
def attention_hooks(layer):
    """Install on the actual layer-0 instances only, inside the selected forward."""
    ticket = _collector
    if ticket is None or not layer._diag_split_layer0:
        yield
        return
    if layer._is_linear:
        with kda_hooks(layer):
            yield
        return
    prefix = layer._prefix + ".attention"
    wrapper = getattr(layer.self_attn, "mla_attn", None)
    inner = getattr(wrapper, "mla_attn", None)
    if (layer._is_linear or wrapper is None or inner is None
            or getattr(wrapper, "prefix", None) != layer._prefix + ".self_attn"
            or getattr(inner, "layer_name", None) != wrapper.prefix + ".attn"):
        unavailable(prefix, "Exact layer-0 MLA module identity unavailable")
        yield
        return
    handles = []
    called = set()
    expected = set()
    ticket["attention_impl"] = inner.impl
    ticket["attention_prefix"] = prefix
    ticket["payload"]["attention_modules"] = dict(wrapper=wrapper.prefix,
                                                  inner=inner.layer_name,
                                                  wrapper_type=type(wrapper).__name__,
                                                  inner_type=type(inner).__name__,
                                                  impl_type=type(inner.impl).__name__)

    def capture(name, value):
        called.add(name)
        # Linear returns (activation, deferred bias). Bias is NOT token-leading.
        if isinstance(value, tuple):
            value = value[0] if value else None
        token_boundary(prefix + "." + name, value)

    def output_hook(name):

        def hook(module, inputs, output):
            capture(name, output)

        return hook

    def input_hook(module, inputs):
        capture("pre_o_proj", inputs[0] if inputs else None)

    try:
        for name in ("fused_qkv_a_proj", "kv_a_proj_with_mqa", "q_a_layernorm", "kv_a_layernorm", "q_b_proj", "q_proj",
                     "kv_b_proj", "g_proj", "o_proj"):
            module = getattr(wrapper, name, None)
            if module is not None:
                expected.add(name)
                handles.append(module.register_forward_hook(output_hook(name)))
        expected.add("pre_o_proj")
        handles.append(wrapper.o_proj.register_forward_pre_hook(input_hook))
        handles.append(inner.impl.latent_cache_k.register_forward_hook(cache_written))
        yield
        for name in sorted(expected - called):
            unavailable(prefix + "." + name, "Module not called on this path (decode uses absorbed KV weights)")
    finally:
        for handle in reversed(handles):
            handle.remove()
        ticket.pop("attention_impl", None)
        ticket.pop("attention_prefix", None)


@contextmanager
def collecting(ticket):
    global _collector
    if _collector is not None:
        raise RuntimeError("Nested layer diagnostic forward")
    _collector = ticket
    try:
        yield
        _collector = None
        flush(ticket)
    finally:
        _collector = None
        ticket["payload"]["boundaries"].clear()
