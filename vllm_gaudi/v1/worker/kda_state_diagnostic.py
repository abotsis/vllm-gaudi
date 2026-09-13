# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in incoming RAW KDA state snapshots (never effective masked state).

No plugin imports: this module is also exercised by CPU-only source tests.
"""
import os
import stat

import torch

FIELDS = (
    "load_indices_tensor",
    "store_indices_tensor",
    "query_start_loc_p",
    "query_start_loc",
    "has_initial_states_p",
    "padding_mask_flat",
    "num_accepted_tokens",
    "seq_lens_tensor",
    "context_lens_tensor",
    "prep_initial_states",
    "last_chunk_indices_p",
    "blocks_caching_range",
    "mamba_chunks_to_block_mapping",
    "seqlens_offsets_for_blocks",
)
MAX_BYTES = 256 * 1024 * 1024


def state_directory(runner):
    if getattr(runner, "_diag_state_disabled_at_boot", True):
        return None
    directory = runner._diag_sampler_directory()
    if directory is None:
        return None
    try:
        info = os.lstat(os.path.join(directory, "STATE_ENABLED"))
        if (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600):
            return directory
    except OSError:
        pass
    return None


def resolve(indices, group):
    # Exactly HpuGlm5NextKdaAttention._resolve_state_indices; absolute group.
    if indices is not None and indices.dim() > 1:
        assert group is not None
        indices = indices.index_select(0, group.view(1)).squeeze(0).contiguous()
    return indices


def capture(runner, directory, context, tokens, positions, metadata, logits_indices, mark_step, rank):
    """Called before the sole forward, after trimming; exceptions handled by caller."""
    ordinal = getattr(runner, "_diag_state_calls", 0)
    if ordinal >= 9:
        return
    request_ids, logits_requests = context
    if runner.use_merged_prefill or tokens.ndim != 2 or positions.shape != tokens.shape:
        raise ValueError("State diagnostic requires rectangular, unmerged inputs")
    if not metadata.is_prompt and tokens.shape[1] != 1:
        raise ValueError("State diagnostic requires NSPEC0 single-token decode")
    admitted = getattr(runner, "_diag_state_admitted", [])
    seen = getattr(runner, "_diag_state_seen", set())
    runner._diag_state_admitted = admitted
    runner._diag_state_seen = seen
    records = []
    for row, rid in enumerate(request_ids):
        if rid not in logits_requests:
            continue
        request = runner.requests[rid]
        history = request.output_token_ids
        output_position = len(history)
        if output_position >= 3 or (rid, output_position) in seen:
            continue
        if rid not in admitted:
            if output_position != 0 or len(admitted) >= 3:
                continue
            admitted.append(rid)
        if len(seen) >= 9:
            break
        seen.add((rid, output_position))
        records.append(
            dict(request_id=rid,
                 row=row,
                 output_position=output_position,
                 output_token_ids=list(history),
                 prompt_token_ids=list(request.prompt_token_ids),
                 num_computed_tokens=request.num_computed_tokens,
                 input_batch_index=runner.input_batch.req_id_to_index[rid]))
    if not records:
        return
    # Charge selected capture attempts, not later unselected decode steps.
    # Failed captures are not refunded.
    runner._diag_state_calls = ordinal + 1
    modules = [(name, module) for name, module in runner.model.named_modules()
               if type(module).__name__ == "HpuGlm5NextKdaAttention"
               and type(module).__module__ == "vllm_gaudi.models.glm5_next"]
    if not modules:
        raise ValueError("No actual KDA modules found")
    raw_metadata = {field: getattr(metadata, field, None) for field in FIELDS}
    tensors = [value for value in raw_metadata.values() if isinstance(value, torch.Tensor)]
    # Charge all persisted tensor payload before any device work (including maps).
    size = sum(t.numel() * t.element_size() for t in tensors)
    size += logits_indices.numel() * logits_indices.element_size()
    size += len(records) * (tokens.element_size() + positions.element_size())
    for _, module in modules:
        if len(module.kv_cache) != 2:
            raise ValueError("Unexpected KDA pool layout")
        size += module.cache_group_idx.numel() * module.cache_group_idx.element_size()
        for pool in module.kv_cache[:2]:
            size += len(records) * (pool.numel() // pool.shape[0]) * pool.element_size()
    used = getattr(runner, "_diag_state_bytes", 0)
    if used + size > MAX_BYTES:
        return
    runner._diag_state_bytes = used + size

    def cpu(tensor):
        return tensor.detach().to(device="cpu", non_blocking=False, copy=True)

    mark_step()
    maps = {key: cpu(value) if isinstance(value, torch.Tensor) else value for key, value in raw_metadata.items()}
    logit_map = cpu(logits_indices).reshape(-1)
    if metadata.is_prompt:
        if len(logits_requests) != logit_map.numel() or len(set(logits_requests)) != len(logits_requests):
            raise ValueError("Ambiguous prefill request-to-logit mapping")
    elif logit_map.numel() < len(request_ids):
        raise ValueError("Incomplete decode logits mapping")
    for record in records:
        row = record["row"]
        index = (logits_requests.index(record["request_id"]) if metadata.is_prompt else row)
        flat = int(logit_map[index])
        mapped_row, column = divmod(flat, tokens.shape[1])
        if mapped_row != row:
            raise ValueError("Request-to-token row mapping disagrees with logits")
        record["flat_logit_index"] = flat
        record["token"] = cpu(tokens[row, column])
        record["absolute_position"] = cpu(positions[row, column])
        expected = len(record["prompt_token_ids"]) + record["output_position"] - 1
        if int(record["absolute_position"]) != expected:
            raise ValueError("Output position disagrees with absolute input position")
        history = record["output_token_ids"] or record["prompt_token_ids"]
        if not history or int(record["token"]) != history[-1]:
            raise ValueError("Input token disagrees with exact request history")
    layers = []
    rows = [record["row"] for record in records]
    for name, module in modules:
        group = cpu(module.cache_group_idx)
        load = resolve(maps["load_indices_tensor"], group)
        store = resolve(maps["store_indices_tensor"], group)
        if store is None:
            store = load
        if (load is None or store is None or load.ndim != 1 or store.ndim != 1 or load.numel() != tokens.shape[0]):
            raise ValueError("Unexpected NSPEC0 state map")
        layer = dict(name=name, group=group, pools=[])
        for pool_index, pool in enumerate(module.kv_cache[:2]):
            # RAW incoming rows only. Full padded load/store maps remain in maps.
            selected = load[rows]
            if metadata.is_prompt and ((selected < -pool.shape[0]) | (selected >= pool.shape[0])).any():
                raise ValueError("Prefill index outside native indexing range")
            read_rows = torch.remainder(selected, pool.shape[0]).long()
            owned = pool.index_select(0, read_rows.to(pool.device)).detach().clone()
            mark_step()
            raw = cpu(owned)  # blocking, independent storage on EVERY TP rank
            layer["pools"].append(
                dict(kind="conv" if pool_index == 0 else "recurrent",
                     shape=tuple(pool.shape),
                     dtype=str(pool.dtype),
                     raw=raw,
                     read_rows=read_rows.tolist(),
                     write_rows=torch.remainder((load if pool_index == 0 and not metadata.is_prompt else store)[rows],
                                                pool.shape[0]).tolist()))
        layers.append(layer)
    payload = dict(schema=1,
                   rank=rank,
                   ordinal=ordinal,
                   raw_state=True,
                   phase="prefill" if metadata.is_prompt else "decode",
                   records=records,
                   request_ids=list(request_ids),
                   logits_requests=list(logits_requests),
                   input_shape=tuple(tokens.shape),
                   metadata=maps,
                   logits_indices=logit_map,
                   layers=layers)
    # Directory fd pins the validated directory; exclusive files never overwrite.
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(dfd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("State diagnostic directory changed")
        fd = os.open(f"state-rank{rank}-{ordinal:02d}.pt",
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600,
                     dir_fd=dfd)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
    finally:
        os.close(dfd)
