# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import habana_frameworks.torch.core as htcore
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    The current HPU implementation of spec decode will flatten the num_draft_tokens
    to 1. And so the shape size of padded_draft_token_ids will be
    the [real batch size * num_draft_tokens, 1].

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs * max_draft_tokens, 1)
        padded_target_token_ids (torch.Tensor): A 1D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs * max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        num_draft_tokens: list[int]: List of number draft tokens for each sequence.
            Shape: (num_seqs)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    # 0. wait for device processing to finish
    # NOTE(chendi): Found CPU processing is faster than HPU for this step.
    padded_draft_token_ids = padded_draft_token_ids.cpu().to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.cpu().to(torch.int32)
    bonus_token_ids = bonus_token_ids.cpu().to(torch.int32)
    cu_num_draft_tokens = cu_num_draft_tokens.cpu()
    # 1. Get tensor dimensions and device for calculations
    num_seqs = len(num_draft_tokens)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # 2. Calculate the number of draft tokens for each sequence from the
    # cumulative sum
    start_indices = torch.cat((torch.tensor([0], device=device,
                                            dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # 3. Find the first mismatch, ignoring padding tokens
    # Create a mask to only consider valid tokens for each sequence
    pos = torch.arange(max_draft_tokens, device=device)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)

    matches = (padded_draft_token_ids == padded_target_token_ids)

    mismatches = ~matches
    any_mismatch = mismatches.any(dim=1)
    # For sequence that the num draft tokens is 0, always consider all match
    any_mismatch[num_draft_tokens_per_seq == 0] = False
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    # 4. Determine the number of accepted tokens for each sequence
    # If a mismatch occurs, we accept tokens up to and including the mismatch.
    # If no mismatch, accept all *actual* draft tokens.
    num_accepted = ((first_mismatch_idx + 1) * any_mismatch + num_draft_tokens_per_seq * (~any_mismatch))

    # 5. Create the output tensor by masking the target tokens
    # Initialize the output tensor with the padding value.
    # Create output buffer.
    output_tokens = torch.empty(
        (num_seqs, max_draft_tokens + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_tokens.fill_(PLACEHOLDER_TOKEN_ID)

    # Create a mask that is True for all positions up to the number of
    # accepted tokens.
    acceptance_mask = pos < num_accepted.unsqueeze(-1)
    acceptance_mask = acceptance_mask & valid_token_mask

    # Use the mask to copy the accepted target tokens into the output tensor.
    output_slice = output_tokens[:, :max_draft_tokens]
    output_slice[acceptance_mask] = padded_target_token_ids[acceptance_mask]

    # 6. Add the bonus token where all draft tokens were accepted
    # Create a boolean mask for sequences where all drafts were a match.
    all_accepted_mask = ~any_mismatch

    # If any sequences were fully accepted, place the bonus tokens.
    if all_accepted_mask.sum() > 0:
        # Get the column indices (positions) for the bonus tokens using the mask
        bonus_pos_indices = num_draft_tokens_per_seq[all_accepted_mask].long()

        # Get the corresponding bonus token values using the mask.
        bonus_values = bonus_token_ids[all_accepted_mask].squeeze(-1)

        # Place the bonus tokens using boolean indexing for rows and integer
        # indexing for columns.
        output_tokens[all_accepted_mask, bonus_pos_indices] = bonus_values

    return output_tokens


def _token_to_request_rows(num_tokens: int, cu_num_tokens: torch.Tensor) -> torch.Tensor:
    """Map flat token position -> request row, from an INCLUSIVE cumsum.

    Comparison-matrix formulation rather than searchsorted or repeat_interleave:
    static shapes ([num_tokens, batch] bool, both tiny), no data-dependent
    output size, no kernel the HPU backend lacks.
    """
    pos = torch.arange(num_tokens, device=cu_num_tokens.device)
    return (pos.unsqueeze(1) >= cu_num_tokens.unsqueeze(0)).sum(dim=1)


def expand_batch_to_tokens(
    x: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Pure-torch replacement for the upstream triton expand_kernel.

    apply_sampling_constraints calls this for every non-greedy speculative
    batch BEFORE rejection sampling, so without this patch any temperature>0
    request dies on the missing triton backend before sampling even starts.
    """
    rows = _token_to_request_rows(num_tokens, cu_num_tokens)
    expanded = x.index_select(0, rows)
    if replace_from != replace_to:
        expanded = torch.where(expanded == replace_from,
                               torch.as_tensor(replace_to, dtype=expanded.dtype, device=expanded.device), expanded)
    return expanded


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Pure-torch replacement for the upstream triton recovery kernel.

    Semantics mirrored exactly (NO_DRAFT_PROBS branch): zero the draft token's
    probability, then gumbel-max sample via argmax(prob * 1/q) with ONE
    exponential draw per request reused across its positions, honouring
    per-request seeded generators the same way upstream does.

    Only the RNG runs on the accelerator (per-request seeds live in DEVICE
    generators, so they cannot move); everything downstream runs on host.
    Keeping the mask/mul/argmax chain on device under lazy mode returned
    out-of-vocab and uninitialised token ids through several formulations --
    in-place and functional scatter alike, with first-execution-only
    corruption in the latter -- so the boundary is drawn at the RNG: device
    graphs stay linear (generate, materialise), and no fused consumer exists
    to mis-execute. Cost is a [batch, vocab] d2h per non-greedy step.
    """
    if draft_probs is not None:
        raise NotImplementedError("HPU rejection sampling supports draft_probs=None only (MTP/eagle/ngram "
                                  "on this plugin propose token ids, not distributions).")
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    q_dtype = torch.float64 if use_fp64_gumbel else torch.float32
    vocab = target_probs.shape[-1]
    rows_q = []
    for i in range(batch_size):
        generator = sampling_metadata.generators.get(i)
        row = torch.zeros(vocab, dtype=q_dtype, device=device)
        # Skip seeded generation for rows with no draft tokens, as upstream
        # does: keeps generator advancement aligned for reproducibility.
        if generator is not None and num_draft_tokens[i] > 0:
            row.exponential_(generator=generator)
        else:
            row.exponential_()
        rows_q.append(row)
    q = torch.stack(rows_q)
    htcore.mark_step()

    # Host math in fp32 (upstream's triton kernel is fp32): the previous
    # float64 copies doubled the D2H volume and the CPU time of the largest
    # per-step tensors.
    inv_q = q.cpu().float().reciprocal()
    drafts = draft_token_ids.cpu().long()
    # The HPU layout pads draft_token_ids past cu[-1] (rectangular buckets),
    # so trailing pad positions map past the last request -- clamp them onto
    # the final row. Their recovered values are never read: a padded draft id
    # of -1 always rejects before recovery is consulted. Callers may pass only
    # the real rows (num_tokens == target_probs.shape[0]).
    num_tokens = min(num_tokens, target_probs.shape[0], drafts.shape[0])
    rows = _token_to_request_rows(num_tokens, cu_num_draft_tokens.cpu()).clamp(max=batch_size - 1)
    probs = target_probs[:num_tokens].cpu().float() if target_probs.device.type != "cpu" \
        else target_probs[:num_tokens].float()
    masked = probs.scatter(1, drafts[:num_tokens].clamp(min=0).view(-1, 1), 0.0)
    recovered = (masked * inv_q.index_select(0, rows)).argmax(dim=-1)
    return recovered.to(torch.int32).to(device)


def apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata):
    """Identity on HPU. Upstream applies temperature/top-k/top-p to the target
    logits IN PLACE on device before rejection sampling. In a mixed
    greedy+random batch those in-place ops corrupted the tensor (greedy rows
    emitted token soup while an all-greedy batch -- where this
    function early-returns -- was clean; the sampled rows looked fine only
    because accepted tokens are the draft model's, not the target argmax).
    Same lazy-graph genre as the four previous sampler bugs; same cure: leave
    the device tensor untouched and apply the constraints on host, inside
    rejection_sample, where the probabilities already live. The greedy argmax
    then runs on the unmodified tensor -- byte-identical to the all-greedy
    path by construction."""
    return logits


def _apply_constraints_cpu(logits_cpu, cu_cpu, sampling_metadata, num_tokens):
    """Temperature/top-k/top-p on host, mirroring upstream semantics.

    fp32, one descending sort shared by top-k and top-p, and each constraint
    skipped when no row in the batch uses it. The earlier float64 version
    sorted the full vocab twice per row and was ~40% of a sampled MTP step.
    """
    rows = _token_to_request_rows(num_tokens, cu_cpu).clamp(max=len(cu_cpu) - 1)
    temp = sampling_metadata.temperature
    if temp is not None:
        t = temp.cpu().float().index_select(0, rows)
        # The engine stores -1.0 as the greedy sentinel in the temperature
        # tensor (vLLM InputBatch convention), NOT GREEDY_TEMPERATURE == 0.
        # Dividing by -1 NEGATES the logits -- softmax then concentrates on the
        # least-likely tokens, which is exactly the mixed-batch soup: greedy
        # rows emitted ids the model scored at logprob -26 while its true
        # argmax sat at -0.02. Treat any t <= 0 as greedy.
        t = torch.where(t <= 0, torch.ones_like(t), t)
        logits_cpu = logits_cpu / t.unsqueeze(-1)
    vocab = logits_cpu.shape[-1]
    k = None
    if sampling_metadata.top_k is not None:
        k = sampling_metadata.top_k.cpu().long().index_select(0, rows)
        k = torch.where(k <= 0, torch.full_like(k, vocab), k).clamp(max=vocab)
        if bool((k >= vocab).all()):
            k = None
    pv = None
    if sampling_metadata.top_p is not None:
        pv = sampling_metadata.top_p.cpu().float().index_select(0, rows).view(-1, 1)
        if bool((pv >= 1.0).all()):
            pv = None
    if k is None and pv is None:
        return logits_cpu
    sorted_logits, sorted_idx = torch.sort(logits_cpu, dim=-1, descending=True)
    drop_sorted = torch.zeros_like(sorted_logits, dtype=torch.bool)
    if k is not None:
        # positions at or past k in descending order are outside the top-k
        drop_sorted |= torch.arange(vocab).view(1, -1) >= k.view(-1, 1)
    if pv is not None:
        # top-p on the top-k-filtered distribution (upstream order: k then p)
        kept = sorted_logits.masked_fill(drop_sorted, float("-inf"))
        probs = torch.softmax(kept, dim=-1)
        cum = probs.cumsum(dim=-1)
        # keep tokens while the cumulative mass BEFORE them is < top_p
        drop_sorted |= (cum - probs) >= pv
    drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
    return logits_cpu.masked_fill(drop, float("-inf"))


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """HPU rejection sampling: greedy fast path + a real non-greedy path.

    Upstream implements both in triton, which does not exist on HPU. The greedy
    algorithm runs on CPU (measured faster there for these shapes); the random
    algorithm keeps the [num_tokens, vocab] work on device and pulls only the
    per-token accept bits, recovered ids, and drafts to host for assembly.

    The result is returned ON DEVICE. This is upstream forward()'s contract,
    and returning the CPU tensor here is what killed the engine whenever
    `logprobs` were requested: _get_logprobs_tensors flattens these token ids
    and hands them to Sampler.gather_logprobs as gather indices against HPU
    logprobs, and a CPU index tensor into an HPU gather is a bridge-fatal
    device mix ("Got a non-HPU tensor"). Verified by direct experiment; the
    same call with device token ids passes.
    """
    if synthetic_mode:
        raise NotImplementedError("synthetic rejection-sampling mode is not supported on HPU")
    # The HPU decode path carries token ids as [num_tokens, 1] columns, so the
    # engine hands 2-D draft ids where upstream's triton version asserts 1-D.
    # The greedy helper flattens internally and never noticed; the random
    # path's (drafts >= 0) mask kept the column shape and broadcast a [T]
    # comparison into [T, T] -- "shape '[1, 4]' is invalid for input of size
    # 16" on the first non-greedy request. Normalise once, up front.
    draft_token_ids = draft_token_ids.reshape(-1)
    device = target_logits.device

    # Greedy sub-result for every row: argmax(logits) == argmax(probs), so the
    # softmax is skipped. For all-greedy batches this is the entire answer.
    target_argmax = target_logits.argmax(dim=-1)
    greedy_out = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                          cu_num_draft_tokens)
    if sampling_metadata.all_greedy:
        return greedy_out.to(device)

    # ---- random path (mirrors rejection_random_sample_kernel, NO_DRAFT_PROBS) ----
    # Device work is limited to the softmax and the seeded RNG, each pulled
    # straight to host: fusing anything downstream of the RNG into a lazy
    # graph mis-executed here (see sample_recovered_tokens). target_logits
    # arrive already temperature/top-k/top-p-constrained by
    # apply_sampling_constraints.
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    uniform = rejection_sampler.generate_uniform_probs(num_tokens, num_draft_tokens, sampling_metadata.generators,
                                                       device)
    htcore.mark_step()

    cu_cpu0 = cu_num_draft_tokens.cpu()
    # Hard sync before pulling. The host-side argmax STILL emitted junk for
    # greedy rows on hardware even though the logprobs path's LATER pull of the
    # same tensor was sane -- so the .cpu() here can complete before the
    # producing graph has finished under PT_HPU_LAZY_ACC_PAR_MODE=1. Force the
    # full pipeline to drain first.
    htcore.mark_step()
    torch.hpu.synchronize()
    # Only the first cu[-1] token rows are real; the rest are bucket padding
    # (draft id -1, always rejected, recovered value never read). Pull and
    # process the real rows only, in fp32: with one request in a bucket of 8
    # this is 8x less host work than the padded float64 path it replaces.
    n_real = min(int(cu_cpu0[-1].item()), num_tokens) if len(cu_cpu0) else 0
    if n_real == 0:
        return greedy_out.to(device)
    raw_logits_cpu = target_logits[:n_real].cpu().float()
    # Recompute the greedy sub-result from the HOST copy of the logits. In a
    # mixed batch the bonus sampler's random path (gumbel + generators) fuses
    # into the same device graph the argmax above ran in, and that graph
    # emits garbage ids for the greedy rows -- measured live: a greedy row
    # emitted token ids the model itself scored at logprob -26 while its true
    # argmax sat at -0.02, so the model was healthy and the ids were junk.
    # All-greedy batches take the bonus sampler's greedy path (a different,
    # long-proven graph), which is why only mixed batches corrupt and why the
    # device argmax stays trusted on the all-greedy fast path above.
    host_argmax = torch.zeros(num_tokens, dtype=torch.int64)
    host_argmax[:n_real] = raw_logits_cpu.argmax(dim=-1)
    greedy_out = rejection_sample_pytorch(draft_token_ids, host_argmax, bonus_token_ids, num_draft_tokens,
                                          cu_num_draft_tokens)
    logits_cpu = _apply_constraints_cpu(raw_logits_cpu, cu_cpu0, sampling_metadata, n_real)
    probs_cpu = torch.softmax(logits_cpu, dim=-1)
    recovered_real = sample_recovered_tokens(max_spec_len, num_draft_tokens, cu_num_draft_tokens,
                                             draft_token_ids[:n_real], draft_probs, probs_cpu, sampling_metadata,
                                             device, use_fp64_gumbel)

    drafts_cpu64 = draft_token_ids.cpu().long()
    p_draft = torch.zeros(num_tokens, dtype=probs_cpu.dtype)
    p_draft[:n_real] = probs_cpu.gather(-1, drafts_cpu64[:n_real].clamp(min=0).view(-1, 1)).squeeze(-1)
    # Accept iff target_prob(draft) >= u (draft_prob treated as 1); a negative
    # draft id is a padded position and always rejects.
    accept_cpu = (p_draft >= uniform.cpu().float()) & (drafts_cpu64 >= 0)
    accept_cpu[n_real:] = False

    # ---- host assembly, same masking scheme as the greedy helper ----
    recovered_cpu = torch.zeros(num_tokens, dtype=torch.int32)
    recovered_cpu[:n_real] = recovered_real.cpu()
    draft_cpu = draft_token_ids.cpu().to(torch.int32)
    bonus_cpu = bonus_token_ids.cpu().to(torch.int32).view(-1)
    cu_cpu = cu_num_draft_tokens.cpu()
    # <= 0, not == GREEDY_TEMPERATURE: the engine's tensor carries -1.0 for
    # greedy requests (see _apply_constraints_cpu). Comparing against 0 routed
    # every greedy row down the random path in mixed batches.
    is_greedy_cpu = (sampling_metadata.temperature <= 0).cpu()

    starts = torch.cat((torch.tensor([0], dtype=cu_cpu.dtype), cu_cpu[:-1]))
    ndraft = (cu_cpu - starts).to(torch.int64)
    max_draft = draft_cpu.view(batch_size, -1).shape[-1]
    pos = torch.arange(max_draft)
    valid = pos < ndraft.unsqueeze(-1)

    acc_v = accept_cpu.view(batch_size, max_draft) & valid
    rej_v = (~acc_v) & valid
    any_rej = rej_v.any(dim=1)
    first_rej = torch.argmax(rej_v.int(), dim=1)
    n_acc = torch.where(any_rej, first_rej, ndraft)

    out = torch.full((batch_size, max_draft + 1), rejection_sampler.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    prefix = pos < n_acc.unsqueeze(-1)
    out[:, :max_draft][prefix] = draft_cpu.view(batch_size, max_draft)[prefix]
    correction = torch.where(any_rej,
                             recovered_cpu.view(batch_size, max_draft).gather(1, first_rej.view(-1, 1)).view(-1),
                             bonus_cpu)
    out[torch.arange(batch_size), n_acc] = correction

    out = torch.where(is_greedy_cpu.view(-1, 1), greedy_out, out)
    return out.to(device)


def _get_logprobs_tensors_cpu(
    self,
    max_num_logprobs: int,
    metadata,
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
):
    """CPU mirror of RejectionSampler._get_logprobs_tensors.

    The upstream device version, run for the first time on HPU (warmup never
    exercises the logprobs path), returns garbage top-k token ids (observed:
    -665235088 among 32k-vocab indices; clean on the second execution). The
    engine then hands them to the tokenizer for the response's top_logprobs and
    dies with OverflowError. Same first-execution lazy-graph genre as the
    non-greedy sampler; same cure: pull the inputs once and compute on host,
    where every consumer of these tensors already lives (numpy filter in
    parse_output, list conversion in the runner). Cost is a few tens of MB d2h
    on requests that ask for logprobs.
    """
    from vllm.v1.outputs import LogprobsTensors

    cu_cpu = metadata.cu_num_sampled_tokens.cpu()
    starts = torch.zeros_like(cu_cpu)
    starts[1:] = cu_cpu[:-1]

    logits_cpu = logits.cpu().to(torch.float32)
    final_logits = torch.zeros_like(logits_cpu)
    # CPU advanced indexing gives -1 the same "last row" meaning CUDA does, so
    # the padded lanes land exactly where upstream's would.
    final_logits[metadata.target_logits_indices.cpu()] = target_logits.cpu().to(torch.float32)
    final_logits[metadata.bonus_logits_indices.cpu()] = bonus_logits.cpu().to(torch.float32)

    sampled_cpu = sampled_token_ids.cpu()
    offsets = torch.arange(sampled_cpu.shape[-1], dtype=starts.dtype)
    accepted_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
    accepted_idx.clamp_(max=final_logits.shape[0] - 1)
    accepted_tokens = sampled_cpu.clone().flatten().long()
    accepted_tokens[accepted_tokens == rejection_sampler.PLACEHOLDER_TOKEN_ID] = 0

    accepted_logprobs = final_logits.index_select(0, accepted_idx)
    if not self.is_logits_logprobs_mode:
        accepted_logprobs = torch.log_softmax(accepted_logprobs, dim=-1)

    topk_lp, topk_idx = torch.topk(accepted_logprobs, max_num_logprobs, dim=-1)
    tok_lp = accepted_logprobs.gather(-1, accepted_tokens.view(-1, 1))
    ranks = (accepted_logprobs >= tok_lp).sum(dim=-1)
    ids = torch.cat((accepted_tokens.view(-1, 1), topk_idx), dim=1).to(torch.int32)
    lps = torch.cat((tok_lp, topk_lp), dim=1)
    return LogprobsTensors(ids, lps, ranks)


rejection_sampler.rejection_sample = rejection_sample
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints
rejection_sampler.expand_batch_to_tokens = expand_batch_to_tokens
rejection_sampler.sample_recovered_tokens = sample_recovered_tokens
rejection_sampler.RejectionSampler._get_logprobs_tensors = _get_logprobs_tensors_cpu
