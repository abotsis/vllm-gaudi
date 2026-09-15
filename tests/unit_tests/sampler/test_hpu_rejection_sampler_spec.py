# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU rejection sampler: device contract, greedy parity, and the non-greedy path.

Three production failures anchor these tests:
  * `logprobs: true` under spec decode killed the engine because the patched
    rejection_sample returned CPU token ids into upstream code that gathers
    HPU logprobs with them (bridge-fatal device mix). The device-contract test
    here fails on that regression in seconds instead of a 20-minute boot.
  * `temperature > 0` first died on mis-padded sampling metadata, then would
    have hit the greedy-only assert. The non-greedy tests cover the replacement
    implementation, mirrored from upstream's triton kernels.
  * apply_sampling_constraints calls expand_batch_to_tokens (triton upstream)
    before sampling for any non-greedy batch, so its pure-torch patch is on the
    critical path too.
"""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

import vllm_gaudi.v1.sample.hpu_rejection_sampler as hpu_rs
from vllm.v1.sample import rejection_sampler as rs
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

DEV = "hpu"
PLACEHOLDER = rs.PLACEHOLDER_TOKEN_ID


def _metadata(temps, generators=None):
    t = torch.tensor(temps, dtype=torch.float32, device=DEV)
    return SamplingMetadata(
        temperature=t,
        all_greedy=bool((t == 0).all()),
        all_random=bool((t != 0).all()),
        top_p=None,
        top_k=None,
        generators=generators or {},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(len(temps), device=DEV),
        presence_penalties=torch.zeros(len(temps), device=DEV),
        repetition_penalties=torch.ones(len(temps), device=DEV),
        output_token_ids=[[] for _ in temps],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _batch(B, nspec, V, ndraft=None, seed=0):
    """target logits + drafts; ndraft per row (default nspec everywhere)."""
    g = torch.Generator().manual_seed(seed)
    ndraft = ndraft if ndraft is not None else [nspec] * B
    T = B * nspec
    logits = torch.randn(T, V, generator=g).to(DEV)
    drafts = torch.randint(0, V, (T, ), generator=g, dtype=torch.int32).to(DEV)
    flat_valid = torch.cat([torch.arange(nspec) < n for n in ndraft])
    drafts = torch.where(flat_valid.to(DEV), drafts, torch.tensor(-1, dtype=torch.int32, device=DEV))
    bonus = torch.randint(0, V, (B, 1), generator=g, dtype=torch.int32).to(DEV)
    cu = torch.tensor(ndraft, device=DEV).cumsum(0)
    return logits, drafts, bonus, ndraft, cu


def _run(logits, drafts, bonus, ndraft, cu, sm, max_spec):
    return rs.rejection_sample(drafts, ndraft, max_spec, cu, None, logits, bonus, sm)


def test_result_is_on_device():
    """The regression that killed the engine: output must live on HPU."""
    logits, drafts, bonus, nd, cu = _batch(4, 3, 97)
    out = _run(logits, drafts, bonus, nd, cu, _metadata([0.0] * 4), 3)
    assert out.device.type == "hpu"
    out2 = _run(logits, drafts, bonus, nd, cu, _metadata([0.7] * 4), 3)
    assert out2.device.type == "hpu"


def test_greedy_parity_with_reference():
    logits, drafts, bonus, nd, cu = _batch(4, 3, 97, ndraft=[3, 3, 2, 0])
    out = _run(logits, drafts, bonus, nd, cu, _metadata([0.0] * 4), 3).cpu()
    ref = hpu_rs.rejection_sample_pytorch(drafts, logits.argmax(-1), bonus, nd, cu)
    assert torch.equal(out, ref.cpu())


def test_random_accepts_when_target_agrees():
    """Target distribution concentrated on the draft tokens -> all accepted + bonus."""
    B, nspec, V = 3, 3, 50
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V)
    hot = torch.full_like(logits, -30.0)
    hot[torch.arange(B * nspec, device=DEV), drafts.long()] = 30.0
    out = _run(hot, drafts, bonus, nd, cu, _metadata([1.0] * B), nspec).cpu()
    assert torch.equal(out[:, :nspec], drafts.view(B, nspec).cpu())
    assert torch.equal(out[:, nspec], bonus.view(-1).cpu())


def test_random_rejects_when_target_disagrees():
    """Target mass entirely off the draft tokens -> reject at 0, recovered != draft."""
    B, nspec, V = 3, 3, 50
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V)
    cold = torch.full_like(logits, -30.0)
    other = (drafts.long() + 1) % V
    cold[torch.arange(B * nspec, device=DEV), other] = 30.0
    out = _run(cold, drafts, bonus, nd, cu, _metadata([1.0] * B), nspec).cpu()
    d = drafts.view(B, nspec).cpu()
    assert (out[:, 0] != d[:, 0]).all(), "position 0 must be a recovered token"
    assert (out[:, 0] == other.view(B, nspec)[:, 0].cpu()).all(), \
        "recovered token must follow the target distribution"
    assert (out[:, 1:] == PLACEHOLDER).all(), "nothing after the first rejection"


def test_mixed_greedy_and_random_rows():
    B, nspec, V = 4, 3, 50
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V)
    hot = torch.full_like(logits, -30.0)
    hot[torch.arange(B * nspec, device=DEV), drafts.long()] = 30.0
    sm = _metadata([0.0, 1.0, 0.0, 1.0])
    out = _run(hot, drafts, bonus, nd, cu, sm, nspec).cpu()
    # target argmax == draft everywhere, so greedy rows accept everything too:
    # both row kinds emit drafts + bonus, via different code paths.
    assert torch.equal(out[:, :nspec], drafts.view(B, nspec).cpu())
    assert torch.equal(out[:, nspec], bonus.view(-1).cpu())


def test_zero_draft_row_gets_bonus():
    B, nspec, V = 2, 3, 50
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V, ndraft=[3, 0])
    out = _run(logits, drafts, bonus, nd, cu, _metadata([1.0] * B), nspec).cpu()
    assert out[1, 0] == bonus.view(-1).cpu()[1]
    assert (out[1, 1:] == PLACEHOLDER).all()


def test_seeded_determinism():
    B, nspec, V = 2, 3, 211
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V, seed=7)
    mid = torch.zeros_like(logits)  # flat: acceptance genuinely random
    outs = []
    for _ in range(2):
        gens = {i: torch.Generator(device=DEV).manual_seed(123 + i) for i in range(B)}
        sm = _metadata([1.0] * B, generators=gens)
        outs.append(_run(mid, drafts, bonus, nd, cu, sm, nspec).cpu())
    assert torch.equal(outs[0], outs[1]), "same seeds must reproduce the same tokens"


def test_expand_batch_to_tokens_matches_reference():
    x = torch.tensor([3.0, 5.0, 7.0], device=DEV)
    cu = torch.tensor([2, 5, 6], device=DEV)
    got = rs.expand_batch_to_tokens(x, cu, 6).cpu()
    assert torch.equal(got, torch.tensor([3.0, 3.0, 5.0, 5.0, 5.0, 7.0]))
    got2 = rs.expand_batch_to_tokens(torch.tensor([0.0, 2.0, 0.0], device=DEV), cu, 6, replace_from=0,
                                     replace_to=1).cpu()
    assert torch.equal(got2, torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 1.0]))


def test_recovered_never_returns_the_rejected_draft():
    B, nspec, V = 4, 4, 61
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V, seed=3)
    sm = _metadata([1.0] * B, generators={})
    rec = rs.sample_recovered_tokens(nspec, nd, cu, drafts, None, logits.softmax(-1, dtype=torch.float32), sm,
                                     torch.device(DEV))
    valid = (drafts >= 0).cpu()
    assert (rec.cpu()[valid] != drafts.cpu()[valid]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _spec_batch_with_padding(B=8, nspec=4, V=32768, real=2, seed=0):
    """Metadata shaped exactly as HpuModelRunner._prepare_spec_decode_inputs
    builds it: rectangular rows, -1-padded indices past each row's tokens."""
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    import numpy as np
    torch.manual_seed(seed)
    rows = B * (nspec + 1)
    logits = torch.randn(rows, V, device=DEV)
    token_ids = torch.randint(0, V, (rows, ), device=DEV, dtype=torch.int32)
    nd = [nspec] * real + [0] * (B - real)
    tli, bli, li = [], [], []
    for b in range(B):
        n = nd[b] + 1
        base = b * (nspec + 1)
        for i in range(n - 1):
            tli.append(base + i)
            li.append(base + i)
        bli.append(base + n - 1)
        li.append(base + n - 1)
        tli.extend([-1] * (nspec + 1 - n))
        li.extend([-1] * (nspec + 1 - n))
    tli_t = torch.tensor(tli, device=DEV)
    meta = SpecDecodeMetadata(
        draft_token_ids=token_ids[(tli_t + 1).clamp(min=0).long()][:len(tli)],
        num_draft_tokens=nd,
        cu_num_draft_tokens=torch.tensor(np.cumsum(nd), device=DEV),
        cu_num_sampled_tokens=torch.tensor(np.cumsum([x + 1 for x in nd]), device=DEV),
        target_logits_indices=tli_t.long(),
        bonus_logits_indices=torch.tensor(bli, device=DEV),
        logits_indices=torch.tensor(li, device=DEV),
    )
    return logits, meta


def test_logprobs_token_ids_stay_in_vocab_from_first_execution():
    """The third engine kill: with logprobs requested, the device logprobs
    chain emitted garbage top-k ids (e.g. -665235088) on its FIRST execution
    only -- warmup never runs this path, so the first real logprobs request
    took the server down inside the tokenizer (OverflowError). Values, not
    shapes, and iteration 0 specifically, are what this pins."""
    from vllm.v1.sample.sampler import Sampler
    rej = rs.RejectionSampler(Sampler())
    V = 32768
    for it in range(2):
        logits, meta = _spec_batch_with_padding(V=V, seed=it)
        sm = _metadata([0.0] * 8)
        sm = type(sm)(**{**sm.__dict__, "max_num_logprobs": 3})
        out = rej(meta, None, logits, sm)
        lp = out.logprobs_tensors
        tok = lp.logprob_token_ids.cpu()
        assert int(tok.min()) >= 0 and int(tok.max()) < V, \
            f"iteration {it}: out-of-vocab logprob token ids (min={int(tok.min())}, max={int(tok.max())})"
        rnk = lp.selected_token_ranks.cpu()
        assert int(rnk.min()) >= 1 and int(rnk.max()) <= V
        assert torch.isfinite(lp.logprobs.cpu()).all()
        sampled = out.sampled_token_ids.cpu()
        assert int(sampled.max()) < V and int(sampled.min()) >= -1


def test_column_shaped_draft_ids_from_the_engine():
    """The engine's decode path hands draft_token_ids as a [T, 1] column, not
    the 1-D vector upstream's triton kernels assert. The greedy helper always
    flattened internally; the random path broadcast a [T] comparison against
    the [T, 1] validity mask into [T, T] and died on the first temperature>0
    request ("shape '[1, 4]' is invalid for input of size 16"). Same batch as
    the engine: one real request, four drafts, column-shaped."""
    B, nspec, V = 1, 4, 97
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V, seed=5)
    drafts_col = drafts.view(-1, 1)  # the engine's layout
    for temps in ([0.7], [0.0]):
        out = rs.rejection_sample(drafts_col, nd, nspec, cu, None, logits.clone(), bonus, _metadata(temps))
        assert out.shape == (B, nspec + 1)
        assert out.device.type == "hpu"
        flat = out.cpu().flatten()
        assert int(flat.max()) < V and int(flat.min()) >= -1


def test_engine_greedy_sentinel_is_minus_one():
    """The bug three boot-level fixes missed: the engine's temperature tensor
    holds -1.0 for greedy requests (vLLM InputBatch sentinel), not 0. A
    ==GREEDY_TEMPERATURE comparison routed greedy rows down the random path in
    mixed batches, and dividing by -1 NEGATED their logits -- soup made of the
    model's least-likely tokens. Greedy rows under the REAL sentinel must
    byte-match the pure-greedy result."""
    B, nspec, V = 4, 4, 97
    logits, drafts, bonus, nd, cu = _batch(B, nspec, V, seed=21)
    ref = rs.rejection_sample(drafts, nd, nspec, cu, None, logits.clone(), bonus, _metadata([0.0] * B)).cpu()
    gens = {i: torch.Generator(device=DEV).manual_seed(3 + i) for i in range(B)}
    mix = rs.rejection_sample(drafts, nd, nspec, cu, None, logits.clone(), bonus, _metadata([-1.0, 0.7, -1.0, 1.0],
                                                                                            gens)).cpu()
    for r in (0, 2):
        assert torch.equal(mix[r], ref[r]), f"greedy row {r} diverged under the -1.0 sentinel"
    srows = mix[[1, 3]].flatten()
    assert int(srows.max()) < V and int(srows.min()) >= -1


def test_mixed_batch_greedy_rows_match_pure_greedy_full_forward():
    """Boot-10's finding: in a mixed greedy+random batch, upstream applies
    temperature/top-k/top-p to the target logits IN PLACE on device before
    rejection sampling (the all-greedy batch early-returns), and those in-place
    device ops corrupted the tensor -- greedy rows emitted token soup while
    sampled rows looked fine only because accepted tokens are the draft's.
    With constraints moved to host, greedy rows must be byte-identical to the
    pure-greedy forward by construction. Uses non-None top_p/top_k tensors and
    dataclasses.replace for realism (the engine always sends tensors here)."""
    from dataclasses import replace as dc_replace
    from vllm.v1.sample.sampler import Sampler
    rej = rs.RejectionSampler(Sampler())
    B, nspec, V = 4, 4, 8192
    for it in range(2):
        torch.manual_seed(40 + it)
        rows = B * nspec
        logits = torch.randn(B * (nspec + 1), V, device=DEV)
        drafts = torch.randint(0, V, (rows, 1), device=DEV, dtype=torch.int32)  # column
        import numpy as np
        nd = [nspec] * B
        tli = torch.tensor([b * (nspec + 1) + i for b in range(B) for i in range(nspec)], device=DEV)
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        meta = SpecDecodeMetadata(
            draft_token_ids=drafts,
            num_draft_tokens=nd,
            cu_num_draft_tokens=torch.tensor(np.cumsum(nd), device=DEV),
            cu_num_sampled_tokens=torch.tensor(np.cumsum([x + 1 for x in nd]), device=DEV),
            target_logits_indices=tli.long(),
            bonus_logits_indices=torch.tensor([b * (nspec + 1) + nspec for b in range(B)], device=DEV),
            logits_indices=torch.arange(B * (nspec + 1), device=DEV),
        )
        base = _metadata([0.0] * B)
        sm_greedy = dc_replace(base,
                               top_p=torch.ones(B, device=DEV),
                               top_k=torch.zeros(B, dtype=torch.int64, device=DEV))
        gens = {i: torch.Generator(device=DEV).manual_seed(7 + i) for i in range(B)}
        sm_mixed = dc_replace(_metadata([0.0, 0.7, 0.0, 1.0], generators=gens),
                              top_p=torch.ones(B, device=DEV),
                              top_k=torch.zeros(B, dtype=torch.int64, device=DEV))

        ref = rej(meta, None, logits.clone(), sm_greedy).sampled_token_ids.cpu()
        mix = rej(meta, None, logits.clone(), sm_mixed).sampled_token_ids.cpu()
        for r in (0, 2):
            assert torch.equal(mix[r], ref[r]), \
                f"iteration {it}: greedy row {r} diverged in a mixed batch"
        srows = mix[[1, 3]].flatten()
        assert int(srows.max()) < V and int(srows.min()) >= -1
