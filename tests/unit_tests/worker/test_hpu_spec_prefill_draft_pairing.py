# SPDX-License-Identifier: Apache-2.0
"""The MTP/Eagle draft for prefills must pair sampled tokens with prefill
batches through the batches that actually produced logits.

An intermediate chunk (chunked prefill, or the Mamba block-aligned split that
turns any prompt of >= 2 KDA blocks into two chunks) yields no logits and no
sampled token. Pairing by batch index then handed the first batch the SECOND
batch's token together with zero-row hidden states; the draft head captured a
0-token HPU graph and died in the router ("cannot reshape tensor of 0
elements", 2026-09-03, a tool-calling prompt under structured output). The
block-table start index must also advance for skipped batches. CPU-only.
"""
import types

import torch

from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner, PrefillInputData


class _FakeDrafter(EagleProposer):
    """Passes the runner's isinstance check; records the pairing it is given."""

    def __init__(self):  # noqa: D401 -- deliberately skips EagleProposer.__init__
        self.calls = []


def _runner(hidden_rows_per_batch, num_spec=4):
    r = HPUModelRunner.__new__(HPUModelRunner)
    r.speculative_config = types.SimpleNamespace(method="mtp", num_speculative_tokens=num_spec, use_eagle=lambda: True)
    r.drafter = _FakeDrafter()
    r.use_aux_hidden_state_outputs = False
    calls = r.drafter.calls

    def fake_prefill(prefill_sampled,
                     hidden_prefills,
                     aux,
                     idx,
                     token_ids,
                     position_ids,
                     attn_metadata,
                     logits_indices,
                     batch_start_idx,
                     sampled_idx=None):
        if sampled_idx is None:
            sampled_idx = idx
        calls.append((idx, sampled_idx, int(hidden_prefills[idx].shape[0]), batch_start_idx,
                      int(prefill_sampled[sampled_idx].item())))
        return torch.full((1, num_spec), fill_value=idx, dtype=torch.long)

    r.propose_eagle_prefill = fake_prefill
    return r


def _prefill_data(n_batches, logits_rows):
    d = PrefillInputData()
    for b in range(n_batches):
        d.request_ids.append([f"req{b}"])
        d.prompt_lens.append([100])
        d.token_ids.append(torch.zeros(1, 8, dtype=torch.long))
        d.position_ids.append(torch.zeros(1, 8, dtype=torch.long))
        d.attn_metadata.append(None)
        d.logits_indices.append(torch.arange(logits_rows[b]))
        d.logits_requests.append([f"req{b}"] if logits_rows[b] else [])
    return d


def _propose(r, prefill_data, hidden_rows, sampled_tokens, with_logits):
    return HPUModelRunner.propose_draft_token_ids(
        r,
        scheduler_output=None,
        sampled_token_ids=[],
        sampling_metadata=None,
        hidden_states=None,
        sample_hidden_states=None,
        aux_hidden_states=None,
        prefill_sampled_token_ids_tensor=[torch.tensor([t]) for t in sampled_tokens],
        decode_sampled_token_ids_tensor=None,
        hidden_states_prefills=[torch.zeros(rows, 1, 16) for rows in hidden_rows],
        sample_hidden_states_prefills=None,
        aux_hidden_states_prefills=None,
        num_decodes=0,
        prefill_data=prefill_data,
        decode_data=None,
        prefill_batches_with_logits=with_logits)


def test_intermediate_chunk_before_finishing_prompt_is_skipped():
    # batch 0: intermediate chunk (no logits); batch 1: finishing prompt.
    r = _runner([0, 1])
    drafts = _propose(r, _prefill_data(2, [0, 1]), hidden_rows=[0, 1], sampled_tokens=[777], with_logits=[1])
    assert r.drafter.calls == [(1, 0, 1, 1, 777)
                               ], r.drafter.calls  # batch 1, sampled slot 0, start advanced past batch 0
    assert drafts.shape == (1, 4) and int(drafts[0, 0]) == 1


def test_three_batches_middle_empty():
    r = _runner([1, 0, 1])
    drafts = _propose(r,
                      _prefill_data(3, [1, 0, 1]),
                      hidden_rows=[1, 0, 1],
                      sampled_tokens=[11, 33],
                      with_logits=[0, 2])
    assert r.drafter.calls == [(0, 0, 1, 0, 11), (2, 1, 1, 2, 33)]
    assert drafts.shape == (2, 4)


def test_only_empty_chunks_yield_no_drafts():
    r = _runner([0])
    drafts = _propose(r, _prefill_data(1, [0]), hidden_rows=[0], sampled_tokens=[], with_logits=[])
    assert r.drafter.calls == []
    assert drafts is None


def test_legacy_callers_without_the_list_pair_by_index():
    r = _runner([1, 1])
    _propose(r, _prefill_data(2, [1, 1]), hidden_rows=[1, 1], sampled_tokens=[5, 6], with_logits=None)
    assert r.drafter.calls == [(0, 0, 1, 0, 5), (1, 1, 1, 1, 6)]
