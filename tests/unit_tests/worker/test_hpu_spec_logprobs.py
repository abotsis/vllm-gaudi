# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hpu_model_runner._build_spec_logprobs_output.

A speculative step emits a variable number of tokens per request, so its
logprobs cannot use the one-row-per-request layout that
``_build_logprobs_output`` produces for ordinary decodes. The engine reads them
back through ``LogprobsLists.slice_request(req_idx, n)``, which locates a
request's first row via ``cu_num_generated_tokens[req_idx]`` -- so rows must be
grouped by *request index*, in ascending index order, with an offset entry for
every index including requests that emitted nothing.

The arithmetic here is all index bookkeeping, which is exactly the kind of thing
an end-to-end probe cannot localise: wrong offsets surface as logprobs attached
to the wrong token, not as an error.
"""

import types

import numpy as np
import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.v1.outputs import LogprobsLists, LogprobsTensors

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

BUILD = HPUModelRunner._build_spec_logprobs_output
NCOL = 3  # max_num_logprobs + 1


def _runner(req_id_to_index):
    """Minimal stand-in: the method touches nothing else on the runner."""
    return types.SimpleNamespace(input_batch=types.SimpleNamespace(req_id_to_index=req_id_to_index))


def _spec_lists(counts, pad_rows=0):
    """Mimic RejectionSampler.parse_output's return for the given per-row token
    counts, with `pad_rows` extra padded lanes after the real requests.

    Row r carries the sentinel value r in every column, so a misplaced slice is
    visible as a wrong number rather than a shape error.
    """
    all_counts = list(counts) + [1] * pad_rows
    total = sum(all_counts)
    tok = np.arange(total, dtype=np.int64).repeat(NCOL).reshape(total, NCOL)
    lps = tok.astype(np.float32) * -1.0
    rnk = np.arange(total, dtype=np.int64)
    cu = [0] + np.cumsum(all_counts).tolist()
    return LogprobsLists(tok, lps, rnk, cu)


def _prefill_segment(req_ids, start_value=100):
    n = len(req_ids)
    tok = torch.arange(start_value, start_value + n * NCOL, dtype=torch.int64).reshape(n, NCOL)
    lps = tok.to(torch.float32) * -1.0
    rnk = torch.arange(start_value, start_value + n, dtype=torch.int64)
    return (list(req_ids), LogprobsTensors(tok, lps, rnk))


def test_decode_only_ragged_counts():
    """Two decodes accepting different numbers of tokens."""
    runner = _runner({"d0": 0, "d1": 1})
    out = BUILD(runner, _spec_lists([3, 1]), ["d0", "d1"], [3, 1], [], 2)

    assert out.logprob_token_ids.shape == (4, NCOL)
    assert out.cu_num_generated_tokens == [0, 3]
    # d0 owns source rows 0-2, d1 owns row 3
    assert out.logprob_token_ids[:, 0].tolist() == [0, 1, 2, 3]
    assert out.sampled_token_ranks.tolist() == [0, 1, 2, 3]


def test_padded_lanes_are_excluded():
    """parse_output's offsets span the padded lanes; they must not be emitted."""
    runner = _runner({"d0": 0, "d1": 1})
    out = BUILD(runner, _spec_lists([2, 2], pad_rows=4), ["d0", "d1"], [2, 2], [], 2)

    assert out.logprob_token_ids.shape == (4, NCOL)
    assert out.logprob_token_ids[:, 0].tolist() == [0, 1, 2, 3]


def test_mixed_prefill_and_decode():
    """A decode emitting several tokens alongside a prefill emitting one."""
    runner = _runner({"d0": 0, "p0": 1})
    out = BUILD(runner, _spec_lists([2]), ["d0"], [2], [_prefill_segment(["p0"])], 2)

    assert out.logprob_token_ids.shape == (3, NCOL)
    assert out.cu_num_generated_tokens == [0, 2]
    assert out.logprob_token_ids[:2, 0].tolist() == [0, 1]  # decode rows
    assert out.logprob_token_ids[2, 0] == 100  # prefill row


def test_rows_follow_request_index_not_source_order():
    """The prefill sits at a lower request index than the decode."""
    runner = _runner({"p0": 0, "d0": 1})
    out = BUILD(runner, _spec_lists([2]), ["d0"], [2], [_prefill_segment(["p0"])], 2)

    assert out.cu_num_generated_tokens == [0, 1]
    assert out.logprob_token_ids[0, 0] == 100  # p0 first
    assert out.logprob_token_ids[1:, 0].tolist() == [0, 1]  # then d0


def test_gap_index_still_gets_an_offset():
    """A request index that sampled nothing contributes no rows but must not
    shift the offsets of the requests after it."""
    runner = _runner({"d0": 0, "d1": 2})
    out = BUILD(runner, _spec_lists([2, 1]), ["d0", "d1"], [2, 1], [], 3)

    assert out.logprob_token_ids.shape == (3, NCOL)
    # index 1 is absent: its offset equals the next request's start
    assert out.cu_num_generated_tokens == [0, 2, 2]
    assert out.logprob_token_ids[2, 0] == 2


def test_unknown_request_id_is_skipped():
    runner = _runner({"d0": 0})
    out = BUILD(runner, _spec_lists([1, 1]), ["d0", "gone"], [1, 1], [], 1)

    assert out.logprob_token_ids.shape == (1, NCOL)
    assert out.cu_num_generated_tokens == [0]


def test_no_rows_returns_none():
    runner = _runner({})
    assert BUILD(runner, _spec_lists([1]), ["absent"], [1], [], 0) is None


def test_none_segment_is_tolerated():
    """A sampling call that produced no logprobs shows up as a None segment."""
    runner = _runner({"d0": 0})
    out = BUILD(runner, _spec_lists([2]), ["d0"], [2], [(["d0"], None)], 1)

    assert out.logprob_token_ids.shape == (2, NCOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
