# SPDX-License-Identifier: Apache-2.0
"""Batched (bs > 1) prefill for the KDA layers: the pieces the runner's
merge opt-in relies on. CPU-only."""
import types

import torch

from vllm_gaudi.models.glm5_next import HpuGlm5NextForConditionalGeneration, _kda_save_state
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


def test_model_opts_into_batched_prefill():
    assert HpuGlm5NextForConditionalGeneration.supports_batched_mamba_prefill is True


def test_save_state_padding_row_is_identity_with_guard():
    pool = torch.arange(4 * 2 * 3 * 3, dtype=torch.float32).view(4, 2, 3, 3)
    before = pool.clone()
    new = torch.full((2, 2, 3, 3), -7.0)
    # row 0 -> slot 1; row 1 is padding (-1), which remainder() maps to slot 3
    _kda_save_state(new, pool, torch.tensor([1, -1]), guard_padding=True)
    assert torch.equal(pool[1], new[0])
    assert torch.equal(pool[3], before[3]), "padding row must not clobber the last slot"
    assert torch.equal(pool[0], before[0]) and torch.equal(pool[2], before[2])


def test_save_state_without_guard_is_the_decode_behaviour():
    pool = torch.zeros(4, 2, 3, 3)
    new = torch.ones(2, 2, 3, 3)
    _kda_save_state(new, pool, torch.tensor([0, 2]))
    assert torch.equal(pool[0], new[0]) and torch.equal(pool[2], new[1])
    assert pool[1].abs().sum() == 0 and pool[3].abs().sum() == 0


def _merge_runner(batched):
    r = HPUModelRunner.__new__(HPUModelRunner)
    r.num_mamba_like_layers = 34
    r._batched_mamba_prefill = batched
    r.use_merged_prefill = False
    r.max_prefill_batch_size = 2
    r.max_num_tokens = 8192
    r.bucketing_manager = types.SimpleNamespace(find_prompt_bucket=lambda bs, seq, nb: (bs, max(seq, 128), nb))
    return r


def _contents(n_tokens):
    return types.SimpleNamespace(context_lens=[0], get_num_tokens=lambda: [n_tokens])


def test_merge_gate_is_opt_in_for_mamba_models():
    lhs, rhs = _contents(100), _contents(200)
    assert HPUModelRunner._can_merge_prefill_contents(_merge_runner(False), lhs, rhs) is False
    assert HPUModelRunner._can_merge_prefill_contents(_merge_runner(True), lhs, rhs) is True
