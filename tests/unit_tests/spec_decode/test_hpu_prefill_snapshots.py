# SPDX-License-Identifier: Apache-2.0
"""CPU ownership oracle; does not establish an HPU runtime incident."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

RUNNER = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/worker/hpu_model_runner.py"


def load(name, namespace):
    tree = ast.parse(RUNNER.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


def setup(aux=False, spec=True, logprobs=False, skip=False):
    events = []
    ns = {
        "torch": SimpleNamespace(hpu=SimpleNamespace(synchronize=lambda: events.append("sync"))),
        "htorch": SimpleNamespace(core=SimpleNamespace(mark_step=lambda: events.append("mark")))
    }
    snapshot = load("_snapshot_prefill_hidden_states", ns)
    runner = SimpleNamespace(speculative_config=SimpleNamespace(use_eagle=lambda: True) if spec else None,
                             drafter=SimpleNamespace(),
                             _skip_draft_proposal=skip,
                             use_aux_hidden_state_outputs=aux,
                             input_batch=SimpleNamespace(num_prompt_logprobs={"r": 1} if logprobs else {}))
    return runner, snapshot, events


@pytest.mark.parametrize("aux", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_persistent_target_outputs_owned_before_deferred_propose(aux, mixed):
    runner, snapshot, events = setup(aux=aux, logprobs=True)
    hidden = torch.empty(2, 3, 4)
    auxiliary = [torch.empty_like(hidden), torch.empty_like(hidden)]

    def target(value):
        events.append("target")
        hidden.fill_(value)
        for i, tensor in enumerate(auxiliary):
            tensor.fill_(value + i + 1)
        return hidden, auxiliary

    held, held_aux, prompt_logprobs = [], [], []
    for value in [10, 20]:
        h, a = target(value)
        h, a = snapshot(runner, h, a, ["r"] if mixed else ["r", "s"], ["r", "s"])
        held.append(h)
        held_aux.append(a)
        prompt_logprobs.append(h[0])
        assert h.data_ptr() != hidden.data_ptr()
        if aux:
            assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a, auxiliary))
    assert events == ["target", "mark", "sync", "target", "mark", "sync"]
    assert held[0].data_ptr() != held[1].data_ptr()
    for i, value in enumerate([10, 20]):
        torch.testing.assert_close(held[i], torch.full_like(hidden, value))
        torch.testing.assert_close(prompt_logprobs[i], torch.full_like(hidden[0], value))

    received = []
    runner.drafter.propose = lambda *args: received.append(args[2].clone())
    runner._get_attention_group_id_for_hybrid = lambda: 0
    runner.input_batch.block_table = [SimpleNamespace(get_cpu_tensor=lambda: torch.zeros(2, 1))]
    runner.input_batch.req_id_to_index = {"r": 0, "s": 1}
    propose = load("propose_eagle_prefill", {
        "torch": torch,
        "Optional": Optional,
        "async_h2d_copy": lambda values, **kwargs: torch.tensor(values, **kwargs)
    })
    finishing = ["r"] if mixed else ["r", "s"]
    plan = {
        "token_ids": torch.zeros(2, 3, dtype=torch.long),
        "finishing": finishing,
        "rows": {
            "r": (0, 0, 3, 3),
            "s": (1, 0, 3, 6 if mixed else 3)
        }
    }
    for i in range(2):
        propose(runner, [torch.ones(len(finishing), dtype=torch.long)] * 2,
                held,
                held_aux,
                i,
                plan["token_ids"],
                torch.arange(3).repeat(2, 1),
                object(),
                torch.tensor([2, 5][:len(finishing)]),
                0,
                mtp_prompt_batch=plan)
    for value, actual in zip([10, 20], received):
        expected = (torch.cat([torch.full_like(hidden, value + 1),
                               torch.full_like(hidden, value + 2)], -1) if aux else torch.full_like(hidden, value))
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("spec,logits,skip", [(True, [], False), (False, ["r"], False), (True, ["r"], True)])
def test_no_deferred_consumer_no_copy(spec, logits, skip):
    runner, snapshot, events = setup(aux=True, spec=spec, skip=skip)
    hidden, aux = torch.ones(1, 2, 3), [torch.ones(1, 2, 3)]
    h, a = snapshot(runner, hidden, aux, logits, ["r"])
    assert h is hidden and a is aux
    assert events == []


def test_ngram_does_not_consume_hidden_states():
    runner, snapshot, events = setup()
    runner.speculative_config.use_eagle = lambda: False
    hidden = torch.ones(1, 2, 3)
    assert snapshot(runner, hidden, None, ["r"], ["r"])[0] is hidden
    assert events == []


def test_logprobs_only_owns_hidden_not_unused_aux():
    runner, snapshot, events = setup(aux=True, spec=False, logprobs=True)
    hidden, aux = torch.ones(1, 2, 3), [torch.ones(1, 2, 3)]
    h, a = snapshot(runner, hidden, aux, [], ["r"])
    hidden.zero_()
    assert h.sum() == 6 and a is aux
    assert events == ["mark", "sync"]


def test_snapshot_precedes_all_retention_sites():
    tree = ast.parse(RUNNER.read_text())
    sample = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    text = ast.unparse(sample)
    snapshot = text.index("self._snapshot_prefill_hidden_states(")
    assert text.index("self.drafter.prefill_cache_only(") < snapshot
    assert snapshot < text.index("non_flattened_hidden_states_prefills.append(")
    assert snapshot < text.index("prefill_hidden_states_for_logprobs[rid] =")
    assert snapshot < text.index("aux_hidden_states_prefills.append(")


def test_async_gate_is_capability_scoped():
    prepare = load("_prepare_mtp_prompt_cache_fill", {"HpuModelAdapter": type("Adapter", (), {})})
    runner = SimpleNamespace(speculative_config=True,
                             use_async_scheduling=True,
                             drafter=SimpleNamespace(model=SimpleNamespace(requires_prompt_cache_fill=True)))
    with pytest.raises(NotImplementedError, match="async scheduling"):
        prepare(runner, [], None, None, None, False)
    runner.drafter.model.requires_prompt_cache_fill = False
    assert prepare(runner, [], None, None, None, False) is None
