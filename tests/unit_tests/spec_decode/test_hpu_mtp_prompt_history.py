# SPDX-License-Identifier: Apache-2.0
"""CPU tests of production methods without importing HPU/vLLM runtimes."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
EAGLE = ROOT / "vllm_gaudi/v1/spec_decode/hpu_eagle.py"


def load(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def api():
    copies = []

    def copy(values, *, device, dtype):
        copies.append(list(values))
        return torch.tensor(values, device=device, dtype=dtype)

    class Adapter:

        def __init__(self, model):
            self.model = model

    ns = {"torch": torch, "async_h2d_copy": copy, "HpuModelAdapter": Adapter}
    shift = load(RUNNER, "_shift_mtp_prompt_chunk", ns)
    prepare = load(RUNNER, "_prepare_mtp_prompt_cache_fill", ns)
    fill = load(EAGLE, "prefill_cache_only", ns)
    propose = load(EAGLE, "propose", ns)
    return SimpleNamespace(shift=shift, prepare=prepare, fill=fill, propose=propose, copies=copies, adapter=Adapter)


@pytest.mark.parametrize("shape", [(5, ), (1, 5), (2, 5)])
@pytest.mark.parametrize("query", [1, 3])
def test_shift_preserves_bucket_and_padding(api, shape, query):
    ids = torch.full(shape, -1, dtype=torch.int32)
    ids.reshape(-1)[:query] = torch.arange(10, 10 + query)
    before = ids.clone()
    prompt = torch.arange(10, 20)
    shifted = api.shift(ids, prompt, 0, query, 10)
    assert shifted.shape == ids.shape
    assert shifted.reshape(-1)[:query].tolist() == list(range(11, 11 + query))
    assert shifted.reshape(-1)[query:].tolist() == before.reshape(-1)[query:].tolist()
    assert api.copies == [[10 + query]]
    torch.testing.assert_close(ids, before)


def runner(api, capability=True, wrapped=True):
    model = SimpleNamespace(requires_prompt_cache_fill=capability)
    if wrapped:
        model = api.adapter(model)
    return SimpleNamespace(speculative_config=True,
                           drafter=SimpleNamespace(model=model),
                           use_prefix_caching=False,
                           requests={"r": SimpleNamespace(output_token_ids=[])},
                           input_batch=SimpleNamespace(req_id_to_index={"r": 0},
                                                       num_computed_tokens_cpu=[0],
                                                       num_prompt_tokens=[7],
                                                       token_ids_cpu=torch.arange(10, 17)[None]))


@pytest.mark.parametrize("gate", ["off", "warmup", "skip", "no_spec"])
def test_gate_does_not_copy(api, gate):
    r = runner(api, capability=gate != "off")
    r._skip_draft_proposal = gate == "skip"
    r.speculative_config = gate != "no_spec"
    indices = torch.tensor([0]) if gate == "final" else torch.empty(0)
    assert api.prepare(r, ["r"], torch.tensor([[10]]), indices, SimpleNamespace(num_scheduled_tokens={"r": 1}),
                       gate == "warmup") is None
    assert not api.copies


def test_reject_invalid_rows_and_prefix(api):
    r = runner(api)
    args = (torch.tensor([[10]]), torch.empty(0), SimpleNamespace(num_scheduled_tokens={"r": 1}), False)
    with pytest.raises(ValueError, match="rectangular"):
        api.prepare(r, ["r", "s"], *args)
    r.use_prefix_caching = True
    with pytest.raises(NotImplementedError, match="prefix"):
        api.prepare(r, ["r"], *args)


@pytest.mark.parametrize("computed,query,prompt", [(0, 0, 7), (-1, 2, 7), (0, 5, 7), (4, 3, 7)])
def test_reject_invalid_or_final_chunk(api, computed, query, prompt):
    with pytest.raises(ValueError, match="intermediate"):
        api.shift(torch.zeros(1, 4), torch.arange(7), computed, query, prompt)
    assert not api.copies


def test_cache_only_rejects_non_mtp(api):
    with pytest.raises(ValueError, match="only for MTP"):
        api.fill(SimpleNamespace(method="eagle"), None, None, None, None)


def test_three_chunks_only_final_samples(api):

    class Draft:

        def __init__(self):
            self.calls = []
            self.logits = 0

        def __call__(self, **kw):
            self.calls.append(kw)
            return kw["hidden_states"]

        def compute_logits(self, hidden):
            self.logits += 1
            return torch.tensor([[0., 1.]])

    draft = Draft()
    proposer = SimpleNamespace(method="mtp", model=draft, num_speculative_tokens=1)
    r = runner(api, wrapped=False)
    metadata = object()
    for computed, query in [(0, 3), (3, 3)]:
        r.input_batch.num_computed_tokens_cpu[0] = computed
        ids = torch.full((1, 4), -1, dtype=torch.int32)
        ids[0, :query] = torch.arange(10 + computed, 10 + computed + query)
        shifted = api.prepare(r, ["r"], ids, torch.empty(0), SimpleNamespace(num_scheduled_tokens={"r": query}), False)
        hidden = torch.zeros(1, 4, 2)
        assert api.fill(proposer, shifted["token_ids"], ids, hidden, metadata) is None
        assert draft.calls[-1]["hidden_states"] is hidden
        assert draft.calls[-1]["attn_metadata"] is metadata
    assert len(draft.calls) == 2 and draft.logits == 0
    assert [c["input_ids"].tolist() for c in draft.calls] == [[[11, 12, 13, -1]], [[14, 15, 16, -1]]]
    # Real final-prefill runner method must keep length-one tokens rank-safe
    # and substitute the sampled boundary, not a nonexistent prompt token.
    final = load(RUNNER, "propose_eagle_prefill", {
        "torch": torch,
        "Optional": Optional,
        "async_h2d_copy": lambda values, **kw: torch.tensor(values, **kw)
    })
    proposer.propose = lambda *args: api.propose(proposer, *args)
    r.drafter = proposer
    r.use_aux_hidden_state_outputs = False
    r._get_attention_group_id_for_hybrid = lambda: 0
    r.input_batch.block_table = [SimpleNamespace(get_cpu_tensor=lambda: torch.zeros(1, 1))]
    r.input_batch.num_computed_tokens_cpu[0] = 6
    plan = api.prepare(r, ["r"],
                       torch.tensor([[16]]),
                       torch.tensor([0]),
                       SimpleNamespace(num_scheduled_tokens={"r": 1}),
                       False,
                       logits_requests=["r"])
    result = final(r, [torch.tensor([99])], [torch.zeros(1, 1, 2)],
                   None,
                   0,
                   torch.tensor([[16]]),
                   torch.tensor([[6]]),
                   metadata,
                   torch.tensor([0]),
                   0,
                   mtp_prompt_batch=plan)
    assert result.tolist() == [[1]]
    assert len(draft.calls) == 3 and draft.logits == 1
    assert draft.calls[-1]["input_ids"].tolist() == [[99]]


@pytest.mark.parametrize("finishes", [(False, False), (True, True), (True, False), (False, True)])
@pytest.mark.parametrize("queries", [(3, 2), (1, 1)])
def test_rectangular_prompt_rows(api, finishes, queries):
    r = runner(api)
    ids = torch.full((3, 4), -1, dtype=torch.int32)
    names = ["a", "b"]
    computed = [0, 2]
    prompts = [c + q + (0 if f else 2) for c, q, f in zip(computed, queries, finishes)]
    history = torch.stack([torch.arange(10, 20), torch.arange(30, 40)])
    for row, (c, q) in enumerate(zip(computed, queries)):
        ids[row, :q] = history[row, c:c + q]
    r.requests = {rid: SimpleNamespace(output_token_ids=[]) for rid in names}
    # Include an earlier decode and deliberately reverse CPU request order.
    r.input_batch.req_ids = ["decode", "b", "a"]
    r.input_batch.req_id_to_index = {"decode": 0, "b": 1, "a": 2}
    r.input_batch.num_computed_tokens_cpu = [7, computed[1], computed[0]]
    r.input_batch.num_prompt_tokens = [7, prompts[1], prompts[0]]
    r.input_batch.token_ids_cpu = torch.stack([history[0], history[1], history[0]])
    table = torch.tensor([[900], [200], [100]])
    r.input_batch.block_table = [SimpleNamespace(get_cpu_tensor=lambda: table)]
    r._get_attention_group_id_for_hybrid = lambda: 0
    r.use_aux_hidden_state_outputs = False
    finishing = [rid for rid, f in zip(names, finishes) if f]
    # Reverse sampler order to prove identity mapping, not batch offsets.
    finishing.reverse()
    indices = torch.tensor([names.index(rid) * 4 + queries[names.index(rid)] - 1
                            for rid in finishing] + ([-1] if finishing else []))
    plan = api.prepare(r,
                       names,
                       ids,
                       indices,
                       SimpleNamespace(num_scheduled_tokens=dict(zip(names, queries))),
                       False,
                       logits_requests=finishing)
    calls = []
    hidden = torch.arange(24).reshape(12, 2)
    if finishing:
        # Mutation after snapshot must not trigger a recomputation guard.
        for rid in finishing:
            r.requests[rid].output_token_ids.append(88)
        r.drafter.propose = lambda *args: calls.append(args) or torch.ones(len(finishing), 2)
        final = load(RUNNER, "propose_eagle_prefill", {
            "torch": torch,
            "Optional": Optional,
            "async_h2d_copy": lambda values, **kw: torch.tensor(values, **kw)
        })
        result = final(r, [torch.tensor([80 + i for i in range(len(finishing))] + [-999])], [hidden],
                       None,
                       0,
                       ids,
                       torch.zeros_like(ids),
                       object(),
                       indices,
                       1,
                       mtp_prompt_batch=plan)
        assert result.shape == (len(finishing), 2)
        shifted, _, hidden_arg, last, _, blocks, _ = calls[0]
        assert blocks.tolist() == [[100 if rid == "a" else 200] for rid in finishing]
        assert last.tolist() == indices[:len(finishing)].tolist()
        torch.testing.assert_close(hidden_arg, hidden.reshape(3, 4, 2))
    else:
        shifted = plan["token_ids"]
        proposer = SimpleNamespace(method="mtp", model=lambda **kw: calls.append(kw))
        api.fill(proposer, shifted, torch.zeros_like(ids), hidden.reshape(3, 4, 2), object())
    assert len(calls) == 1
    for row, (rid, c, q, f) in enumerate(zip(names, computed, queries, finishes)):
        assert shifted[row, :q - 1].tolist() == history[row, c + 1:c + q].tolist()
        boundary = 80 + finishing.index(rid) if f else history[row, c + q].item()
        assert shifted[row, q - 1].item() == boundary
        assert shifted[row, q:].tolist() == ids[row, q:].tolist()
    assert shifted[2].tolist() == [-1] * 4


@pytest.mark.parametrize("invalid", ["merged", "recompute", "missing", "duplicate", "overflow"])
def test_snapshot_rejects_unsupported_or_invalid(api, invalid):
    r = runner(api)
    r.use_merged_prefill = invalid == "merged"
    if invalid == "recompute":
        r.requests["r"].output_token_ids.append(42)
    query = 8 if invalid == "overflow" else 7
    requests = [] if invalid == "missing" else ["r", "r"] if invalid == "duplicate" else ["r"]
    with pytest.raises((ValueError, NotImplementedError)):
        api.prepare(r, ["r"],
                    torch.zeros(1, 7),
                    torch.tensor([6, -1]),
                    SimpleNamespace(num_scheduled_tokens={"r": query}),
                    False,
                    logits_requests=requests)


def test_draft_ids_are_emitted_row_ids_and_clear_together():
    take = load(RUNNER, "take_draft_token_ids", {
        "torch": torch,
        "Optional": Optional,
        "DraftTokenIds": lambda ids, tokens: (ids, tokens)
    })
    r = SimpleNamespace(_draft_token_ids=torch.tensor([[1], [2]]),
                        _draft_req_ids=["decode", "finish"],
                        input_batch=SimpleNamespace(req_ids=["decode", "intermediate", "finish"]))
    assert take(r) == (["decode", "finish"], [[1], [2]])
    assert r._draft_token_ids is None and r._draft_req_ids is None
    assert take(r) is None
    r._draft_token_ids = [[1]]
    r._draft_req_ids = ["a", "b"]
    with pytest.raises(AssertionError, match="emitted rows"):
        take(r)


def test_proposal_routes_decode_intermediate_and_finishing_batches():

    class Eagle:
        pass

    ns = dict(torch=torch,
              Optional=Optional,
              Union=Union,
              SamplingMetadata=object,
              PrefillInputData=object,
              DecodeInputData=object,
              EagleProposer=Eagle,
              shallow_tuple=lambda data: data)
    propose = load(RUNNER, "propose_draft_token_ids", ns)
    calls = []
    r = SimpleNamespace(speculative_config=SimpleNamespace(method="mtp",
                                                           use_eagle=lambda: True,
                                                           num_speculative_tokens=2),
                        drafter=Eagle(),
                        input_batch=SimpleNamespace(req_ids=["decode", "intermediate", "finish"]),
                        propose_eagle_decode=lambda *a: torch.tensor([[10, 11]]),
                        propose_eagle_prefill=lambda *a, **kw: calls.append((a, kw)) or torch.tensor([[20, 21]]))
    batch_data = ([['intermediate'],
                   ['finish']], [None, None], [None, None], [None,
                                                             None], [None,
                                                                     None], [torch.empty(0),
                                                                             torch.tensor([0, -1])], [[], ['finish']])
    plans = [object(), object()]
    result = propose(r,
                     None, [],
                     None,
                     None,
                     None,
                     None,
                     prefill_sampled_token_ids_tensor=[torch.tensor([8])],
                     hidden_states_prefills=[None, None],
                     num_decodes=1,
                     prefill_data=batch_data,
                     decode_data=object(),
                     prefill_batches_with_logits=[1],
                     mtp_prompt_batches=plans)
    assert result.tolist() == [[10, 11], [20, 21]]
    assert r._draft_req_ids == ["decode", "finish"]
    assert len(calls) == 1
    assert calls[0][1] == {"sampled_idx": 0, "mtp_prompt_batch": plans[1]}


def test_skip_or_no_sampling_clears_both_draft_fields():
    # Execute the actual branch in isolation: sample_tokens needs HPU runtime.
    tree = ast.parse(RUNNER.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    branch = next(n for n in method.body
                  if isinstance(n, ast.If) and "sampling_metadata is None" in ast.unparse(n.test))
    clear_branch = ast.If(test=branch.test, body=branch.body, orelse=[])
    code = compile(ast.fix_missing_locations(ast.Module(body=[clear_branch], type_ignores=[])), str(RUNNER), "exec")
    for metadata, skip in [(None, False), (object(), True)]:
        r = SimpleNamespace(speculative_config=True,
                            _skip_draft_proposal=skip,
                            _draft_token_ids=[[1]],
                            _draft_req_ids=["stale"])
        exec(code, {"self": r, "sampling_metadata": metadata})
        assert r._draft_token_ids is None and r._draft_req_ids is None
