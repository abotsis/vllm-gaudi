# SPDX-License-Identifier: Apache-2.0
"""CPU-only execution of the production terminal-MTP helper and proposal guard."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/worker/hpu_model_runner.py"


@pytest.fixture
def production():
    tree = ast.parse(RUNNER.read_text())
    helper = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_can_skip_terminal_mtp_draft")
    sample = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    branch = next(n for n in sample.body
                  if isinstance(n, ast.If) and "sampling_metadata is None" in ast.unparse(n.test))
    # The real guard must run after the sampled tokens have entered request history.
    updates = [
        n for n in ast.walk(sample)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "req_state.output_token_ids.extend"
    ]
    assert updates and all(n.lineno < branch.lineno for n in updates)
    assert "self._can_skip_terminal_mtp_draft(scheduler_output, warmup_mode)" in ast.unparse(branch.test)
    ns = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(RUNNER), "exec"), ns)
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(RUNNER), "exec")
    return ns[helper.name], code, branch


def request(length=1, limit=1, params=True):
    return SimpleNamespace(output_token_ids=[42] * length,
                           sampling_params=SimpleNamespace(max_tokens=limit) if params else None)


def run_guard(production,
              requests,
              *,
              scheduled=None,
              method="mtp",
              asynchronous=False,
              warmup=False,
              metadata=True,
              skip=False,
              forbid_helper=False):
    helper, code, branch = production
    calls = []
    r = SimpleNamespace(requests=requests,
                        use_async_scheduling=asynchronous,
                        speculative_config=SimpleNamespace(method=method) if method else None,
                        _skip_draft_proposal=skip,
                        _draft_token_ids=[[99]],
                        _draft_req_ids=["stale"])

    def propose(*args, **kwargs):
        calls.append((args, kwargs))
        r._draft_req_ids = ["new"]
        return [[7]]

    def forbidden(*args):
        pytest.fail("existing skip/no-sampling precedence must short-circuit the helper")

    r.propose_draft_token_ids = propose
    r._can_skip_terminal_mtp_draft = forbidden if forbid_helper else MethodType(helper, r)
    # Supply the unrelated proposal inputs without importing accelerator runtime.
    ns = {n.id: None for n in ast.walk(branch) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    ns.update(self=r,
              getattr=getattr,
              sampling_metadata=object() if metadata else None,
              scheduler_output=SimpleNamespace(
                  num_scheduled_tokens=dict.fromkeys(requests if scheduled is None else scheduled, 1)),
              warmup_mode=warmup,
              num_prefills=0,
              num_decodes=0)
    exec(code, ns)
    return r, calls


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [1, 3])
def test_finished_only_clears_stale_ids(production, batch, length):
    requests = {str(i): request(length) for i in range(batch)}
    # Unscheduled cached state (and padding) must not inhibit a terminal skip.
    requests["unscheduled"] = request(0)
    r, calls = run_guard(production, requests, scheduled=[str(i) for i in range(batch)])
    assert not calls
    assert r._draft_token_ids is None and r._draft_req_ids is None


@pytest.mark.parametrize("requests", [
    {},
    {
        "below": request(1, 2)
    },
    {
        "intermediate": request(0)
    },
    {
        "finished": request(),
        "intermediate": request(0)
    },
    {
        "intermediate": request(0),
        "finished": request()
    },
    {
        "missing": request(params=False)
    },
    {
        "finished": request(),
        "missing": request(params=False)
    },
    {
        "unbounded": request(limit=None)
    },
])
def test_nonterminal_set_still_proposes(production, requests):
    r, calls = run_guard(production, requests)
    assert len(calls) == 1
    assert r._draft_token_ids == [[7]] and r._draft_req_ids == ["new"]


@pytest.mark.parametrize("options", [
    {
        "asynchronous": True
    },
    {
        "warmup": True
    },
    {
        "method": "ngram"
    },
    {
        "method": "eagle"
    },
])
def test_scope_exclusions_still_propose(production, options):
    r, calls = run_guard(production, {"finished": request()}, **options)
    assert len(calls) == 1
    assert r._draft_token_ids == [[7]]


def test_no_spec_does_not_enter_guard(production):
    r, calls = run_guard(production, {"finished": request()}, method=None, forbid_helper=True)
    assert not calls
    assert r._draft_token_ids == [[99]] and r._draft_req_ids == ["stale"]


@pytest.mark.parametrize("options", [{"metadata": False}, {"skip": True}, {"metadata": False, "skip": True}])
def test_existing_precedence_clears_both_without_helper(production, options):
    r, calls = run_guard(production, {}, forbid_helper=True, **options)
    assert not calls
    assert r._draft_token_ids is None and r._draft_req_ids is None
