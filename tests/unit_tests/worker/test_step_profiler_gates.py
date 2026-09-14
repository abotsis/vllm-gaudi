# SPDX-License-Identifier: Apache-2.0
"""Step profiler arming: rank-0 only and no stacks by default (CPU, AST-extracted)."""
import ast
import os
from pathlib import Path

WORKER = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/worker/hpu_worker.py"


def _load():
    tree = ast.parse(WORKER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_step_profiler")
    calls = []
    ns = {"os": os, "setup_profiler": lambda warmup, active, with_stack: calls.append((active, with_stack)) or "p"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WORKER), "exec"), ns)
    return ns["setup_step_profiler"], calls


def test_defaults_rank0_only_no_stack(monkeypatch):
    monkeypatch.delenv("VLLM_PROFILER_RANK0_ONLY", raising=False)
    monkeypatch.delenv("VLLM_PROFILER_WITH_STACK", raising=False)
    f, calls = _load()
    assert f(None, 0) is None
    assert f((2, 4), 1) is None and f((2, 4), 7) is None
    assert f((2, 4), 0) == "p" and calls == [(3, False)]


def test_env_opens_all_ranks_and_stacks(monkeypatch):
    monkeypatch.setenv("VLLM_PROFILER_RANK0_ONLY", "0")
    monkeypatch.setenv("VLLM_PROFILER_WITH_STACK", "1")
    f, calls = _load()
    assert f((5, 5), 3) == "p" and calls == [(1, True)]
