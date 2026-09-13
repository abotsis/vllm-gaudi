# SPDX-License-Identifier: Apache-2.0
"""HPU gather_logprobs replacement: correct values, int64 ids (CPU, AST-extracted)."""
import ast
from pathlib import Path

import torch

from vllm.v1.outputs import LogprobsTensors

ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "vllm_gaudi/patches.py"
RUNNER = ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"


def _load(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _count_greater(logprobs, token_logprobs):
    return (logprobs >= token_logprobs).sum(-1)


def test_hpu_gather_logprobs_values_and_int64_ids(monkeypatch):
    import sys
    import types
    # `import vllm.v1.sample.sampler as m` needs the package chain to resolve without importing the real sampler.
    sample_pkg = types.ModuleType("vllm.v1.sample")
    sampler_mod = types.ModuleType("vllm.v1.sample.sampler")
    sampler_mod.batched_count_greater_than = _count_greater
    sample_pkg.sampler = sampler_mod
    import vllm.v1 as v1
    monkeypatch.setattr(v1, "sample", sample_pkg, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample", sample_pkg)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.sampler", sampler_mod)
    gather = _load(PATCHES, "_hpu_gather_logprobs", {"torch": torch})
    logprobs = torch.log_softmax(torch.tensor([[3.0, 1.0, 2.0, 0.0], [0.0, 5.0, 1.0, 2.0]]), dim=-1)
    sampled = torch.tensor([2, 1], dtype=torch.int64)
    out = gather(logprobs, 2, sampled)
    assert isinstance(out, LogprobsTensors)
    assert out.logprob_token_ids.dtype == torch.int64
    assert out.logprob_token_ids.tolist() == [[2, 0, 2], [1, 1, 3]]
    torch.testing.assert_close(out.logprobs[:, 0], logprobs.gather(-1, sampled[:, None]).squeeze(-1))
    assert out.selected_token_ranks.tolist() == [2, 1]


def test_prompt_gather_keeps_int64_ids():
    gather = _load(RUNNER, "_gather_prompt_logprobs_hpu", {"torch": torch, "LogprobsTensors": LogprobsTensors})
    logprobs = torch.log_softmax(torch.tensor([[3.0, 1.0, 2.0, 0.0]]), dim=-1)
    out = gather(None, logprobs, 2, torch.tensor([1], dtype=torch.int64))
    assert out.logprob_token_ids.dtype == torch.int64
    assert out.logprob_token_ids.tolist() == [[1, 0, 2]]
    assert out.selected_token_ranks.tolist() == [3]


def test_no_int32_narrowing_left_in_either_gather():
    for path, name in ((PATCHES, "_hpu_gather_logprobs"), (RUNNER, "_gather_prompt_logprobs_hpu")):
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        casts = [n for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr == "int32"]
        assert not casts, f"{name} still narrows indices to int32"
