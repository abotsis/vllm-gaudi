# SPDX-License-Identifier: Apache-2.0
"""_run_sampling must hand retention code host-owned logprob tensors (CPU-only, AST-extracted)."""
import ast
from pathlib import Path
from types import SimpleNamespace

from vllm.v1.outputs import LogprobsTensors

RUNNER = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/worker/hpu_model_runner.py"


def _load(name):
    tree = ast.parse(RUNNER.read_text())
    node = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    namespace = {"LogprobsTensors": LogprobsTensors}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


class FakeTensor:
    """Device tensor stand-in: records the exact copy request and returns a distinct object."""

    def __init__(self, tag, device="hpu"):
        self.tag = tag
        self.device = SimpleNamespace(type=device)
        self.copies = []

    def to(self, **kwargs):
        self.copies.append(kwargs)
        return FakeTensor(self.tag + "@cpu", device="cpu")


def test_device_logprobs_are_copied_once_blocking_and_fields_preserved():
    own = _load("_own_sampled_logprobs")
    ids, lps, ranks, cu = (FakeTensor(t) for t in ("ids", "lps", "ranks", "cu"))
    out = SimpleNamespace(sampled_token_ids=object(), logprobs_tensors=LogprobsTensors(ids, lps, ranks, [0, 1, 3], cu))
    result = own(None, out)
    assert result is out
    lp = out.logprobs_tensors
    for src, dst in ((ids, lp.logprob_token_ids), (lps, lp.logprobs), (ranks, lp.selected_token_ranks),
                     (cu, lp.cu_num_generated_tokens_tensor)):
        assert dst is not src and dst.device.type == "cpu" and dst.tag == src.tag + "@cpu"
        assert src.copies == [dict(device="cpu", non_blocking=False, copy=True)]
    assert lp.cu_num_generated_tokens == [0, 1, 3]


def test_cpu_none_and_missing_logprobs_pass_through():
    own = _load("_own_sampled_logprobs")
    cpu = LogprobsTensors(FakeTensor("i", "cpu"), FakeTensor("l", "cpu"), FakeTensor("r", "cpu"))
    out = SimpleNamespace(logprobs_tensors=cpu)
    assert own(None, out).logprobs_tensors is cpu and not cpu.logprob_token_ids.copies
    out = SimpleNamespace(logprobs_tensors=None)
    assert own(None, out).logprobs_tensors is None
    bare = SimpleNamespace()
    assert own(None, bare) is bare


def test_run_sampling_returns_owned_output_and_prompt_copies_block():
    tree = ast.parse(RUNNER.read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_sampling")
    returned = [n.value for n in ast.walk(run) if isinstance(n, ast.Return)]
    assert len(returned) == 1
    first = returned[0].elts[0]
    assert isinstance(first, ast.Call) and first.func.attr == "_own_sampled_logprobs"
    prompt = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_prompt_logprobs_dict")
    copies = [
        n for n in ast.walk(prompt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "copy_"
    ]
    assert len(copies) == 3
    for call in copies:
        flags = {kw.arg: kw.value.value for kw in call.keywords}
        assert flags == {"non_blocking": False}
