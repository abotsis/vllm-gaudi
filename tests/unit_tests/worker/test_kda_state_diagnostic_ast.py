# SPDX-License-Identifier: Apache-2.0
"""CPU source tests: no vLLM/plugin import and no accelerator initialization."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
MODEL = ROOT / "vllm_gaudi/models/glm5_next.py"
spec = importlib.util.spec_from_file_location("state_diag", ROOT / "vllm_gaudi/v1/worker/kda_state_diagnostic.py")
assert spec is not None and spec.loader is not None
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("shape", [(5, ), (4, 5), (4, 5, 3)])
def test_actual_resolver(shape):
    actual = source_function(MODEL, "_resolve_state_indices", {})
    values = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    group = torch.tensor(3)
    expected = actual(SimpleNamespace(cache_group_idx=group), SimpleNamespace(load_indices_tensor=values))
    result = diag.resolve(values, group)
    assert torch.equal(expected, result)
    assert result.is_contiguous()


def test_trim_and_processor_passthrough():
    tree = ast.parse(RUNNER.read_text())
    trim = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "trim_attn_metadata")
    fields = {n.value for n in ast.walk(trim) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert set(diag.FIELDS) <= fields
    processor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HPUAttentionMetadataProcessor")
    writes = set()
    for node in ast.walk(processor):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "custom_tuple_replace":
            assert all(k.arg is not None for k in node.keywords)
            writes.update(k.arg for k in node.keywords)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr not in diag.FIELDS
    assert writes
    assert writes.isdisjoint(diag.FIELDS)
    generic = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_model_generic")
    calls = [n for n in ast.walk(generic) if isinstance(n, ast.Call)]
    forward = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "forward"]
    capture = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "_diag_state_before")
    trimming = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "trim_attn_metadata")
    assert len(forward) == 1
    assert trimming.lineno < capture.lineno < forward[0].lineno


def fixture_runner(directory):
    cls = type("HpuGlm5NextKdaAttention", (SimpleNamespace, ), {"__module__": "vllm_gaudi.models.glm5_next"})
    module = cls()
    module.cache_group_idx = torch.tensor(2)
    module.kv_cache = [torch.arange(24.).reshape(6, 4), torch.arange(48.).reshape(6, 2, 4)]
    runner = SimpleNamespace(_diag_state_disabled_at_boot=False,
                             _diag_sampler_directory=lambda: str(directory),
                             use_merged_prefill=False,
                             requests={},
                             input_batch=SimpleNamespace(req_id_to_index={}),
                             model=SimpleNamespace(named_modules=lambda: [("model.layers.0.attn", module)]))
    metadata = SimpleNamespace(is_prompt=False,
                               load_indices_tensor=torch.tensor([[0, 0, -1], [1, 1, -1], [2, 3, -1]]),
                               store_indices_tensor=torch.tensor([[0, 0, -1], [1, 1, -1], [4, 5, -1]]))
    return runner, module, metadata


def request(runner, rid, row, position):
    runner.requests[rid] = SimpleNamespace(output_token_ids=[8] * position,
                                           prompt_token_ids=[1, 8],
                                           num_computed_tokens=1 + position)
    runner.input_batch.req_id_to_index[rid] = row


def test_gate(tmp_path):
    runner, _, _ = fixture_runner(tmp_path)
    assert diag.state_directory(runner) is None
    gate = tmp_path / "STATE_ENABLED"
    gate.touch(mode=0o600)
    assert diag.state_directory(runner) == str(tmp_path)
    gate.chmod(0o644)
    assert diag.state_directory(runner) is None
    gate.unlink()
    gate.symlink_to(tmp_path / "missing")
    assert diag.state_directory(runner) is None
    gate.unlink()
    gate.touch(mode=0o600)
    runner._diag_state_disabled_at_boot = True
    assert diag.state_directory(runner) is None


def test_serial_then_concurrent_owned_and_bounded(tmp_path):
    tmp_path.chmod(0o700)
    runner, module, metadata = fixture_runner(tmp_path)
    steps = []
    for ids in [["serial"], ["a", "b"]]:
        for position in range(4):
            for row, rid in enumerate(ids):
                request(runner, rid, row, position)
            diag.capture(runner,
                         str(tmp_path), (ids, ids),
                         torch.tensor([[8], [8], [0]]),
                         torch.full((3, 1), 1 + position),
                         metadata,
                         torch.arange(3),
                         lambda: steps.append(1),
                         rank=7)
    files = sorted(tmp_path.glob("state-rank7-*.pt"))
    assert len(files) == 6
    assert len(runner._diag_state_seen) == 9
    assert runner._diag_state_admitted == ["serial", "a", "b"]
    data = torch.load(files[-1], weights_only=True)
    assert data["rank"] == 7
    assert data["metadata"]["load_indices_tensor"].shape == (3, 3)
    pools = data["layers"][0]["pools"]
    assert pools[0]["read_rows"] == [2, 3]
    assert pools[0]["write_rows"] == [2, 3]
    assert pools[1]["write_rows"] == [4, 5]
    raw = pools[0]["raw"].clone()
    module.kv_cache[0].zero_()
    assert torch.equal(raw, pools[0]["raw"])
    assert steps
    assert all(os.stat(f).st_mode & 0o777 == 0o600 for f in files)
    assert runner._diag_state_bytes <= diag.MAX_BYTES


def test_charge_before_work_and_failed_io(tmp_path):
    tmp_path.chmod(0o700)
    runner, _, metadata = fixture_runner(tmp_path)
    request(runner, "a", 0, 0)
    runner._diag_state_bytes = diag.MAX_BYTES
    diag.capture(runner,
                 str(tmp_path), (["a"], ["a"]),
                 torch.ones(3, 1, dtype=torch.long),
                 torch.ones(3, 1, dtype=torch.long),
                 metadata,
                 torch.arange(3),
                 lambda: pytest.fail("cap must precede device work"),
                 rank=0)
    assert runner._diag_state_calls == 1
    assert len(runner._diag_state_seen) == 1
    assert not list(tmp_path.iterdir())
    runner, _, metadata = fixture_runner(tmp_path)
    request(runner, "a", 0, 0)
    existing = tmp_path / "state-rank0-00.pt"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        diag.capture(runner,
                     str(tmp_path), (["a"], ["a"]),
                     torch.full((3, 1), 8, dtype=torch.long),
                     torch.ones(3, 1, dtype=torch.long),
                     metadata,
                     torch.arange(3),
                     lambda: None,
                     rank=0)
    assert existing.read_bytes() == b"keep"
    assert runner._diag_state_bytes > 0
    assert runner._diag_state_calls == 1


def test_prefill_exact_mapping_and_raw_flags(tmp_path):
    tmp_path.chmod(0o700)
    runner, module, metadata = fixture_runner(tmp_path)
    metadata.is_prompt = True
    metadata.has_initial_states_p = torch.tensor([False, False, False])
    request(runner, "a", 0, 0)
    diag.capture(runner,
                 str(tmp_path), (["a"], ["a"]),
                 torch.full((3, 2), 8, dtype=torch.long),
                 torch.tensor([[0, 1], [0, 1], [0, 1]]),
                 metadata,
                 torch.tensor([1]),
                 lambda: None,
                 rank=3)
    data = torch.load(tmp_path / "state-rank3-00.pt", weights_only=True)
    assert data["records"][0]["flat_logit_index"] == 1
    assert not data["metadata"]["has_initial_states_p"].any()
    assert torch.equal(data["layers"][0]["pools"][1]["raw"], module.kv_cache[1][[2]])
