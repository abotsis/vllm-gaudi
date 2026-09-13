# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests; source extraction avoids importing the HPU plugin."""
import ast
import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
MODEL = ROOT / "vllm_gaudi/models/glm5_next.py"
NAME = "vllm_gaudi.v1.worker.layer_diagnostic"
spec = importlib.util.spec_from_file_location(NAME, ROOT / "vllm_gaudi/v1/worker/layer_diagnostic.py")
assert spec is not None and spec.loader is not None
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def fixture(tmp_path):
    tmp_path.chmod(0o700)
    return SimpleNamespace(_diag_layer_disabled_at_boot=False,
                           _diag_sampler_directory=lambda: str(tmp_path),
                           requests={},
                           input_batch=SimpleNamespace(req_id_to_index={}),
                           use_merged_prefill=False)


def ticket(runner, path, ids, position=0, prompt=False):
    for row, rid in enumerate(ids):
        runner.requests[rid] = SimpleNamespace(prompt_token_ids=[1, 8],
                                               output_token_ids=[8] * position,
                                               num_computed_tokens=1 + position)
        runner.input_batch.req_id_to_index[rid] = row
    width = 2 if prompt else 1
    tokens = torch.full((len(ids) + 1, width), 8)
    positions = torch.full_like(tokens, 1 + position)
    mapping = torch.arange(len(ids)) * width + width - 1
    return diag.prepare(runner, str(path), (ids, ids), tokens, positions, SimpleNamespace(is_prompt=prompt), mapping,
                        lambda: None, 7, True)


def test_gate(tmp_path):
    runner = fixture(tmp_path)
    assert diag.directory(runner) is None
    gate = tmp_path / "LAYER_ENABLED"
    gate.touch(mode=0o600)
    assert diag.directory(runner) == str(tmp_path)
    gate.chmod(0o644)
    assert diag.directory(runner) is None
    gate.unlink()
    gate.symlink_to(tmp_path / "missing")
    assert diag.directory(runner) is None
    gate.unlink()
    gate.touch(mode=0o600)
    runner._diag_layer_disabled_at_boot = True
    assert diag.directory(runner) is None


def test_bounds_owned_prefill_and_independence(tmp_path):
    runner = fixture(tmp_path)
    runner._diag_state_calls = runner._diag_sampler_calls = 999
    for ids in [["serial"], ["a", "b"]]:
        for position in range(4):
            selected = ticket(runner, tmp_path, ids, position, prompt=position == 0)
            if position == 3:
                assert selected is None
                continue
            values = torch.arange(48.).reshape(6, 2, 4)
            expected = values[selected["indices"]].clone()
            with diag.collecting(selected):
                diag.boundary("initial_streams", values)
                values.zero_()
            assert diag._collector is None
            data = torch.load(tmp_path / f"layer-rank7-{runner._diag_layer_calls - 1:02d}.pt", weights_only=True)
            assert torch.equal(data["boundaries"][0]["rows"], expected)
            assert data["baseline_graph_eligible"] and data["diagnostic_bypass"]
    assert runner._diag_layer_admitted == ["serial", "a", "b"]
    assert len(runner._diag_layer_seen) == 9
    assert runner._diag_layer_calls == 6
    assert ticket(runner, tmp_path, ["fourth"]) is None
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in tmp_path.iterdir())


def test_cap_off_and_finally(tmp_path, monkeypatch):
    # Off path does not even inspect tensor metadata.
    diag.boundary("off", object())
    runner = fixture(tmp_path)
    selected = ticket(runner, tmp_path, ["a"])
    runner._diag_layer_bytes = diag.MAX_BYTES
    with diag.collecting(selected):
        diag.boundary("capped", torch.ones(2, 4))
        assert selected["payload"]["truncated"]
        assert not selected["payload"]["boundaries"]
    selected = ticket(fixture(tmp_path), tmp_path, ["a"])
    with pytest.raises(RuntimeError, match="forward"), diag.collecting(selected):
        raise RuntimeError("forward")
    assert diag._collector is None
    with pytest.raises(FileExistsError), diag.collecting(selected):
        pass
    assert diag._collector is None


def test_actual_generic_single_forward_and_graph_policy(tmp_path, monkeypatch):
    tree = ast.parse(RUNNER.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_model_generic")
    forward_calls = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "forward"
    ]
    assert len(forward_calls) == 1
    flags, calls = [], []
    monkeypatch.setitem(sys.modules, NAME, diag)
    monkeypatch.setitem(sys.modules, "vllm_gaudi.ops.causal_conv1d_pytorch",
                        SimpleNamespace(set_conv_pool_hpu_graphs_active=flags.append))
    namespace = dict(contextlib=contextlib,
                     os=__import__("os"),
                     htorch=SimpleNamespace(utils=SimpleNamespace(internal=SimpleNamespace(is_lazy=lambda: True)),
                                            core=SimpleNamespace(mark_step=lambda: None)),
                     trim_attn_metadata=lambda x: x,
                     get_tensor_model_parallel_rank=lambda: 0,
                     LoraMask=SimpleNamespace(setLoraMask=lambda x: None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    runner = fixture(tmp_path)
    runner._seq_len = runner._num_blocks = lambda x: 1
    runner._check_config = runner._diag_state_before = lambda *a: None
    runner._use_graphs = lambda *a: True
    runner.model_has_chunked_attention = runner.is_driver_worker = runner.use_aux_hidden_state_outputs = False
    runner._decode_tensor_cache = True
    runner.profiler = SimpleNamespace(record_event=lambda *a: contextlib.nullcontext())
    runner.requests["a"] = SimpleNamespace(prompt_token_ids=[1, 8], output_token_ids=[], num_computed_tokens=1)
    runner.input_batch.req_id_to_index["a"] = 0

    def forward(**kwargs):
        calls.append((kwargs["bypass_hpu_graphs"], diag._collector is not None))
        value = torch.ones(1, 4)
        diag.boundary("final_norm", value)
        return value

    runner.model = SimpleNamespace(forward=forward, compute_logits=lambda x: x)
    args = (runner, torch.tensor([[8]]), torch.tensor([[1]]), SimpleNamespace(is_prompt=False), torch.tensor([0]), None,
            None, None)
    run = namespace[node.name]
    run(*args, diag_state_context=(["a"], ["a"]))
    (tmp_path / "LAYER_ENABLED").touch(mode=0o600)
    run(*args, diag_state_context=(["a"], ["a"]))
    run(*args, diag_state_context=(["a"], ["a"]))  # duplicate position is not selected
    assert calls == [(False, False), (True, True), (False, False)]
    assert flags == [True, True, True]
    assert diag._collector is None
    assert namespace["os"].environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"] == "0"


@pytest.mark.parametrize("mapping,token,position", [([1], 8, 1), ([0], 9, 1), ([0], 8, 2)])
def test_bad_mapping_history_never_installs_collector(tmp_path, mapping, token, position):
    runner = fixture(tmp_path)
    runner.requests["a"] = SimpleNamespace(prompt_token_ids=[1, 8], output_token_ids=[], num_computed_tokens=1)
    runner.input_batch.req_id_to_index["a"] = 0
    with pytest.raises(ValueError):
        diag.prepare(runner, str(tmp_path), (["a"], ["a"]), torch.tensor([[token], [0]]), torch.tensor([[position],
                                                                                                        [0]]),
                     SimpleNamespace(is_prompt=False), torch.tensor(mapping), lambda: None, 0, True)
    assert diag._collector is None
    assert runner._diag_layer_calls == 1


def test_model_boundary_placement():
    tree = ast.parse(MODEL.read_text())
    for name, expected in [("HpuGlm5NextDecoderLayer", 7), ("HpuGlm5NextModel", 2)]:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        hooks = [
            n for n in ast.walk(forward)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_diag_layer_boundary"
        ]
        assert len(hooks) == expected
        for index, statement in enumerate(forward.body):
            if isinstance(statement, ast.Expr) and statement.value in hooks:
                assert isinstance(forward.body[index - 1], ast.Assign)


def test_actual_mhc_pre_outputs_have_token_leading_axis():
    tree = ast.parse(MODEL.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mhc_pre_hpu")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), "exec"), namespace)
    post, comb, value = namespace[node.name](torch.ones(6, 4, 8, dtype=torch.bfloat16), torch.zeros(24, 32),
                                             torch.ones(3), torch.zeros(24), 1e-6, 1e-6, 1e-6, 2.0, 2)
    assert post.shape == (6, 4, 1)
    assert comb.shape == (6, 4, 4)
    assert value.shape == (6, 8)


def decoder_forward(namespace):
    tree = ast.parse(MODEL.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HpuGlm5NextDecoderLayer")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), "exec"), namespace)
    return cls, node, namespace["forward"]


def test_split_hooks_are_layer0_only_and_bare_tensors():
    cls, forward, _ = decoder_forward({})
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "self._diag_split_layer0 = layer_idx == 0" in ast.unparse(init)
    split_hooks = []
    for statement in forward.body:
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "self._diag_split_layer0":
            for call in (n for n in ast.walk(statement) if isinstance(n, ast.Call)):
                assert isinstance(call.func, ast.Name)
                assert call.func.id in ("_diag_layer_boundary", "_diag_layer_unavailable")
                if call.func.id == "_diag_layer_boundary":
                    assert isinstance(call.args[1], ast.Name)  # no view/cast/copy at call site
                    split_hooks.append(call.args[1].id)
    assert split_hooks == ["x", "post_a", "comb_a", "x", "attn_out"]
    # No dynamic module hooks or wrapper monkeypatching in the extension.
    assert "register_forward" not in MODEL.read_text()


@pytest.mark.parametrize("layer0", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_split_forward_rows_math_and_inactive_no_copies(tmp_path, monkeypatch, layer0, active):
    namespace = dict(_diag_layer_boundary=diag.boundary, _diag_attention_hooks=diag.attention_hooks)
    _, _, forward = decoder_forward(namespace)
    streams = torch.arange(48.).reshape(6, 2, 4)
    original = streams.clone()
    post = torch.arange(12.).reshape(6, 2, 1)
    comb = torch.arange(24.).reshape(6, 2, 2)
    x = torch.arange(24.).reshape(6, 4)
    norm, raw = x + 10, x + 20
    final = streams + 100
    calls = []

    def mhc_post(value, residual, post_mix, comb_mix):
        calls.append(value)
        return final

    namespace["_mhc_post"] = mhc_post
    model = SimpleNamespace(_prefix=f"model.layers.{0 if layer0 else 1}",
                            _diag_split_layer0=layer0,
                            _is_linear=False,
                            attn_hc=lambda _: (post, comb, x),
                            input_layernorm=lambda _: norm,
                            self_attn=lambda **_: raw,
                            ffn_hc=lambda _: (post, comb, x),
                            post_attention_layernorm=lambda _: x,
                            mlp=lambda _: x)
    selected = ticket(fixture(tmp_path), tmp_path, ["a", "b"], prompt=True)
    if active:
        with diag.collecting(selected):
            result = forward(model, None, streams)
        payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
        boundaries = payload["boundaries"]
        assert len(boundaries) == (7 if layer0 else 2)
        if layer0:
            expected = [x, post, comb, norm, raw]
            for item, source in zip(boundaries[:5], expected):
                assert item["shape"] == tuple(source.shape)
                assert torch.equal(item["rows"], source[[1, 3]])
            missing = payload["unavailable_boundaries"]
            assert missing[0]["name"].endswith(".attention")
            assert "identity unavailable" in missing[0]["reason"]
        else:
            assert "unavailable_boundaries" not in payload
    else:

        def forbidden(*args, **kwargs):
            raise AssertionError("inactive tensor operation")

        with monkeypatch.context() as patch:
            for method in ("clone", "detach", "to", "index_select", "numel", "element_size"):
                patch.setattr(torch.Tensor, method, forbidden)
            result = forward(model, None, streams)
            diag.unavailable("off", object())
    assert result is final
    assert calls[0] is raw and calls[1] is x
    assert torch.equal(streams, original)
    assert diag._collector is None
