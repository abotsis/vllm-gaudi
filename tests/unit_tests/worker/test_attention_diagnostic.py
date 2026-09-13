# SPDX-License-Identifier: Apache-2.0
"""CPU-only attention diagnostic contract tests, without importing the HPU plugin."""
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.unit_tests.worker.test_layer_diagnostic_ast import diag, fixture, ticket


class Projection(torch.nn.Module):

    def forward(self, value):
        return value + 1, torch.ones(value.shape[-1])


class Wrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.prefix = "model.layers.0.self_attn"
        self.fused_qkv_a_proj = Projection()
        self.o_proj = Projection()
        self.mla_attn = SimpleNamespace(layer_name=self.prefix + ".attn",
                                        impl=SimpleNamespace(latent_cache_k=torch.nn.Identity()))

    def forward(self, value):
        return self.o_proj(self.fused_qkv_a_proj(value)[0])[0]


@pytest.mark.parametrize("fail", [False, True])
def test_scoped_hooks_finally_and_tuple_rows(tmp_path, fail):
    wrapper = Wrapper()
    layer = SimpleNamespace(_diag_split_layer0=True,
                            _prefix="model.layers.0",
                            _is_linear=False,
                            self_attn=SimpleNamespace(mla_attn=wrapper))
    selected = ticket(fixture(tmp_path), tmp_path, ["a", "b"], prompt=True)
    value = torch.arange(24.).reshape(6, 4)
    original = value.clone()
    with diag.collecting(selected):
        try:
            with diag.attention_hooks(layer):
                result = wrapper(value)
                assert torch.equal(result, original + 2)
                diag.attention_boundary(object(), "wrong_owner", object())
                diag.attention_boundary(wrapper.mla_attn.impl, "wrong_shape", torch.ones(7, 4))
                if fail:
                    raise RuntimeError("inference failure")
        except RuntimeError:
            assert fail
        assert not wrapper.o_proj._forward_hooks
        assert not wrapper.o_proj._forward_pre_hooks
        assert not wrapper.fused_qkv_a_proj._forward_hooks
        assert not wrapper.mla_attn.impl.latent_cache_k._forward_hooks
        assert "attention_impl" not in selected
        count = len(selected["payload"]["boundaries"])
        wrapper(value)
        assert len(selected["payload"]["boundaries"]) == count
    assert torch.equal(value, original)
    payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
    for item, delta in zip(payload["boundaries"], [1, 1, 2]):
        assert torch.equal(item["rows"], (original + delta)[[1, 3]])
    assert "Token-leading" in payload["unavailable_boundaries"][0]["reason"]


def test_inactive_hooks_do_not_inspect_layer():
    with diag.attention_hooks(object()):
        diag.attention_boundary(object(), "off", object())
        diag.attention_metadata(object(), object())
        diag.cache_written(object(), (), None)


def test_selected_physical_cache_rows(tmp_path):
    # Execute the real plain cache implementation, not a mathematical stand-in.
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / "vllm_gaudi/extension/utils.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLLMKVCache")
    namespace: dict[str, Any] = dict(torch=torch,
                                     __name__="vllm_gaudi.extension.utils",
                                     get_config=lambda: SimpleNamespace(use_contiguous_pa=False))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    module = namespace["VLLMKVCache"]()
    selected = ticket(fixture(tmp_path), tmp_path, ["a", "b"], prompt=True)
    values = torch.arange(24.).reshape(6, 4)
    cache = torch.zeros(100, 4)
    slots = torch.tensor([91, 12, 63, 44, 25, 86])
    with diag.collecting(selected):
        selected["attention_prefix"] = "attention"
        handle = module.register_forward_hook(diag.cache_written)
        try:
            module(values, cache, slots)
        finally:
            handle.remove()
        assert selected["runner"]._diag_layer_bytes == 2 * 4 * 4
    payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
    item = payload["boundaries"][0]
    assert item["row_indices"] == [12, 44]
    assert torch.equal(item["rows"], values[[1, 3]])
    assert torch.equal(cache[slots], values)
