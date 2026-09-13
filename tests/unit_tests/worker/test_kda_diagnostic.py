# SPDX-License-Identifier: Apache-2.0
"""CPU-only KDA layer-0 diagnostic contract tests, without importing the HPU plugin."""
import ast
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.worker.test_layer_diagnostic_ast import MODEL, diag, fixture, ticket


class Projection(torch.nn.Module):

    def forward(self, value):
        return value + 1, torch.ones(value.shape[-1])


class HpuGlm5NextKdaAttention(torch.nn.Module):
    """Same class name and module attributes as the production layer; toy math."""

    def __init__(self):
        super().__init__()
        for name in diag.KDA_MODULES:
            setattr(self, name, Projection())

    def forward(self, x, pool, slots, states, initial_state=None):
        mixed, _ = self.qkv_proj(x)
        g = self.f_b_proj(self.f_a_proj(x)[0])[0]
        beta = self.b_proj(x)[0]
        diag.kda_boundary(self, "forget_gate", g)
        diag.kda_boundary(self, "beta", beta)
        diag.kda_pool_rows(self, "conv_pool_in", pool, slots)
        pool.index_copy_(0, slots, mixed)
        diag.kda_pool_rows(self, "conv_pool_out", pool, slots)
        diag.kda_boundary(self, "conv_out", mixed)
        diag.kda_boundary(self, "ssm_state_in", states)
        states += 1  # in-place, like kda_decode_step
        diag.kda_boundary(self, "ssm_state_out", states)
        diag.kda_sequence_boundary(self, "final_state", initial_state)
        gate = self.g_b_proj(self.g_a_proj(x)[0])[0]
        diag.kda_boundary(self, "gate", gate)
        return self.o_proj(mixed + gate)[0]


def make_layer(attn, linear=True):
    return SimpleNamespace(_diag_split_layer0=True, _prefix="model.layers.0", _is_linear=linear, self_attn=attn)


@pytest.mark.parametrize("fail", [False, True])
def test_kda_hooks_capture_order_and_cleanup(tmp_path, fail):
    attn = HpuGlm5NextKdaAttention()
    selected = ticket(fixture(tmp_path), tmp_path, ["a", "b"])  # decode: input_shape (3, 1), rows 0 and 1
    x = torch.arange(12.).reshape(3, 4)
    pool = torch.zeros(6, 4)
    slots = torch.tensor([5, 2, 0])
    states = torch.zeros(3, 2, 2)
    with diag.collecting(selected):
        try:
            with diag.attention_hooks(make_layer(attn)):
                out = attn(x, pool, slots, states)
                assert torch.equal(out, (x + 1) + (x + 2) + 1)
                diag.kda_boundary(object(), "wrong_owner", x)
                diag.kda_boundary(attn, "wrong_shape", torch.ones(7, 4))
                if fail:
                    raise RuntimeError("inference failure")
        except RuntimeError:
            assert fail
        for name in diag.KDA_MODULES:
            assert not getattr(attn, name)._forward_hooks
        assert not attn.o_proj._forward_pre_hooks
        assert "kda_owner" not in selected and "kda_prefix" not in selected
        count = len(selected["payload"]["boundaries"])
        attn(x, pool.clone(), slots, states.clone())
        assert len(selected["payload"]["boundaries"]) == count
    payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
    names = [b["name"] for b in payload["boundaries"]]
    assert names == [
        f"model.layers.0.kda.{n}"
        for n in ("qkv_proj", "f_a_proj", "f_b_proj", "b_proj", "forget_gate", "beta", "conv_pool_in", "conv_pool_out",
                  "conv_out", "ssm_state_in", "ssm_state_out", "g_a_proj", "g_b_proj", "gate", "pre_o_proj", "o_proj")
    ]
    by = {b["name"].split(".")[-1]: b for b in payload["boundaries"]}
    assert torch.equal(by["qkv_proj"]["rows"], (x + 1)[[0, 1]])
    assert by["conv_pool_in"]["row_indices"] == [5, 2] and torch.equal(by["conv_pool_in"]["rows"], torch.zeros(2, 4))
    assert torch.equal(by["conv_pool_out"]["rows"], (x + 1)[[0, 1]])
    # in-place state update must not leak into the earlier snapshot
    assert torch.equal(by["ssm_state_in"]["rows"], torch.zeros(2, 2, 2))
    assert torch.equal(by["ssm_state_out"]["rows"], torch.ones(2, 2, 2))
    assert torch.equal(by["pre_o_proj"]["rows"], ((x + 1) + (x + 2))[[0, 1]])
    assert payload["attention_modules"]["kind"] == "kda"
    reasons = {u["name"].split(".")[-1]: u["reason"] for u in payload["unavailable_boundaries"]}
    assert "final_state" in reasons  # None is not a tensor
    assert "wrong_shape" in reasons and "wrong_owner" not in reasons


def test_kda_sequence_rows_and_pad_slots(tmp_path):
    attn = HpuGlm5NextKdaAttention()
    selected = ticket(fixture(tmp_path), tmp_path, ["a"], prompt=True)  # prefill: input_shape (2, 2), token row 1
    with diag.collecting(selected), diag.attention_hooks(make_layer(attn)):
        diag.kda_sequence_boundary(attn, "final_state", torch.arange(8.).reshape(2, 4))
        diag.kda_sequence_boundary(attn, "bad_state", torch.arange(4.).reshape(1, 4))
        diag.kda_pool_rows(attn, "pad", torch.zeros(4, 2), torch.tensor([-1, 3]))
        diag.kda_pool_rows(attn, "short", torch.zeros(4, 2), torch.tensor([], dtype=torch.long))
    payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
    (state, ) = payload["boundaries"]
    assert state["row_indices"] == [0] and torch.equal(state["rows"], torch.arange(4.).reshape(1, 4))
    reasons = {u["name"].split(".")[-1]: u["reason"] for u in payload["unavailable_boundaries"]}
    assert "batch axis" in reasons["bad_state"]
    assert "pad" in reasons["pad"]
    assert "not proven" in reasons["short"]


def test_linear_layer_without_kda_identity_is_unavailable(tmp_path):
    selected = ticket(fixture(tmp_path), tmp_path, ["a"])
    with diag.collecting(selected), diag.attention_hooks(make_layer(torch.nn.Identity())):
        pass
    payload = torch.load(tmp_path / "layer-rank7-00.pt", weights_only=True)
    assert payload["unavailable_boundaries"] == [
        dict(name="model.layers.0.kda", reason="Exact layer-0 KDA module identity unavailable")
    ]


def test_inactive_kda_probes_do_not_inspect():
    diag.kda_boundary(object(), "off", object())
    diag.kda_sequence_boundary(object(), "off", object())
    diag.kda_pool_rows(object(), "off", object(), object())


def test_model_forward_instruments_every_kda_stage():
    tree = ast.parse(MODEL.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "HpuGlm5NextKdaAttention")
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_orig")
    calls = {}
    for node in ast.walk(fwd):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("_diag_kda"):
            assert isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
            calls.setdefault(node.func.id, []).append(node.args[1].value)
    assert set(calls["_diag_kda"]) == {
        "forget_gate", "beta", "conv_out", "ssm_state_in", "kda_out", "ssm_state_out", "core", "gate"
    }
    assert calls["_diag_kda_seq"] == ["ssm_state_in", "ssm_state_out"]
    assert calls["_diag_kda_pool"] == ["conv_pool_in", "conv_pool_out"]
    # both prefill and decode branches record conv_out
    assert calls["_diag_kda"].count("conv_out") == 2


def test_moe_forward_records_router_logits_and_bias():
    tree = ast.parse(MODEL.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "HpuGlm5NextMoE")
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = [
        n for n in ast.walk(fwd)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_diag_layer_boundary"
    ]
    names = {}
    for call in calls:
        # name is self._prefix + ".<suffix>"
        assert isinstance(call.args[0], ast.BinOp)
        names[call.args[0].right.value] = call
    assert set(names) == {".router_logits", ".router_bias"}
    bias = names[".router_bias"]
    rows = next(kw for kw in bias.keywords if kw.arg == "row_indices")
    assert [c.value for c in rows.value.elts] == [0]
    assert not names[".router_logits"].keywords
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "self._prefix = prefix" for n in ast.walk(init))
