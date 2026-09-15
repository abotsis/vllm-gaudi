# SPDX-License-Identifier: Apache-2.0
"""CPU/AST coverage: no vLLM imports or real HPU calls/capture."""

import ast
import contextlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[3] / "vllm_gaudi/models/glm5_next_mtp.py"


class Projection(nn.Linear):

    def forward(self, value):
        return super().forward(value), None


class FlatAttention(nn.Module):

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.invalid = False

    def forward(self, *, positions, hidden_states):
        self.events.append("attention")
        value = hidden_states * 2 + positions.unsqueeze(-1) * 3
        value = value.reshape(-1, hidden_states.shape[-1])
        return value[:-1] if self.invalid else value


@pytest.fixture
def setup(monkeypatch):
    events = []
    names = {
        "_private_copy", "_DraftGraphCore", "_DraftPreAttentionGraphCore", "_DraftPostAttentionGraphCore",
        "HpuGlm5NextMTPModel"
    }
    nodes = [
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    ns = dict(torch=torch,
              nn=nn,
              contextlib=contextlib,
              _DRAFT_ATTN=True,
              _DRAFT_GRAPH=True,
              _SPLIT_GRAPH=True,
              _ATTN_SCALE_FILE="",
              _attn_scale=lambda: 1.0,
              _BYPASS=-1.,
              _PADWRITE=-2.)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)

    def wrap(core, **kwargs):
        assert kwargs == dict(disable_tensor_cache=True, dry_run=False)
        name = "pre" if isinstance(core, ns["_DraftPreAttentionGraphCore"]) else "post"
        core.register_forward_pre_hook(lambda *args: events.append(name))
        return core

    bridge = ModuleType("habana_frameworks.torch")
    bridge.hpu = SimpleNamespace(wrap_in_hpu_graph=wrap)
    core = ModuleType("habana_frameworks.torch.core")
    core.mark_step = lambda: events.append("mark")
    parent = ModuleType("habana_frameworks")
    parent.torch = bridge
    bridge.core = core
    for name, module in [("habana_frameworks", parent), ("habana_frameworks.torch", bridge),
                         ("habana_frameworks.torch.core", core)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch, "hpu", SimpleNamespace(synchronize=lambda: events.append("sync")), raising=False)
    torch.manual_seed(17)
    mtp = nn.Module()
    for name in ("enorm", "hnorm", "input_layernorm", "post_attention_layernorm", "shared_head_norm"):
        setattr(mtp, name, nn.LayerNorm(4))
    mtp.eh_proj = Projection(8, 4)
    mtp.mlp = nn.Linear(4, 4)
    mtp.self_attn = FlatAttention(events)
    # Exercise processed plain tensor attributes and registered buffers too.
    mtp.eh_proj.processed = torch.randn(4)
    mtp.eh_proj.register_buffer("scale", torch.randn(4))
    target = SimpleNamespace(mtp=mtp,
                             model=SimpleNamespace(embed_tokens=nn.Embedding(16, 4)),
                             lm_head=nn.Identity(),
                             logits_processor=None)
    model = ns["HpuGlm5NextMTPModel"](target)
    return model, ns, events


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("position", [0, 7])
def test_split_matches_eager_and_orders_seams(setup, batch, position):
    model, _, events = setup
    ids = torch.arange(batch)
    positions = torch.full((batch, ), position)
    states = torch.randn(batch, 1, 4)
    expected = model._forward(ids.reshape(batch, 1), positions.reshape(batch, 1), states)
    events.clear()
    output = model(ids, positions, states)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == states.shape
    assert events == ["mark", "sync", "pre", "mark", "sync", "attention", "mark", "sync", "post", "mark", "sync"]


def test_ownership(setup):
    model, _, _ = setup
    pre, post = model.pre_attention_graph_core, model.post_attention_graph_core
    assert post.mlp is model.mtp.mlp
    assert model._shares_target_modules is True
    pairs = [(pre.embed_tokens, model.embed_tokens)]
    pairs += [(getattr(pre, name), getattr(model.mtp, name))
              for name in ("enorm", "hnorm", "eh_proj", "input_layernorm")]
    pairs += [(getattr(post, name), getattr(model.mtp, name))
              for name in ("post_attention_layernorm", "shared_head_norm")]
    for private, original in pairs:
        assert private is not original
        for a, b in zip(private.parameters(), original.parameters()):
            assert a.data_ptr() != b.data_ptr()
            torch.testing.assert_close(a, b)
    for name in ("processed", "scale"):
        assert getattr(pre.eh_proj, name).data_ptr() != getattr(model.mtp.eh_proj, name).data_ptr()
    assert not hasattr(pre, "self_attn") and not hasattr(post, "self_attn")


@pytest.mark.parametrize("fallback", ["prefill", "rank2", "embeds", "diagnostic"])
def test_eager_fallback(setup, fallback):
    model, ns, events = setup
    states = torch.randn(1, 1, 4)
    if fallback == "prefill":
        states = torch.randn(1, 3, 4)
    elif fallback == "rank2":
        states = torch.randn(1, 4)
    elif fallback == "diagnostic":
        ns["_ATTN_SCALE_FILE"] = "diagnostic-enabled"
    embeds = torch.randn_like(states) if fallback == "embeds" else None
    sentinel = object()
    model._forward = lambda *args: sentinel
    assert model(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), states, embeds) is sentinel
    assert events == []


@pytest.mark.parametrize("bad", ["ids", "positions", "attention"])
def test_numel_validation(setup, bad):
    model, _, events = setup
    ids = torch.zeros(5 if bad != "ids" else 4, dtype=torch.long)
    positions = torch.zeros(5 if bad != "positions" else 4, dtype=torch.long)
    model.mtp.self_attn.invalid = bad == "attention"
    with pytest.raises(ValueError, match="MTP"):
        model(ids, positions, torch.randn(5, 1, 4))
    assert "post" not in events
    if bad != "attention":
        assert events == []


def test_forward_context_encloses_both_graphs_and_attention(setup, monkeypatch):
    model, _, events = setup
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config = lambda: None
    context_module = ModuleType("vllm.forward_context")
    metadata = object()

    @contextlib.contextmanager
    def context(actual, config):
        assert actual is metadata
        events.append("enter")
        yield
        events.append("exit")

    context_module.set_forward_context = context
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context_module)
    model(torch.ones(1, dtype=torch.long),
          torch.zeros(1, dtype=torch.long),
          torch.randn(1, 1, 4),
          attn_metadata=metadata)
    assert events[0] == "enter" and events[-1] == "exit"
    assert events.index("pre") < events.index("attention") < events.index("post")


@pytest.mark.parametrize("flag", ["_SPLIT_GRAPH", "_DRAFT_GRAPH", "_DRAFT_ATTN"])
def test_constructor_requires_all_flags(setup, flag):
    model, ns, _ = setup
    ns[flag] = False
    other = ns["HpuGlm5NextMTPModel"](model._target)
    assert other.pre_attention_graph_core is None
    assert other.post_attention_graph_core is None


@pytest.mark.parametrize("attention,diagnostic,expected", [(True, "", True), (False, "", False),
                                                           (True, "scale-file", False)])
def test_prompt_cache_fill_capability(setup, attention, diagnostic, expected):
    model, ns, _ = setup
    ns["_DRAFT_ATTN"] = attention
    ns["_ATTN_SCALE_FILE"] = diagnostic
    other = ns["HpuGlm5NextMTPModel"](model._target)
    assert other.requires_prompt_cache_fill is expected


def test_split_flag_defaults_off():
    tree = ast.parse(SOURCE.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_SPLIT_GRAPH" for t in node.targets))
    assert ast.unparse(assignment.value) == "os.environ.get('VLLM_GLM_MTP_SPLIT_GRAPH', '0') == '1'"
