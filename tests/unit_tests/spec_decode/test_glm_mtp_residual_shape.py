# SPDX-License-Identifier: Apache-2.0
"""CPU regression coverage of the production MTP attention residual path.

Extract only _forward so collecting these tests never imports vLLM or HPU
modules. Distinct token/feature values expose accidental cross-lane broadcasts.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_gaudi/models/glm5_next_mtp.py"
SHAPES = [(1, 5, 4), (4, 1, 4), (3, 1, 4)]
SHAPE_IDS = ["prefill", "virtual-decode", "batch-decode"]


def _extract_forward(draft_attention=True, scale=1.0):
    tree = ast.parse(SOURCE.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HpuGlm5NextMTPModel")
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_forward")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {
        "torch": torch,
        "_DRAFT_ATTN": draft_attention,
        "_attn_scale": lambda: scale,
        "_BYPASS": -1.0,
        "_PADWRITE": -2.0,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["_forward"]


def _inputs(shape):
    batch, sequence, hidden = shape
    ids = torch.arange(1, batch * sequence + 1).reshape(batch, sequence)
    positions = torch.arange(batch * sequence).reshape(batch, sequence)
    states = torch.arange(batch * sequence * hidden, dtype=torch.float64).reshape(shape) / 8 + 2
    return ids, positions, states


def _fake_model(hidden, invalid_numel=False):
    calls = []

    def embed_tokens(ids):
        return ids.unsqueeze(-1).to(torch.float64) * 3 + torch.arange(hidden, dtype=torch.float64)

    def project(value):
        embedding, states = value.split(hidden, dim=-1)
        return embedding * 2 + states * 3, None

    def self_attn(*, positions, hidden_states):
        calls.append((positions.clone(), hidden_states.clone()))
        output = hidden_states * 2 + positions.unsqueeze(-1) * 5 + 7
        output = output.reshape(-1, hidden)
        if invalid_numel:
            return output.reshape(-1)[:-1]
        return output

    mtp = SimpleNamespace(
        enorm=lambda x: x + 1,
        hnorm=lambda x: x * 2 - 1,
        eh_proj=project,
        input_layernorm=lambda x: x / 2 + 3,
        self_attn=self_attn,
        post_attention_layernorm=lambda x: x / 4 - 2,
        mlp=lambda x: x * 3 + 1,
        shared_head_norm=lambda x: x / 2 - 5,
    )
    return SimpleNamespace(mtp=mtp, embed_tokens=embed_tokens), calls


def _reference(ids, positions, states, attention):
    # Evaluate one scalar at a time: no reshape/broadcast or fake-module reuse
    # can accidentally reproduce the production residual alignment bug.
    expected = torch.empty_like(states)
    for batch in range(states.shape[0]):
        for token in range(states.shape[1]):
            for feature in range(states.shape[2]):
                position = positions[batch, token].item()
                embedding = ids[batch, token].item() * 3 + feature if position != 0 else 0
                state = states[batch, token, feature].item()
                residual = (embedding + 1) * 2 + (state * 2 - 1) * 3
                if attention:
                    residual += (residual / 2 + 3) * 2 + position * 5 + 7
                value = residual + (residual / 4 - 2) * 3 + 1
                expected[batch, token, feature] = value / 2 - 5
    return expected


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_flattened_attention_preserves_residual_lanes(shape):
    ids, positions, states = _inputs(shape)
    model, calls = _fake_model(shape[-1])
    output = _extract_forward()(model, ids, positions, states)
    expected = _reference(ids, positions, states, attention=True)
    assert output.shape == states.shape
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(calls) == 1
    assert calls[0][1].shape == states.shape
    torch.testing.assert_close(calls[0][0], positions)
    # The first post-verification selection must retain each virtual lane's
    # own residual rather than broadcasting lane zero across the selection.
    torch.testing.assert_close(output[:, 0, :], expected[:, 0, :], rtol=0, atol=0)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_attention_rejects_wrong_numel(shape):
    ids, positions, states = _inputs(shape)
    model, calls = _fake_model(shape[-1], invalid_numel=True)
    with pytest.raises(ValueError, match="MTP attention output must preserve"):
        _extract_forward()(model, ids, positions, states)
    assert len(calls) == 1


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("draft_attention,scale", [(False, 1.0), (True, -1.0)],
                         ids=["attention-disabled", "scale-bypass"])
def test_attention_bypass_unchanged(shape, draft_attention, scale):
    ids, positions, states = _inputs(shape)
    model, calls = _fake_model(shape[-1], invalid_numel=True)
    output = _extract_forward(draft_attention, scale)(model, ids, positions, states)
    assert not calls
    assert output.shape == states.shape
    torch.testing.assert_close(output, _reference(ids, positions, states, attention=False), rtol=0, atol=0)
