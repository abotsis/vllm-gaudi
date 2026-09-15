# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for the actual proposer (no vLLM/HPU imports).

Run from the repository root with backend/plugin autoload disabled to bypass
HPU initialization and the HPU parent conftest::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        pytest --confcutdir=tests/unit_tests/spec_decode \\
        tests/unit_tests/spec_decode/test_hpu_eagle_feedback.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def propose():
    source = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/spec_decode/hpu_eagle.py"
    tree = ast.parse(source.read_text())
    proposer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HpuEagleProposer")
    method = next(node for node in proposer.body if isinstance(node, ast.FunctionDef) and node.name == "propose")
    # Execute the unmodified production method, not a reimplementation of its loop.
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["propose"]


class RecordingDraft:

    def __init__(self, method):
        self.method = method
        self.calls = []
        self.feedback = []
        self.last_hidden = []
        self.logit_inputs = []
        self.tokens = []

    def __call__(self, *, input_ids, positions, hidden_states, inputs_embeds, attn_metadata):
        assert hidden_states.device.type == "cpu"
        assert inputs_embeds is None
        self.calls.append((input_ids.clone(), positions.clone(), hidden_states.clone(), attn_metadata))
        step = len(self.calls)
        # Every step and row is distinguishable from both target and prior output.
        feedback = hidden_states + 100 * step
        last_hidden = feedback if self.method == "mtp" else feedback + 1000
        self.feedback.append(feedback.clone())
        self.last_hidden.append(last_hidden.clone())
        return feedback if self.method == "mtp" else (last_hidden, feedback)

    def compute_logits(self, hidden_states):
        self.logit_inputs.append(hidden_states.clone())
        tokens = (hidden_states[:, 0].long() + len(self.logit_inputs)) % 7
        self.tokens.append(tokens)
        return torch.nn.functional.one_hot(tokens, num_classes=7).float()


@pytest.mark.parametrize("method", ["mtp", "eagle"])
@pytest.mark.parametrize("shape,indices", [((2, 4, 3), [2, 7]), ((6, 1, 3), [1, 5])], ids=["prefill", "decode"])
@pytest.mark.parametrize("num_spec", [1, 4])
def test_first_and_recursive_draft_feedback(propose, method, shape, indices, num_spec):
    target = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape)
    original_target = target.clone()
    token_ids = torch.arange(shape[0] * shape[1]).reshape(shape[:2])
    positions = token_ids.clone()
    selected = torch.tensor(indices)
    initial_metadata = object()
    block_table = object()
    runner = object()
    metadata_calls = []
    draft = RecordingDraft(method)

    def prepare_attn_metadata(table, cpu_positions, model_runner):
        assert table is block_table
        assert model_runner is runner
        assert cpu_positions.device.type == "cpu"
        metadata = object()
        metadata_calls.append((cpu_positions.clone(), metadata))
        return metadata

    proposer = SimpleNamespace(method=method,
                               model=draft,
                               num_speculative_tokens=num_spec,
                               max_model_len=128,
                               prepare_attn_metadata=prepare_attn_metadata)
    result = propose(proposer, token_ids, positions, target, selected, initial_metadata, block_table, runner)

    assert result.shape == (len(indices), num_spec)
    assert result.dtype == torch.int64
    assert len(draft.calls) == len(draft.logit_inputs) == num_spec
    assert len(metadata_calls) == num_spec - 1
    torch.testing.assert_close(target, original_target)
    torch.testing.assert_close(draft.calls[0][2], original_target)
    assert draft.calls[0][3] is initial_metadata
    torch.testing.assert_close(draft.logit_inputs[0], draft.last_hidden[0].reshape(-1, shape[-1])[selected])
    torch.testing.assert_close(result, torch.stack(draft.tokens, dim=1))

    for step in range(1, num_spec):
        input_ids, input_positions, input_hidden, metadata = draft.calls[step]
        # Step 1 must select from the FIRST draft return, never target_hidden_states.
        # Steps 2 and 3 must feed back the immediately preceding draft return.
        expected = draft.feedback[0].reshape(-1, shape[-1])[selected] if step == 1 else draft.feedback[step - 1][:, 0]
        torch.testing.assert_close(input_hidden, expected[:, None, :])
        assert not torch.equal(input_hidden[:, 0], original_target.reshape(-1, shape[-1])[selected])
        torch.testing.assert_close(input_ids, draft.tokens[step - 1].int()[:, None])
        expected_positions = token_ids.reshape(-1)[selected] + step
        torch.testing.assert_close(input_positions, expected_positions[:, None])
        torch.testing.assert_close(metadata_calls[step - 1][0], expected_positions)
        assert metadata is metadata_calls[step - 1][1]
        torch.testing.assert_close(draft.logit_inputs[step], draft.last_hidden[step][:, 0])
