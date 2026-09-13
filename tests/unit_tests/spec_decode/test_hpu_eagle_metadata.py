# SPDX-License-Identifier: Apache-2.0
"""CPU tests executing extracted production methods without importing HPU/vLLM."""
import ast
import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def methods():
    source = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/spec_decode/hpu_eagle.py"
    tree = ast.parse(source.read_text())
    proposer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HpuEagleProposer")
    names = {"prepare_inputs", "prepare_attn_metadata"}
    nodes = [n for n in proposer.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch,
                     dataclasses=dataclasses,
                     itertools=itertools,
                     SpecDecodeMetadata=object,
                     _PAD_FIX_REPORTED=False,
                     _WARNED_SHORT_BLOCK_TABLE=False,
                     logger=SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
                     async_h2d_copy=lambda tensor, device: tensor.to(device),
                     HPUAttentionMetadataV1=SimpleNamespace(make_decode_metadata=lambda **kw: SimpleNamespace(**kw)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@dataclasses.dataclass
class Metadata:
    slot_mapping: object


@pytest.mark.parametrize("pad_slot", [0, 128])
def test_rejected_lanes_preserve_sample_indices(methods, pad_slot):
    original = torch.arange(8).view(2, 4)
    metadata = Metadata(original)
    result, hidden, sampled = methods.prepare_inputs(SimpleNamespace(device="cpu"),
                                                     metadata,
                                                     SimpleNamespace(num_draft_tokens=[2, 2]), [[10], [11, 12]],
                                                     pad_slot_id=pad_slot)
    assert hidden.tolist() == [0, -1, -1, 3, 4, -1]
    assert sampled.tolist() == [0, 4]
    assert result.slot_mapping.tolist() == [[0, pad_slot, pad_slot, 3], [4, pad_slot, 6, 7]]
    assert original.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize("pad_slot", [None, -1])
def test_rejected_lanes_require_reserved_slot(methods, pad_slot):
    with pytest.raises(ValueError, match="nonnegative reserved pad_slot_id"):
        methods.prepare_inputs(SimpleNamespace(device="cpu"),
                               Metadata(torch.arange(3)),
                               SimpleNamespace(num_draft_tokens=[2]), [[10]],
                               pad_slot_id=pad_slot)


def test_optional_pad_without_rejected_writes(methods):
    for slots, samples in [(torch.arange(3), [[10, 11, 12]]), (None, [[10]])]:
        metadata = Metadata(slots)
        result, _, _ = methods.prepare_inputs(SimpleNamespace(device="cpu"), metadata,
                                              SimpleNamespace(num_draft_tokens=[2]), samples)
        assert result is metadata


def test_attention_block_size_overflow_and_padding(methods):
    calls = []

    def buffers(tables, slots, batch_size, block_size=None, force_non_contiguous=False):
        calls.append((tables, slots, batch_size, block_size, force_non_contiguous))
        groups = [i for i, table in enumerate(tables) for _ in table]
        usage = [
            block_size if j < len(table) - 1 else slots[i][0] % block_size + 1 for i, table in enumerate(tables)
            for j in range(len(table))
        ]
        return torch.tensor([b for table in tables for b in table]), torch.tensor(groups), torch.tensor(usage)

    runner = SimpleNamespace(block_size=16,
                             attn_block_size=4,
                             _PAD_BLOCK_ID=32,
                             _PAD_SLOT_ID=128,
                             defragmenter=SimpleNamespace(resolve_all=lambda tables: tables, resolve=lambda b: b),
                             get_habana_paged_attn_buffers=buffers)
    # Actual rows: valid final position, first overflow, far overflow. Two bucket-padding rows.
    positions = torch.tensor([11, 12, 1000, 7, 0])
    table = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=torch.int32)
    result = methods.prepare_attn_metadata(SimpleNamespace(device="cpu", max_model_len=12), table, positions, runner)
    assert calls == [([[2, 3, 4], [], []], [[19], [128], [128], [128], [129]], 5, 4, False)]
    assert result.block_size == 4
    assert result.slot_mapping.dtype == torch.int64
    assert result.block_list.tolist() == [2, 3, 4]
    assert result.block_groups.tolist() == [0, 0, 0]
    assert result.block_usage.tolist() == [4, 4, 4]
    assert positions.tolist() == [11, 12, 1000, 7, 0]
    # Simulate MLA's direct write: overflow must not overwrite either real first block.
    cache = torch.full((132, ), -1)
    cache.index_copy_(0, result.slot_mapping.flatten(), torch.arange(5))
    assert cache[20].item() == cache[32].item() == -1
    assert cache[19].item() == 0
