# SPDX-License-Identifier: Apache-2.0
"""CPU-only source-extracted tests; run directly to avoid package/conftest imports."""

import ast
import os
from pathlib import Path
import unittest

os.environ['PT_HPU_AUTOLOAD'] = '0'

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)


NS = {'torch': torch, 'F': F}
_extract(ROOT / 'vllm_gaudi/ops/causal_conv1d_pytorch.py', ['_depthwise_conv1d_tpc', '_apply_activation'], NS)
_extract(ROOT / 'vllm_gaudi/ops/hpu_kda_conv.py', ['hpu_kda_conv_update'], NS)
update = NS['hpu_kda_conv_update']


def _oracle(history, tokens, weight, bias, activation):
    stream = torch.cat([history, tokens]).T.unsqueeze(0)
    out = F.conv1d(stream.float(),
                   weight.float().unsqueeze(1),
                   None if bias is None else bias.float(),
                   groups=weight.shape[0]).to(tokens.dtype)
    if activation is not None:
        out = F.silu(out)
    return out.squeeze(0).T


def _tail(history, tokens, size):
    stream = torch.cat([history, tokens])
    return stream[stream.shape[0] - size:]


class TestKDAConv(unittest.TestCase):

    def test_transitions(self):
        torch.set_num_threads(1)
        torch.manual_seed(530)
        for dtype in (torch.float32, torch.bfloat16):
            for width in (2, 4, 7):
                for activation in (None, 'silu', 'swish'):
                    for acceptance in range(6):
                        with self.subTest(dtype=dtype, width=width, activation=activation, acceptance=acceptance):
                            self._scenario(dtype, width, activation, acceptance)

    def _scenario(self, dtype, width, activation, acceptance):
        batch, dim, capacity = 5, 7, 5
        garbage, row_width = 27, width + 3
        pool = torch.randn(garbage + 1, row_width, dim, dtype=dtype)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if activation else None

        def mapping(canonicals):
            return torch.tensor([[slot, *range(12 + i * capacity, 17 + i * capacity)]
                                 for i, slot in enumerate(canonicals)] + [[-1] * 6, [2, 22, 23, 24, 25, 26]])

        load, store = mapping([0, 1, 2]), mapping([3, 4, 5])
        # Acceptance zero after prefill: canonical tails are raw prompt inputs,
        # not an already advanced candidate or an oldest-window prefix.
        prompts = [torch.randn(n, dim, dtype=dtype) for n in (9, 6, 3)]
        histories = [_tail(torch.zeros(width - 1, dim, dtype=dtype), prompt, width - 1) for prompt in prompts]
        for i, history in enumerate(histories):
            pool[i, -(width - 1):] = history

        def run(current, xx, src, dst, accepted, lengths, incoming):
            before = current.clone()
            length = xx.shape[1]
            qsl = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
            actual = update(xx.reshape(-1, dim),
                            current,
                            weight,
                            src,
                            dst,
                            accepted,
                            qsl,
                            length,
                            bias=bias,
                            activation=activation).reshape(batch, length, dim)
            written = {garbage}
            for i in range(3):
                count = int(lengths[i])
                expected = _oracle(incoming[i], xx[i, :count], weight, bias, activation)
                torch.testing.assert_close(actual[i, :count],
                                           expected,
                                           rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
                                           atol=0.016 if dtype == torch.bfloat16 else 2e-6)
                canonical = int(dst[i, 0])
                written.add(canonical)
                if length == 1:
                    expected_row = torch.zeros_like(current[canonical])
                    expected_row[-(width - 1):] = _tail(incoming[i], xx[i, :1], width - 1)
                else:
                    expected_row = before[src[i, accepted[i]]]
                torch.testing.assert_close(current[canonical], expected_row, rtol=0, atol=0)
                for j in range(count if length > 1 else 0):
                    slot = int(dst[i, j + 1])
                    written.add(slot)
                    expected_row = torch.zeros_like(current[slot])
                    expected_row[-(width - 1):] = _tail(incoming[i], xx[i, :j + 1], width - 1)
                    torch.testing.assert_close(current[slot], expected_row, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual[i, count:], torch.zeros_like(actual[i, count:])))
            for slot in range(garbage):
                if slot not in written:
                    self.assertTrue(torch.equal(current[slot], before[slot]))
            self.assertTrue(torch.equal(current[garbage], torch.zeros_like(current[garbage])))
            self.assertTrue(torch.equal(actual[3:], torch.zeros_like(actual[3:])))

        lengths = torch.tensor([5, 4, 5, 0, 0])
        xx = torch.randn(batch, 5, dim, dtype=dtype)
        run(pool, xx, load, store, torch.zeros(batch, dtype=torch.long), lengths, histories)
        # Slot 26 is occupied by the last live candidate AND referenced by positive
        # padding. It must not be mistaken for the dedicated garbage row 27.
        self.assertTrue(torch.equal(pool[26, -(width - 1):], _tail(histories[2], xx[2], width - 1)))
        accepted = torch.tensor([acceptance, min(acceptance, 4), acceptance, 0, 0])
        committed = [_tail(histories[i], xx[i, :accepted[i]], width - 1) for i in range(3)]
        for length in (1, 2, 5):
            current = pool.clone()
            dst = mapping([6, 7, 8])
            tokens = torch.randn(batch, length, dim, dtype=dtype)
            lens = torch.tensor([length, max(1, length - 1), length, 0, 0])
            run(current, tokens, store, dst, accepted, lens, committed)
            canonical = [current[dst[i, 0], -(width - 1):].clone() for i in range(3)]
            run(current, torch.randn(batch, 1, dim, dtype=dtype), dst, dst, torch.zeros(batch, dtype=torch.long),
                torch.tensor([1, 1, 1, 0, 0]), canonical)

    def test_reject_nonrectangular(self):
        with self.assertRaisesRegex(ValueError, 'rectangular'):
            update(torch.zeros(3, 2), torch.zeros(5, 3, 2), torch.ones(2, 3), torch.tensor([[0, 1, 2], [1, 2, 3]]),
                   torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.zeros(2, dtype=torch.long), torch.tensor([0, 2, 3]), 2)

    def test_no_host_reads_or_advanced_history_index(self):
        tree = ast.parse((ROOT / 'vllm_gaudi/ops/hpu_kda_conv.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(isinstance(node.func, ast.Attribute) and node.func.attr in {'cpu', 'item', 'tolist'} for node in calls))
        self.assertTrue(any(
            isinstance(node.func, ast.Attribute) and node.func.attr == 'index_select' for node in calls))


if __name__ == '__main__':
    unittest.main()
