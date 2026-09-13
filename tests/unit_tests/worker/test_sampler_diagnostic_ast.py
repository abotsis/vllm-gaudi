# SPDX-License-Identifier: Apache-2.0
"""CPU-only AST extraction: no torch, vLLM, plugin, or HPU imports."""
import ast
from copy import deepcopy
import logging
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_gaudi/v1/worker/hpu_model_runner.py"


class Tensor:

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
        self.ndim = 2

    def element_size(self):
        return 4

    def __getitem__(self, index):
        return Tensor(self.rows[index])

    def detach(self):
        return self

    def clone(self):
        return Tensor(deepcopy(self.rows))

    def to(self, **kwargs):
        assert kwargs == dict(device="cpu", non_blocking=False, copy=True)
        return self.clone()


def load_runner(events, rank, saved):
    tree = ast.parse(SOURCE.read_text())
    methods = [
        node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
        if isinstance(node, ast.FunctionDef) and (
            node.name.startswith("_diag_sampler_") or node.name == "_run_sampling")
    ]
    for method in methods:
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
    namespace = dict(os=os,
                     stat=stat,
                     deepcopy=deepcopy,
                     logger=logging.getLogger(__name__),
                     htorch=SimpleNamespace(core=SimpleNamespace(mark_step=lambda: events.append("mark"))),
                     get_tensor_model_parallel_rank=lambda: rank[0],
                     torch=SimpleNamespace(save=lambda capture, stream: saved.append(deepcopy(capture))))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return type("Runner", (), {node.name: namespace[node.name] for node in methods})()


class SamplerDiagnosticTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sampler-test-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        os.chmod(self.directory, 0o700)
        sentinel = Path(self.directory) / "ENABLED"
        sentinel.touch(mode=0o600)
        self.env = patch.dict(os.environ, VLLM_DIAG_SAMPLER_DIR=self.directory)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.events, self.saved, self.rank = [], [], [0]
        self.runner = load_runner(self.events, self.rank, self.saved)
        self.runner.parallel_config = SimpleNamespace(tensor_parallel_size=8,
                                                      pipeline_parallel_size=1,
                                                      data_parallel_size=1)
        self.runner.speculative_config = None
        self.runner.use_async_scheduling = False
        self.runner.warmup_mode = False
        self.runner.input_batch = SimpleNamespace(req_id_to_index={"a": 1, "b": 0}, num_prompt_tokens=[3, 7])
        self.metadata = SimpleNamespace(output_token_ids=[[5], [6], [5]])
        self.logits = Tensor([[1, 2], [3, 4], [99, 99]])

    def capture(self, ids=None):
        return self.runner._diag_sampler_before(self.logits, self.metadata, ["b", "a"],
                                                ["a", "a"] if ids is None else ids, 3)

    def test_unset_no_tensor_or_config_access(self):
        os.environ.pop("VLLM_DIAG_SAMPLER_DIR")
        del self.runner.parallel_config
        self.assertIsNone(self.runner._diag_sampler_before(object(), object(), None, None, None))
        self.assertEqual(self.events, [])

    def test_scope_and_sentinel_gates(self):
        for field, bad in [("speculative_config", object()), ("use_async_scheduling", True), ("warmup_mode", True)]:
            previous = getattr(self.runner, field)
            setattr(self.runner, field, bad)
            self.assertIsNone(self.capture())
            setattr(self.runner, field, previous)
        for field, bad in [("tensor_parallel_size", 1), ("pipeline_parallel_size", 2), ("data_parallel_size", 2)]:
            previous = getattr(self.runner.parallel_config, field)
            setattr(self.runner.parallel_config, field, bad)
            self.assertIsNone(self.capture())
            setattr(self.runner.parallel_config, field, previous)
        os.chmod(self.directory, 0o755)
        self.assertIsNone(self.capture())
        os.chmod(self.directory, 0o700)
        sentinel = Path(self.directory) / "ENABLED"
        sentinel.unlink()
        self.assertIsNone(self.capture())
        sentinel.symlink_to("/etc/passwd")
        self.assertIsNone(self.capture())
        self.assertEqual(self.events, [])

    def test_outside_tmp_rejected(self):
        for directory in ["/tmp", "/var/tmp/x", "/tmp/a/b", "/tmp/../root", "relative"]:
            os.environ["VLLM_DIAG_SAMPLER_DIR"] = directory
            self.assertIsNone(self.capture())
        self.assertEqual(self.events, [])

    def test_owned_rows_histories_duplicates_and_actual_ids(self):
        capture = self.capture()
        self.logits.rows[0][0] = -1
        self.metadata.output_token_ids[0].append(99)
        self.assertEqual(capture["raw_logits"].rows, [[1, 2], [3, 4]])
        self.assertEqual(capture["output_token_ids"], [[5], [6], [5]])
        self.assertEqual(capture["request_ids"], ["b", "a"])
        self.assertEqual(capture["logits_requests"], ["a", "a"])
        self.assertEqual(capture["input_batch_indices"], [1, 1])
        self.assertEqual(capture["prompt_lengths"], [7, 7])
        sampled = Tensor([[8], [9], [10]])
        self.runner._diag_sampler_after(capture, SimpleNamespace(sampled_token_ids=sampled))
        sampled.rows[0][0] = 42
        self.assertEqual(self.saved[0]["sampled_token_ids"].rows, [[8], [9], [10]])
        self.assertEqual(self.saved[0]["logits_shape"], (3, 2))
        for path in Path(self.directory).glob("*.pt"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_nonzero_rank_clones_and_marks_without_copy(self):
        self.rank[0] = 7
        with patch.object(Tensor, "clone", autospec=True, return_value=SimpleNamespace()) as clone:
            self.assertIsNone(self.capture())
            clone.assert_called_once()
        self.assertEqual(self.events, ["mark"])
        self.assertEqual(self.runner._diag_sampler_raw_bytes, 16)

    def test_call_and_byte_caps(self):
        for _ in range(256):
            self.capture()
        self.assertIsNone(self.capture())
        self.assertEqual(len(self.events), 256)
        self.runner._diag_sampler_calls = 0
        self.runner._diag_sampler_raw_bytes = 64 * 1024 * 1024 - 16
        self.assertIsNotNone(self.capture())
        self.assertIsNone(self.capture())
        self.assertEqual(self.runner._diag_sampler_raw_bytes, 64 * 1024 * 1024)

    def test_row_cap_empty_and_fallback(self):
        self.assertIsNone(self.capture([]))
        self.assertIsNone(self.capture(["a"] * 9))
        self.assertIsNone(self.capture(["a"] * 4))
        capture = self.runner._diag_sampler_before(self.logits, self.metadata, ["b"], None, 3)
        self.assertEqual(capture["raw_logits"].rows, [[1, 2]])
        self.assertEqual(capture["selected_row_request_ids"], ["b"])

    def test_hook_order_and_io_failure_nonfatal(self):
        self.runner._prepare_sampling = lambda *args: self.metadata

        def sampler(**kwargs):
            self.events.append("sample")
            kwargs["logits"].rows[0][0] = -100
            self.metadata.output_token_ids[0].append(100)
            return SimpleNamespace(sampled_token_ids=Tensor([[11], [12]]))

        self.runner.sampler = sampler
        self.runner._run_sampling(True, self.logits, ["a", "b"], 3)
        self.assertEqual(self.events, ["mark", "mark", "sample", "mark"])
        self.assertEqual(self.saved[0]["raw_logits"].rows, [[1, 2], [3, 4]])
        self.assertEqual(self.saved[0]["output_token_ids"][0], [5])
        capture = self.capture()
        with patch("os.open", side_effect=OSError("test failure")), self.assertLogs(level="WARNING"):
            self.runner._diag_sampler_after(capture, sampler(logits=self.logits))
        self.assertEqual(self.runner._diag_sampler_calls, 2)


if __name__ == "__main__":
    unittest.main()
