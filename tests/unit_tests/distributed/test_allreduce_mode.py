# SPDX-License-Identifier: Apache-2.0
"""Order-invariant all-reduce helper (CPU, AST-extracted; the module imports habana at top level)."""
import ast
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[3] / "vllm_gaudi/distributed/device_communicators/hpu_communicator.py"


def _load():
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sum_in_rank_order")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["sum_in_rank_order"]


def test_sum_is_independent_of_row_position_and_matches_fp32_sum():
    f = _load()
    torch.manual_seed(0)
    parts = (torch.randn(8, 8, 4096) * 3).to(torch.bfloat16)  # [world, rows, hidden]
    out = f(parts, torch.bfloat16)
    ref = parts.float().sum(0).to(torch.bfloat16)
    # fp32 accumulation of 8 bf16 values in a fixed order equals torch's fp32 reduction up to fp32 rounding,
    # which never moves a bf16 result by more than one ulp; identical rows must come out identical.
    same_rows = parts[:, :1].expand(-1, 8, -1).contiguous()
    out_same = f(same_rows, torch.bfloat16)
    assert all(torch.equal(out_same[i], out_same[0]) for i in range(8))
    assert torch.equal(out_same[0], f(parts[:, :1], torch.bfloat16)[0])
    assert (out.float() - ref.float()).abs().max() <= 2 * torch.finfo(torch.bfloat16).eps * ref.float().abs().max()


def test_all_reduce_dispatches_on_mode():
    tree = ast.parse(SRC.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HpuCommunicator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "all_reduce")
    src = ast.unparse(fn)
    assert "_ALLREDUCE_MODE == 'gather_sum'" in src and "all_gather_into_tensor" in src
    assert "_ALLREDUCE_MODE == 'fp32'" in src
    assert src.count("dist.all_reduce(") == 2  # fp32 path and default path
