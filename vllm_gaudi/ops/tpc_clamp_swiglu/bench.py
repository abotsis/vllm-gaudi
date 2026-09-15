# SPDX-License-Identifier: Apache-2.0
"""Latency microbench: fused TPC clamp_swiglu vs the 5-op pure-torch epilogue.

Shapes (as in the MoE pipeline):
  [8, 1, 2048]  -> h = [8, 1, 4096]  (T=8 tokens,  i_dim=2048)
  [128, 2048]   -> h = [128, 4096]   (T=128 tokens, i_dim=2048)
Warm 20, then 100 timed reps; torch.hpu.synchronize() around the timed window.
Lazy mode: mark_step() per rep for both implementations (identical accounting).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import load_clamp_swiglu  # noqa: E402 — MUST precede habana import (GC_KERNEL_PATH)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import habana_frameworks.torch.core as htcore  # noqa: E402

LIMIT = 10.0
WARMUP, REPS = 20, 100


def torch_epilogue(h: torch.Tensor, d: int) -> torch.Tensor:
    g = h[..., :d].float().clamp(max=LIMIT)
    u = h[..., d:].float().clamp(min=-LIMIT, max=LIMIT)
    return (F.silu(g) * u).to(h.dtype)


def bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
        htcore.mark_step()
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
        htcore.mark_step()
    torch.hpu.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6  # us per call


def main() -> int:
    op = load_clamp_swiglu()

    print(f"{'shape':<14} {'TPC fused':>12} {'torch 5-op':>12} {'speedup':>8}", flush=True)
    for shape in ([8, 1, 2048], [128, 2048]):
        _, i_dim = shape[0] * shape[1] if len(shape) == 3 else shape[0], shape[-1]
        h = (torch.randn(*shape[:-1], 2 * i_dim, device="cpu") * 20).to(torch.bfloat16).to("hpu")

        t_tpc = bench(lambda h=h: op(h, LIMIT))
        t_eager = bench(lambda h=h, d=i_dim: torch_epilogue(h, d))

        print(f"{str(shape):<14} {t_tpc:>10.1f}us {t_eager:>10.1f}us {t_eager / t_tpc:>7.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
