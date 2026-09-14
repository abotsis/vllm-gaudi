#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel counts per call for prefill suspects, measured on one HPU in lazy mode.

Each case: 2 warm-up calls (mark_step + synchronize), then one call under
torch.profiler; reports kernel event count, summed kernel time, wall, top names.
Imports only the leaf op module and AST-extracts the mHC pre function, so no
vLLM engine is needed. Run with the serving venv; needs one free card.
"""
import argparse
import ast
import collections
import os
import importlib
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile_once(fn):
    import gzip
    import json
    import tempfile

    import habana_frameworks.torch.core as htcore
    for _ in range(2):
        fn()
        htcore.mark_step()
        torch.hpu.synchronize()
    t0 = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                                with_stack=False,
                                record_shapes=False) as prof:
        fn()
        htcore.mark_step()
        torch.hpu.synchronize()
    wall = time.perf_counter() - t0
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    os.unlink(path)
    evs = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "kernel"]
    names = collections.Counter()
    dur = collections.Counter()
    for e in kernels:
        names[e["name"]] += 1
        dur[e["name"]] += e.get("dur", 0.0)
    ivs = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels)
    union, end = 0.0, -1
    for a, b in ivs:
        if a > end:
            union += b - a
            end = b
        elif b > end:
            union += b - end
            end = b
    return wall, sum(names.values()), sum(dur.values()) / 1000, names, dur, union / 1000


def report(label, wall, n, ms, names, dur, union, top=6):
    print(f"\n{label}: wall {wall * 1000:8.1f} ms, {n:7d} kernels, summed {ms:8.1f} ms, device-union {union:8.1f} ms")
    for name, cnt in names.most_common(top):
        print(f"    {cnt:7d} x {dur[name] / max(cnt, 1):7.2f} us  {name[:60]}")


def mhc_pre_fn():
    src = (ROOT / "vllm_gaudi/models/glm5_next.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mhc_pre_hpu")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "glm5_next", "exec"), ns)
    return ns["_mhc_pre_hpu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--cases", nargs="*", default=["kda_seq", "kda_parallel", "mhc_pre"])
    args = ap.parse_args()
    dev = torch.device("hpu")
    kda = importlib.import_module("vllm_gaudi.ops.hpu_kda_pytorch")
    for T in args.tokens:
        H, D = args.heads, args.dim
        g_ = torch.Generator(device="cpu").manual_seed(0)
        q, k, v = (torch.randn(1, T, H, D, generator=g_).to(torch.bfloat16).to(dev) for _ in range(3))
        g = (-0.05 * torch.rand(1, T, H, D, generator=g_) - 0.001).to(torch.bfloat16).to(dev)
        beta = torch.rand(1, T, H, generator=g_).to(torch.bfloat16).to(dev)
        for case in args.cases:
            if case.startswith("kda_"):
                kda._KDA_SCAN_PARALLEL = case == "kda_parallel"
                out = profile_once(lambda q=q, k=k, v=v, g=g, beta=beta: kda.hpu_chunk_kda(q, k, v, g, beta))
                report(f"T={T} {case}", *out)
            elif case == "mla_attn":
                # MLA prefill attention as forward_mha issues it: [bs=1, T, H, 256] bf16, causal,
                # valid_seq_lengths=[T], FusedSDPA through the plugin wrapper.
                from vllm_gaudi.extension import kernels as kern
                from vllm_gaudi.extension import ops as ext_ops
                from vllm_gaudi.extension.utils import ModuleFusedSDPA
                fsdpa = ModuleFusedSDPA(kern.fsdpa())
                Ha, Dh = 8, 256
                qa, ka, va = (torch.randn(1, T, Ha, Dh, generator=g_).to(torch.bfloat16).to(dev) for _ in range(3))
                lengths = torch.tensor([T], dtype=torch.int32, device=dev)
                out = profile_once(lambda qa=qa, ka=ka, va=va, lengths=lengths, fsdpa=fsdpa, Dh=Dh: ext_ops.
                                   prompt_attention(impl="fsdpa_impl",
                                                    query=qa,
                                                    key=ka,
                                                    value=va,
                                                    is_causal=True,
                                                    attn_bias=None,
                                                    position_bias=None,
                                                    valid_seq_lengths=lengths,
                                                    scale=Dh**-0.5,
                                                    matmul_qk_op=None,
                                                    softmax_op=None,
                                                    matmul_av_op=None,
                                                    keys_fetch_func=None,
                                                    values_fetch_func=None,
                                                    fsdpa_op=fsdpa))
                report(f"T={T} mla_attn (one layer; 11 MLA layers per forward)", *out)
            elif case == "mhc_pre":
                f = mhc_pre_fn()
                hc, hidden = 4, 4096
                residual = torch.randn(T, hc, hidden, generator=g_).to(torch.bfloat16).to(dev)
                fnw = torch.randn((2 + hc) * hc, hc * hidden, generator=g_).to(dev)
                hc_scale = torch.ones(3, device=dev)
                hc_base = torch.zeros((2 + hc) * hc, device=dev)
                out = profile_once(
                    lambda f=f, r=residual, w=fnw, sc=hc_scale, b=hc_base: f(r, w, sc, b, 1e-5, 1e-6, 1e-6, 1.0, 20))
                report(f"T={T} mhc_pre (one call; ~90 calls per forward)", *out)


if __name__ == "__main__":
    main()
