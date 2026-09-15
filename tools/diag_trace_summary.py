#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize a torch-profiler chrome trace (json or json.gz) from an HPU worker.

First pass is format-agnostic: event categories with total duration, the
argument keys they carry, and the top names per category. --window splits
the trace at gaps larger than --gap-ms between device events (one prefill
request per window) and reports per-window device busy time by category and
engine-like argument values, so a prefill step's MME/TPC/DMA/collective split
and its idle fraction can be read off.
"""
import argparse
import collections
import gzip
import json
import os


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--gap-ms", type=float, default=50.0)
    ap.add_argument("--device-cats",
                    default="kernel,hpu_op,Kernel,gpu_memcpy,hccl",
                    help="comma list of categories counted as device time")
    args = ap.parse_args()
    ev = [e for e in load(args.trace) if e.get("ph") == "X"]
    print(f"{len(ev)} complete events in {os.path.basename(args.trace)}")
    by_cat = collections.defaultdict(lambda: [0, 0.0, collections.Counter(), collections.Counter()])
    for e in ev:
        c = by_cat[e.get("cat", "?")]
        c[0] += 1
        c[1] += e.get("dur", 0.0)
        c[2][e.get("name", "?")] += e.get("dur", 0.0)
        for k in (e.get("args") or {}):
            c[3][k] += 1
    print("\ncategories (count, total ms, arg keys):")
    for cat, (n, dur, names, keys) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        print(f"  {cat:22s} n={n:7d} total={dur / 1000:9.1f} ms  args={sorted(keys)[:10]}")
    for cat, (n, dur, names, keys) in sorted(by_cat.items(), key=lambda kv: -kv[1][1])[:4]:
        print(f"\ntop names in {cat}:")
        for name, d in names.most_common(args.top):
            print(f"   {d / 1000:9.1f} ms  {name[:100]}")
    dev_cats = set(args.device_cats.split(","))
    dev = sorted((e for e in ev if e.get("cat") in dev_cats), key=lambda e: e["ts"])
    if not dev:
        print("\nno device events under", dev_cats, "- pick categories from the list above with --device-cats")
        return
    windows, cur = [], [dev[0]]
    for e in dev[1:]:
        if e["ts"] - (cur[-1]["ts"] + cur[-1].get("dur", 0)) > args.gap_ms * 1000:
            windows.append(cur)
            cur = [e]
        else:
            cur.append(e)
    windows.append(cur)
    print(f"\n{len(windows)} device windows (gap > {args.gap_ms} ms):")
    for i, w in enumerate(windows):
        span = (w[-1]["ts"] + w[-1].get("dur", 0) - w[0]["ts"]) / 1000
        busy_by = collections.Counter()
        for e in w:
            a = e.get("args") or {}
            key = a.get("engine") or a.get("Engine") or a.get("device_type") or a.get("stream") or e.get("cat")
            busy_by[str(key)] += e.get("dur", 0.0) / 1000
        # union busy time
        ivs = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in w)
        union, end = 0.0, -1
        for s, t in ivs:
            if s > end:
                union += t - s
                end = t
            elif t > end:
                union += t - end
                end = t
        print(f"  window {i}: {len(w):6d} events, span {span:8.1f} ms, device-union {union / 1000:8.1f} ms "
              f"({100 * union / 1000 / max(span, 1e-9):5.1f}% busy); by engine: " +
              ", ".join(f"{k}={v:.1f}" for k, v in busy_by.most_common(6)))


if __name__ == "__main__":
    main()
