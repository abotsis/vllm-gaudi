#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-row logit agreement under graph replay, from the sampler raw-logits capture.

Reads sampler-<pid>-<ordinal>.pt written by hpu_model_runner._diag_sampler_*
(rank 0). Calls with one real row are the serial reference run; calls with
several real rows are the concurrent run of identical prompts. For each decode
step: max |logit| difference between rows, rows that differ from row 0, the
sampled tokens, and each row's difference from the serial call at the same
step. CPU-only.
"""
import argparse
import glob
import os

import torch


def load(directory):
    calls = []
    for f in sorted(glob.glob(os.path.join(directory, "sampler-*.pt"))):
        p = torch.load(f, map_location="cpu", weights_only=False)
        calls.append((p["ordinal"], p))
    return [p for _, p in sorted(calls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()
    calls = load(args.directory)
    serial = [p for p in calls if p["real_row_count"] == 1]
    multi = [p for p in calls if p["real_row_count"] > 1]
    print(f"{len(calls)} sampler calls: {len(serial)} single-row, {len(multi)} multi-row")
    print(f"{'step':>4s} {'rows':>4s} {'max|d| rows':>12s} {'rows!=row0':>10s} {'vs serial max|d|':>16s}  tokens")
    for step, p in enumerate(multi[:args.steps]):
        x = p["raw_logits"].float()
        n = x.shape[0]
        d0 = (x - x[0:1]).abs().amax(dim=1)
        differ = [i for i in range(1, n) if d0[i] > 0]
        vs = ""
        if step < len(serial):
            s = serial[step]["raw_logits"].float()[0:1]
            vs = f"{(x - s).abs().amax(dim=1).max().item():.3e}"
        toks = p.get("sampled_token_ids")
        toks = toks.flatten().tolist()[:n] if toks is not None else None
        print(f"{step:4d} {n:4d} {d0.max().item():12.3e} {str(differ):>10s} {vs:>16s}  {toks}")
    # where does the first row disagreement appear?
    first = next(
        (i for i, p in enumerate(multi) if (p["raw_logits"].float() - p["raw_logits"].float()[0:1]).abs().max() > 0),
        None)
    print("first multi-row step with any row disagreement:", first)


if __name__ == "__main__":
    main()
