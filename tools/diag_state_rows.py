#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-layer KDA state agreement between co-batched identical requests (CPU-only).

Reads state-rank0-XX.pt from kda_state_diagnostic (raw incoming conv/recurrent
pool rows snapshotted before each forward, graphs still replayed). For every
decode forward with several admitted records at the same output position, it
compares each record's rows against the first record's, per layer and pool,
and prints the layers whose incoming state differs. It also checks, per
request, whether the state changed between consecutive positions (i.e. the
replayed step actually wrote it) and whether a differing request's state
equals another request's state from the previous position (stale-by-one).
"""
import argparse
import glob
import os

import torch


def load(directory):
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "state-rank0-*.pt"))):
        p = torch.load(f, map_location="cpu", weights_only=False)
        out.append(p)
    return sorted(out, key=lambda p: p["ordinal"])


def layer_rows(p, rec_index):
    """{layer_name: {conv: tensor, recurrent: tensor}} for one record."""
    result = {}
    for layer in p["layers"]:
        result[layer["name"]] = {pool["kind"]: pool["raw"][rec_index].float() for pool in layer["pools"]}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    args = ap.parse_args()
    payloads = load(args.directory)
    print(f"{len(payloads)} state payloads")
    by_req = {}
    for p in payloads:
        recs = [(r["request_id"][-8:], r["output_position"], r["row"], i) for i, r in enumerate(p["records"])]
        summary = [(a, b, c) for a, b, c, _ in recs]
        print(f"ord{p['ordinal']:02d} {p['phase']:7s} shape={p['input_shape']} records={summary}")
        for rid, pos, row, i in recs:
            by_req.setdefault(rid, {})[pos] = (p["ordinal"], row, layer_rows(p, i))
        if p["phase"] != "decode" or len(recs) < 2:
            continue
        base = layer_rows(p, recs[0][3])
        for rid, pos, row, i in recs[1:]:
            other = layer_rows(p, i)
            diffs = []
            for name in base:
                for kind in ("conv", "recurrent"):
                    d = (base[name][kind] - other[name][kind]).abs().max().item()
                    if d > 0:
                        diffs.append((name.split("layers.")[-1].split(".")[0], kind, d))
            print(f"   {recs[0][0]}(row{recs[0][2]}) vs {rid}(row{row}) @pos{pos}: "
                  f"{'IDENTICAL all layers' if not diffs else str(len(diffs)) + ' differing pool rows'}")
            for L, kind, d in diffs[:12]:
                print(f"      layer {L:>2s} {kind:9s} max_abs={d:.3e}")
    print("\nper-request state change between consecutive positions (max_abs over recurrent pools, per layer index):")
    for rid, poss in by_req.items():
        keys = sorted(poss)
        for a, b in zip(keys, keys[1:]):
            la, lb = poss[a][2], poss[b][2]
            changed = [
                name.split("layers.")[-1].split(".")[0] for name in la
                if (la[name]["recurrent"] - lb[name]["recurrent"]).abs().max() > 0
            ]
            print(f"   {rid}: pos{a}->pos{b} (rows {poss[a][1]}->{poss[b][1]}): recurrent changed in "
                  f"{len(changed)}/{len(la)} layers" +
                  ("" if len(changed) == len(la) else f", unchanged: "
                   f"{[n for n in (x.split('layers.')[-1].split('.')[0] for x in la) if n not in changed][:12]}"))
    # stale-by-one check: does a request's pos-2 state equal another request's pos-1 state?
    reqs = list(by_req)
    for r1 in reqs:
        if 2 not in by_req[r1]:
            continue
        for r2 in reqs:
            if r2 == r1 or 1 not in by_req[r2]:
                continue
            l1, l2 = by_req[r1][2][2], by_req[r2][1][2]
            same = sum(1 for n in l1 if (l1[n]["recurrent"] - l2[n]["recurrent"]).abs().max() == 0)
            if same:
                print(f"   stale-by-one: {r1}@pos2 recurrent == {r2}@pos1 in {same}/{len(l1)} layers")


if __name__ == "__main__":
    main()
