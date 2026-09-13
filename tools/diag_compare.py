# SPDX-License-Identifier: Apache-2.0
"""Compare boundaries captured by the layer diagnostic across serving contexts.

CPU-only: reads the layer-rankN-XX.pt payloads written by
vllm_gaudi.v1.worker.layer_diagnostic.flush(). No HPU access required.

Modes:
  inventory   reconstruct the serving pattern (ordinal -> phase/records/shape).
  compare     boundary-level delta map for one or more comparison pairs.
  export      JSON bundle of the delta map (--out, default /tmp/mla_comparison.json).

A comparison pair is "orda:idxA>ordb:idxB" (record indices within the payload's
record order). Each boundary's "rows" tensor has one row per record; scalar-
record payloads (1 row) are indexed at 0 for both sides. Boundaries named
differently on either side are reported as MISSING; deliberate unavailable
markers are surfaced at the end of each pair block.

Delta classes on relL2: ident (0), ulp (<1e-2), small (<1e-1), BIG (>=1e-1).
Interpretation guide (bf16/fp32 boundaries):
  ulp   ~ accumulation-order/GEMM shape numerics,
  small/BIG ~ wrong data path (wrong cache rows, wrong mask/weights, corruption).
"""
import argparse
import glob
import hashlib
import json
import os

import torch

ORDER = [
    "initial_streams",
    "attention_mhc_pre_input",
    "attention_mhc_pre_post_a",
    "attention_mhc_pre_comb_a",
    "attention_norm_input",
    # KDA layer-0 interior, execution order (kda_hooks + kda_boundary in glm5_next.py)
    "kda.qkv_proj",
    "kda.f_a_proj",
    "kda.f_b_proj",
    "kda.forget_gate",
    "kda.b_proj",
    "kda.beta",
    "kda.conv_pool_in",
    "kda.conv_out",
    "kda.conv_pool_out",
    "kda.ssm_state_in",
    "kda.kda_out",
    "kda.ssm_state_out",
    "kda.core",
    "kda.g_a_proj",
    "kda.g_b_proj",
    "kda.gate",
    "kda.pre_o_proj",
    "kda.o_proj",
    "attention_raw_output",
    "attention_mhc_post",
    "ffn_mhc_post",
]


def short_ids(dirs):
    short = {}
    for d in dirs if isinstance(dirs, list) else [dirs]:
        for f in sorted(glob.glob(os.path.join(d, "layer-rank0-*.pt"))):
            p = torch.load(f, map_location="cpu", weights_only=False)
            for r in p["records"]:
                short[r["request_id"]] = r["request_id"].split("-")[-1]
    return short


def load_dir(d, rank=0):
    files = sorted(glob.glob(os.path.join(d, f"layer-rank{rank}-*.pt")))
    return {
        int(os.path.basename(f).split("-")[2].split(".")[0]): torch.load(f, map_location="cpu", weights_only=False)
        for f in files
    }


def prompt_md5(record):
    return hashlib.md5(json.dumps(record["prompt_token_ids"]).encode()).hexdigest()[:8]


def inventory(dirs):
    short = short_ids(dirs)
    for d in dirs:
        data = load_dir(d, 0)
        prompts = {}
        first = next(iter(data.values()), None)
        for r in first["records"] if first else []:
            prompts[short[r["request_id"]]] = prompt_md5(r)
        print(f"== {d}  prompts: {prompts}")
        for key in sorted(data):
            p = data[key]
            recs = [(short[r["request_id"]], r["output_position"]) for r in p["records"]]
            md5s = {short[r["request_id"]]: prompt_md5(r) for r in p["records"]}
            print(
                f" ord{p['ordinal']:02d} {p['phase']:7s} shape={p['input_shape']} reqs={recs} md5={md5s}"
                f" nb={len(p['boundaries'])} trunc={p['truncated']} attn_mod={p.get('attention_modules') is not None}")


def get_record(data, ordinal, rec_index):
    p = data.get(ordinal)
    if p is None or rec_index >= len(p["records"]):
        raise IndexError(f"ordinal {ordinal} record {rec_index} not present")
    return p, rec_index


def deltas(x, y):
    if tuple(x.shape) != tuple(y.shape):
        return dict(note=f"shape {tuple(x.shape)} vs {tuple(y.shape)}")
    xf, yf = x.float(), y.float()
    d = (xf - yf).abs()
    if d.numel() == 0:
        return dict(max_abs=0.0, mean_abs=0.0, max_rel=0.0, rel_l2=0.0, numel=0)
    denom = yf.abs().clamp_min(1e-12)
    n = d.numel()
    rel = d / denom
    return dict(
        max_abs=float(d.max()),
        mean_abs=float(d.mean()),
        max_rel=float(rel.max()),
        rel_l2=float(d.norm() / yf.norm().clamp_min(1e-12)),
        numel=int(n),
        argmax_flat=int(d.flatten().argmax()),
        rows=list(x.shape),
        dtype=str(x.dtype),
    )


def compare_record(pa, ia, pb, ib):
    amap = {e["name"]: e for e in pa["boundaries"]}
    bmap = {e["name"]: e for e in pb["boundaries"]}
    rows = []
    for name in sorted(set(amap) | set(bmap)):
        if name not in amap or name not in bmap:
            rows.append(dict(boundary=name, status="MISSING"))
            continue
        xa, xb = amap[name]["rows"], bmap[name]["rows"]
        ra = ia if xa.shape[0] > 1 else 0
        rb = ib if xb.shape[0] > 1 else 0
        rows.append(dict(boundary=name, status="MATCHED", **deltas(xa[ra], xb[rb])))
    return rows


def delta_class(r):
    """Classify on relL2: max_rel is dominated by near-zero reference elements and is not diagnostic."""
    if r.get("status") == "MISSING":
        return "MISSIN"
    if "note" in r:
        return "SHAPEx"
    m = r.get("rel_l2", 0.0)
    if m == 0:
        return "ident "
    if m < 1e-2:
        return "ulp   "  # ~1-2 bf16 ULP: accumulation-order / recipe-shape rounding
    if m < 1e-1:
        return "small "
    return "BIG   "


def sort_key(row):
    name = row["boundary"]
    for i, key in enumerate(ORDER):
        if key in name:
            layer = -1 if name.startswith("initial") else (
                int(name.split(".")[2]) if name.startswith("model.layers.") else 99)
            return (layer, i, name)
    return (100, 99, name)


def auto_pairs(data):
    """Discover fork/control pairs from record metadata instead of hard-coded ordinals.

    For every (prompt md5, output position) present in more than one forward,
    the reference is the earliest forward whose input batch has one row; every
    other record with that key is paired against it. A second bs=1 record with
    the same key becomes a control pair (expected bit-identical); a bs>1 record
    becomes a fork pair. Returns "orda:idxa>ordb:idxb" strings.
    """
    by_key = {}
    for ordinal in sorted(data):
        p = data[ordinal]
        for index, r in enumerate(p["records"]):
            by_key.setdefault((prompt_md5(r), r["output_position"]), []).append((ordinal, index, p["input_shape"][0]))
    pairs, kinds = [], []
    for key, entries in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ref = next((e for e in entries if e[2] == 1), None)
        if ref is None or len(entries) < 2:
            continue
        for e in entries:
            if e is ref:
                continue
            pairs.append(f"{ref[0]}:{ref[1]}>{e[0]}:{e[1]}")
            kinds.append(("control" if e[2] == 1 else f"fork(bs={e[2]})", key))
    return pairs, kinds


def run(d, pairs, mode, out, auto=False):
    data = load_dir(d, 0)
    short = short_ids([d])
    results, avail = [], {}
    if auto:
        found, kinds = auto_pairs(data)
        for spec, (kind, key) in zip(found, kinds):
            print(f"auto pair {spec:>10s}  {kind:12s} md5={key[0]} pos={key[1]}")
        pairs = list(pairs) + [x for x in found if x not in pairs]
        if not pairs:
            raise SystemExit("no comparable (prompt, position) found in more than one forward")
    for left, right in (p.split(">") for p in pairs):
        oa, ia = map(int, left.split(":"))
        ob, ib = map(int, right.split(":"))
        pa, ia = get_record(data, oa, ia)
        pb, ib = get_record(data, ob, ib)

        def lab(p, i):
            r = p["records"][i]
            rec_index = [index for index, rec in enumerate(p['records']) if rec is r][0]
            return (f"{short[r['request_id']]}@dec{r['output_position']}"
                    f"({p['phase'][:3]},ord{p['ordinal']:02d},r{rec_index},"
                    f"md5={prompt_md5(r)[:4]})")

        rows = compare_record(pa, ia, pb, ib)
        head = f"{lab(pa, ia)} <=> {lab(pb, ib)}"
        for row in rows:
            results.append(dict(pair=head, **row))
        for p in (pa, pb):
            owner = lab(p, ia if p is pa else ib)
            avail.setdefault(owner, {u["name"]: u["reason"] for u in p.get("unavailable_boundaries", [])})

    if mode == "export":
        with open(out, "w") as h:
            json.dump(dict(directory=d, comparisons=results, unavailable_boundaries=avail), h, indent=1)
        print(f"wrote {out} ({len(results)} rows)")
        return

    bypair = {}
    for row in results:
        bypair.setdefault(row["pair"], []).append(row)
    for head, rows in bypair.items():
        print(f"\n==== {head}")
        for row in sorted(rows, key=sort_key):
            if delta_class(row) in ("MISSIN", "SHAPEx"):
                print(f"  [{delta_class(row)}] {row['boundary']}")
                continue
            tag = delta_class(row)
            extra = ""
            if tag not in ("ident ", ):
                extra = f" argmax={row['argmax_flat']}/{row['numel']} rows={row['rows']}"
            print(f"  [{tag}] {row['boundary']:<52s} max_abs={row['max_abs']:.3e} "
                  f"max_rel={row['max_rel']:.3e} relL2={row['rel_l2']:.3e}{extra}")
        for owner, u in avail.items():
            if u:
                print(f"  unavailable for {owner}:")
                for name, reason in sorted(u.items()):
                    print(f"    - {name}: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inventory", "compare", "export"])
    ap.add_argument("dirs", nargs="*", help="capture dirs (inventory accepts several)")
    ap.add_argument("--dir", help="single dir to compare/export")
    ap.add_argument("--pair", action="append", default=[], help="orda:idxa>ordb:idxb (repeatable)")
    ap.add_argument("--out", default="/tmp/mla_comparison.json")
    ap.add_argument("--auto",
                    action="store_true",
                    help="discover fork/control pairs from (prompt md5, position) metadata; adds to --pair")
    args = ap.parse_args()
    if args.mode == "inventory":
        inventory(args.dirs)
    else:
        run(args.dir, args.pair, args.mode, args.out, auto=args.auto)


if __name__ == "__main__":
    main()
