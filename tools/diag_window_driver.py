#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Request drivers for the GLM-5.3 batch-shape investigation (CPU-side, HTTP only).

capture  Reproduce the serial-vs-batched fork the layer diagnostic needs:
         request A alone (3 tokens), then B and C sent ~stagger ms apart so the
         scheduler admits B prefill > C prefill > B serial decode > B+C
         co-batched decodes. Same prompt for all three, max_tokens=3, which
         fits the diagnostic's one-shot budget (3 requests x 3 positions,
         9 forwards per rank). Sentinels must already be armed.

parity   Serial-vs-concurrent greedy probe over greedy_check.PROMPTS: every
         prompt once alone, then all prompts fired together. Reports which
         prompts flip and where. Used for the bucket-parity A/B.

Both modes write a JSON record under --out.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import sys
import threading
import time

GREEDY = "/root/llm/glm53-bench/greedy_check.py"
spec = importlib.util.spec_from_file_location("greedy", GREEDY)
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)

PROMPTS = dict(greedy.PROMPTS)

# The batched-prefill first-token reproducer from the 2026-09-11 campaign: JSON
# echo prompts of different lengths. "delta" flipped its first token between a
# lone (1, ctx) prefill and a co-batched (2, ctx) one on bf16-mHC boots.
_NOTE = "Reference notes: {}. "
DELTA_PROMPTS = {
    "j_alpha":
    _NOTE.format("plants need water and sunlight") * 45 +
    '\nReturn ONLY this JSON object, unchanged: {"id":"alpha","value":391}',
    "j_beta":
    _NOTE.format("healthy soil helps roots") * 80 +
    '\nReturn ONLY this JSON object, unchanged: {"id":"beta","value":529}',
    "j_gamma":
    _NOTE.format("observe leaves regularly") * 60 +
    '\nReturn ONLY this JSON object, unchanged: {"id":"gamma","value":841}',
    "j_delta":
    _NOTE.format("seeds need suitable conditions") * 35 +
    '\nReturn ONLY this JSON object, unchanged: {"id":"delta","value":961}',
}


def ask(base, model, prompt, max_tokens):
    return greedy.post(base, model, prompt, max_tokens)


def run_capture(args):
    prompt = PROMPTS[args.prompt]
    out = {"mode": "capture", "prompt": args.prompt, "max_tokens": args.max_tokens}
    t0 = time.time()
    out["A"] = ask(args.base, args.model, prompt, args.max_tokens)
    out["A_wall_s"] = round(time.time() - t0, 2)
    print(f"A done {out['A_wall_s']}s: {out['A']['text']!r}", flush=True)
    time.sleep(args.gap)

    results = {}
    barrier = threading.Barrier(2)

    def fire(name, delay):
        barrier.wait()
        time.sleep(delay)
        results[name] = ask(args.base, args.model, prompt, args.max_tokens)

    tb = threading.Thread(target=fire, args=("B", 0.0))
    tc = threading.Thread(target=fire, args=("C", args.stagger / 1000.0))
    tb.start()
    tc.start()
    tb.join()
    tc.join()
    out.update(results)
    for k in ("B", "C"):
        print(f"{k}: {results[k]['text']!r}", flush=True)
    out["A_eq_B"] = out["A"]["text"] == out["B"]["text"]
    out["A_eq_C"] = out["A"]["text"] == out["C"]["text"]
    print(f"A==B {out['A_eq_B']}  A==C {out['A_eq_C']}", flush=True)
    return out


def run_parity(args):
    prompts = dict(PROMPTS)
    if not args.no_delta:
        prompts.update(DELTA_PROMPTS)
    names = list(prompts)
    out = {"mode": "parity", "max_tokens": args.max_tokens, "results": {}}
    for name in names:
        out["results"][name] = {"serial": ask(args.base, args.model, prompts[name], args.max_tokens)}
        print(f"  serial {name:9s} {out['results'][name]['serial']['completion_tokens']} tok", flush=True)
    time.sleep(args.gap)
    barrier = threading.Barrier(len(names))

    def fire(name):
        barrier.wait()
        return name, ask(args.base, args.model, prompts[name], args.max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        for name, r in pool.map(fire, names):
            out["results"][name]["concurrent"] = r
    same = 0
    for name in names:
        a, b = out["results"][name]["serial"]["text"], out["results"][name]["concurrent"]["text"]
        d = greedy.first_divergence(a, b)
        out["results"][name]["first_divergence"] = d
        if d is None:
            same += 1
            print(f"  {name:9s} IDENTICAL ({len(a)} chars)", flush=True)
        else:
            print(f"  {name:9s} diverges at char {d}", flush=True)
            print(f"      serial: ...{a[max(0, d-40):d+40]!r}")
            print(f"      concur: ...{b[max(0, d-40):d+40]!r}")
    out["identical"] = same
    out["total"] = len(names)
    print(f"\n{same}/{len(names)} identical", flush=True)
    return out


def run_rowdep(args):
    """Row/composition dependence under graph replay, no diagnostics needed.

    identical: 8 copies of one prompt at once; every row must produce the same
      text and match the serial run (a mismatch = the result depends on the row
      index or on batch mates with identical content).
    condense:  the prompt plus 7 short companions (max_tokens=3) that finish
      early, forcing the input batch to condense while the prompt is mid-decode;
      the prompt's text must still match serial.
    """
    out = {"mode": "rowdep", "max_tokens": args.max_tokens, "cases": {}}
    for name in ("prose", "j_delta"):
        prompt = {**PROMPTS, **DELTA_PROMPTS}[name]
        case = {"serial": ask(args.base, args.model, prompt, args.max_tokens)["text"]}
        barrier = threading.Barrier(8)

        def same(_, barrier=barrier, prompt=prompt):
            barrier.wait()
            return ask(args.base, args.model, prompt, args.max_tokens)["text"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(same, range(8)))
        case["identical_rows_distinct"] = len(set(texts))
        case["identical_rows_match_serial"] = sum(t == case["serial"] for t in texts)
        case["identical_first_divergence"] = [greedy.first_divergence(case["serial"], t) for t in texts]
        barrier2 = threading.Barrier(8)
        short = PROMPTS["factual"]

        def mixed(i, barrier=barrier2, prompt=prompt, short=short):
            barrier.wait()
            if i == 0:
                return ask(args.base, args.model, prompt, args.max_tokens)["text"]
            return ask(args.base, args.model, short, 3)["text"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(mixed, range(8)))
        case["condense_match_serial"] = texts[0] == case["serial"]
        case["condense_first_divergence"] = greedy.first_divergence(case["serial"], texts[0])
        out["cases"][name] = case
        print(
            f"{name:8s} identical-rows: distinct={case['identical_rows_distinct']} "
            f"match_serial={case['identical_rows_match_serial']}/8 div={case['identical_first_divergence']}",
            flush=True)
        print(
            f"{name:8s} condense: match_serial={case['condense_match_serial']} "
            f"div={case['condense_first_divergence']}",
            flush=True)
    ok = all(c["identical_rows_distinct"] == 1 and c["identical_rows_match_serial"] == 8 and c["condense_match_serial"]
             for c in out["cases"].values())
    out["ok"] = ok
    print("rowdep:", "OK" if ok else "DEPENDENCE FOUND", flush=True)
    return out


def run_logits(args):
    """Feed the sampler raw-logits capture (ENABLED sentinel only, graphs replayed):
    one serial run of --prompt, then 8 identical concurrent copies. Small
    --max-tokens: the capture budget is 64 MiB of bf16 logits per rank."""
    prompt = {**PROMPTS, **DELTA_PROMPTS}[args.prompt]
    out = {"mode": "logits", "prompt": args.prompt, "max_tokens": args.max_tokens}
    out["serial"] = None if args.concurrent_only else ask(args.base, args.model, prompt, args.max_tokens)["text"]
    time.sleep(args.gap)
    barrier = threading.Barrier(8)

    def same(_, barrier=barrier):
        barrier.wait()
        return ask(args.base, args.model, prompt, args.max_tokens)["text"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        out["rows"] = list(pool.map(same, range(8)))
    out["distinct"] = len(set(out["rows"]))
    out["match_serial"] = sum(t == out["serial"] for t in out["rows"])
    print(f"logits probe: distinct={out['distinct']} match_serial={out['match_serial']}/8", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "parity", "rowdep", "logits"])
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--prompt", default="prose", choices=list(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--stagger", type=float, default=50.0, help="ms between B and C (capture)")
    ap.add_argument("--gap", type=float, default=2.0, help="s to let the engine drain between phases")
    ap.add_argument("--no-delta", action="store_true", help="parity: omit the JSON first-token reproducer prompts")
    ap.add_argument("--concurrent-only", action="store_true", help="logits: skip the serial run (capture budget)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.max_tokens is None:
        args.max_tokens = {"capture": 3, "parity": 256, "rowdep": 128, "logits": 20}[args.mode]
    out = {"capture": run_capture, "parity": run_parity, "rowdep": run_rowdep, "logits": run_logits}[args.mode](args)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")
    if args.mode == "parity":
        return 0 if out["identical"] == out["total"] else 1
    if args.mode == "rowdep":
        return 0 if out["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
