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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "parity"])
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--prompt", default="prose", choices=list(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--stagger", type=float, default=50.0, help="ms between B and C (capture)")
    ap.add_argument("--gap", type=float, default=2.0, help="s to let the engine drain between phases")
    ap.add_argument("--no-delta", action="store_true", help="parity: omit the JSON first-token reproducer prompts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 3 if args.mode == "capture" else 256
    out = run_capture(args) if args.mode == "capture" else run_parity(args)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")
    if args.mode == "parity":
        return 0 if out["identical"] == out["total"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
