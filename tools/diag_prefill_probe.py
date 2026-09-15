#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefill throughput probe (HTTP only).

sweep:   serial max_tokens=1 requests at prompt lengths that land just under
         and just over each prompt-query bucket (128, 256, 512, 1024, 2048,
         3200): TTFT per length, prompt tokens per second, padding cost.
profile: POST /start_profile, one request per --profile-lengths, POST
         /stop_profile (worker writes torch-profiler traces to
         VLLM_TORCH_PROFILER_DIR).
"""
import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])

SENTENCE = "The river bends past the old mill and the road follows it toward the hills. "


def post(base, path, timeout=600):
    req = urllib.request.Request(base.rstrip("/v1").rstrip("/") + path,
                                 data=b"{}",
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def prompt_of(n_sentences):
    return "Summarize the following text in one sentence.\n\n" + SENTENCE * n_sentences


def timed(base, model, prompt):
    t0 = time.perf_counter()
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    return dt, d["usage"]["prompt_tokens"]


def calibrate(base, model):
    """tokens per sentence from a real request (chat template overhead measured separately)."""
    _, t0 = timed(base, model, prompt_of(1))
    _, t10 = timed(base, model, prompt_of(11))
    per = (t10 - t0) / 10.0
    return per, t0 - per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "profile"])
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--targets",
                    type=int,
                    nargs="*",
                    default=[120, 136, 250, 264, 500, 520, 1000, 1040, 2000, 2060, 3100])
    ap.add_argument("--profile-lengths", type=int, nargs="*", default=[500, 1000, 3100])
    ap.add_argument("--no-endpoints",
                    action="store_true",
                    help="profile: do not call /start_profile,/stop_profile (worker step profiler armed instead)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    per, overhead = calibrate(args.base, args.model)
    print(f"calibration: {per:.2f} tokens/sentence, {overhead:.0f} template tokens", flush=True)

    def n_for(tokens):
        return max(1, round((tokens - overhead) / per))

    out = {"mode": args.mode, "per_sentence_tokens": per, "template_tokens": overhead, "rows": []}
    if args.mode == "sweep":
        for target in args.targets:
            prompt = prompt_of(n_for(target))
            times, ptoks = [], None
            for _ in range(args.reps):
                dt, ptoks = timed(args.base, args.model, prompt)
                times.append(dt)
            times.sort()
            med = times[len(times) // 2]
            row = dict(target=target,
                       prompt_tokens=ptoks,
                       ttft_median_s=med,
                       ttft_min_s=times[0],
                       tok_per_s=ptoks / med)
            out["rows"].append(row)
            print(f"  {ptoks:5d} tokens: TTFT {med:6.3f}s (min {times[0]:.3f})  {ptoks / med:8.0f} tok/s", flush=True)
    else:
        if not args.no_endpoints:
            print("start_profile:", post(args.base, "/start_profile"), flush=True)
        for target in args.profile_lengths:
            dt, ptoks = timed(args.base, args.model, prompt_of(n_for(target)))
            out["rows"].append(dict(target=target, prompt_tokens=ptoks, ttft_s=dt))
            print(f"  profiled {ptoks} tokens: {dt:.3f}s", flush=True)
        time.sleep(2)
        if not args.no_endpoints:
            print("stop_profile:", post(args.base, "/stop_profile"), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
