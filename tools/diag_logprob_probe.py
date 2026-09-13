#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First-token logprob self-consistency probe (HTTP only).

Sends the four JSON echo prompts with logprobs=True, top_logprobs=5, max_tokens=1,
serially and then concurrently, for --reps rounds, and checks every response:
the chosen token's logprob equals the first top entry, the top list is sorted
descending, no duplicate tokens, and no -inf / -9999 sentinel. These were the
defects seen in the 2026-09-11 first_logits probe. Exit 1 on any violation.
"""
import argparse
import concurrent.futures
import json
import sys
import threading
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from diag_window_driver import DELTA_PROMPTS  # noqa: E402


def ask(base, model, prompt, timeout=300, top_logprobs=5):
    body = dict(model=model,
                messages=[dict(role="user", content=prompt)],
                temperature=0,
                top_p=1,
                seed=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check(resp):
    issues = []
    try:
        content = resp["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return ["no logprobs content"], None
    top = content.get("top_logprobs") or []
    vals = [t["logprob"] for t in top]
    toks = [t["token"] for t in top]
    if not top:
        issues.append("empty top_logprobs")
    else:
        if content["logprob"] != vals[0]:
            issues.append(f"chosen {content['logprob']:.4f} != top0 {vals[0]:.4f}")
        if vals != sorted(vals, reverse=True):
            issues.append("top list not sorted desc")
        if len(set(toks)) != len(toks):
            issues.append("duplicate tokens")
    if content["logprob"] <= -9999.0 or any(v <= -9999.0 for v in vals):
        issues.append("-9999/-inf sentinel")
    return issues, dict(token=content.get("token"), logprob=content.get("logprob"), top=list(zip(toks, vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k-sweep",
                    type=int,
                    nargs="*",
                    default=[3, 8, 1],
                    help="extra single requests with these top_logprobs values after the reps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = list(DELTA_PROMPTS)
    records, bad, total = [], 0, 0
    for rep in range(args.reps):
        serial = {n: ask(args.base, args.model, DELTA_PROMPTS[n]) for n in names}
        barrier = threading.Barrier(len(names))

        def fire(n, barrier=barrier):
            barrier.wait()
            return n, ask(args.base, args.model, DELTA_PROMPTS[n])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
            concurrent_ = dict(pool.map(fire, names))
        for phase, block in (("serial", serial), ("concurrent", concurrent_)):
            for n in names:
                issues, summary = check(block[n])
                total += 1
                bad += bool(issues)
                records.append(dict(rep=rep, phase=phase, name=n, issues=issues, summary=summary))
                flag = "BAD " if issues else "ok  "
                top = summary["top"][:3] if summary else None
                print(f"rep{rep} {phase:10s} {n:8s} {flag} {issues or ''} {top}", flush=True)
    # New top_logprobs values compile new gather recipes: does the first
    # execution of each one misbehave (per-recipe), or only the first in the
    # process (already consumed above)?
    for k in args.k_sweep:
        resp = ask(args.base, args.model, DELTA_PROMPTS["j_alpha"], top_logprobs=k)
        issues, summary = check(resp)
        total += 1
        bad += bool(issues)
        records.append(dict(rep=-1, phase=f"k={k}", name="j_alpha", issues=issues, summary=summary))
        print(
            f"k-sweep top_logprobs={k:2d} {'BAD ' if issues else 'ok  '} {issues or ''} "
            f"{summary['top'][:3] if summary else None}",
            flush=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)
    print(f"\n{total - bad}/{total} consistent; wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
