# Layer-0 B1-vs-B2 delta findings + next-window protocol (2026-09-11, CPU-only)

All analysis below was done on already-captured HPU data (`/tmp/glm53-layer0-sync`,
schema-1 payloads from the layer diagnostic). **No GPU/HPU work is required to
reproduce any of it.** Companion tool: `tools/diag_compare.py`.

## 1. The capture contains a golden fork

`/tmp/glm53-layer0-sync` (boot of 14:10, rank0 ordinals):

| ord | phase | shape | records (req@pos) |
|-----|-------|-------|-------------------|
| 00  | prefill | (1,128) | A@0 |
| 01  | decode | (1,1) | A@1 |
| 02  | decode | (1,1) | A@2 |
| 03  | prefill | (1,128) | B@0 |
| 04  | prefill | (1,128) | C@0 |
| 05  | decode | (1,1) | B@1 |
| 06  | decode | (2,1) | B@2, C@1 |
| 07  | decode | (2,1) | C@2 |

All three requests (A=8b0a7014, B=ae2bf44b, C=bbae834f) carry the **same prompt**
(md5 51a3725d, 37 tokens, absolute positions matched by record metadata).
A is served fully serially; B gets one serial decode; B2 co-batching begins at
ord06. This single boot therefore contains the whole serial-vs-batched fork.

NOTE: every prefill in this boot was a standalone (1,128) forward. The
scheduler never co-batched prefill here. The serving-visible first-token flip
reported by the gate may additionally involve co-batched (2,128) prefill, which
this dataset does not cover (see section 5).

## 2. Delta map (rank0, identical prompt/position, different batch context)

`tools/diag_compare.py compare --dir /tmp/glm53-layer0-sync --pair '1:0>6:1'`
 compares A@dec1 (B1) against C@dec1 (B2); `--pair '2:0>6:0'` A@dec2 (B1) vs
B@dec2 (B2). Full machine-readable map: `/tmp/boundary_delta_map.json`.

Golden control `--pair '1:0>5:0'` (A@dec1-B1 vs B@dec1-B1): **bit-identical on
all ~90 boundaries** (boot-internal determinism).

Fork results (relL2 on full boundary tensor; `max_rel` is dominated by
near-zero reference elements and is not diagnostic — use relL2/max_abs):

| boundary | pos1 fork relL2 | pos2 fork relL2 |
|----------|-----------------|-----------------|
| attention_mhc_pre_input / norm_input | 0 (ident) | 0 (ident) |
| attention_mhc_pre_post_a (fp32) | 1.2e-08 (max_abs 3e-08) | 5.1e-08 |
| attention_mhc_pre_comb_a (fp32) | 5.4e-08 | 4.8e-08 |
| **attention_raw_output** | **4.0e-03 (max_abs 4.9e-04)** | **3.2e-03 (2.4e-04)** |
| attention_mhc_post | 4.3e-03 | 3.1e-03 |
| ffn_mhc_post .. deeper layers | 2-6e-02 plateau | 2-6e-02 plateau |

Interpretation:

- With bit-identical attention inputs and weights, `attention_raw_output`
  differs by ~1 bf16 ULP across the token set (relL2 3-4e-3 ≈ bf16 eps).
- Deltas then stay ~bounded (2-6e-2) through all 45 layers: amplification via
  mHC mixing, **no blow-up signature of a logic error** (wrong cache row /
  wrong mask / wrong weights would produce O(1) relL2).
- FP32 mHC coefficient deltas (1e-8) are one-ULP fp32 rounding: the mHC
  pre-step GEMM is also shape-sensitive, but at fp32 resolution.
- All 8 TP ranks are **bit-identical** on every boundary in the B2 forward
  (97/97 ranks0/1/3/7 equality) → the effect is global, not rank-shard noise.

## 3. Root-cause model (current best)

The divergence is batch-**shape**-dependent recipe selection, seeded inside the
layer-0 attention block (first nonzero point: attention_raw_output, after
o_proj; exact seed op unknown without the MLA-interior capture), consistent
with HPU per-shape compiled GEMM/kernel recipes: B1 decode runs shape M=1,
B2 decode runs M=2 (and inside the paged kernel, different block counts), so
accumulation order/tiling differs and per-row results drift by ~1 ULP at each
boundary. Serving mechanism: decode graphs are bucketed
(`extension/bucketing/linear.py` default `(1, 32, max_num_seqs)`, linear preset
per `glm53_serve.sh`); bs=1 and bs=2 select different graph buckets on the real
serving path as well.

No code path in `oot_mla.py` / `hpu_attn.py` / `hpu_paged_attn.py` /
`extension/ops.py::flat_pa_mla/reduce_attn_scores` executes a pristine-vs-upgraded
reasoning downgrade on batch size: b2b gather/scatter are zero-weighted
matmuls (bit-exact adds), `pipelined_pa` masking is per-block, and no
`bs == 1`-conditioned dtype change exists on this path.

## 4. What the next GPU window must decide

A. **Seed op** (one boot, instrumentation already in tree + 34/34 CPU tests):
   run the MLA-interior capture per the protocol below, then compare the fork
   pairs with `tools/diag_compare.py`. Decision tree:
   - differs from `fused_qkv_a_proj`/`q_b_proj` outputs on → projection GEMMs;
   - identical up to `decode_q_absorbed`, differs at `decode_query`+ → absorb
     bmm (W_UK_T) or paged kernel; `decode_attention_latent` discriminating;
   - identical up to `decode_attention_latent` → `_v_up_proj` bmm or `o_proj`
     (`pre_o_proj` input hook distinguishes);
   - identical through `decode_v_up_proj` and `pre_o_proj` differs → o_proj
     only. In all cases quantify: if first-diff is ~1 ULP on bf16, it is
     recipe-shape rounding (managed, not "fixed"); if some op shows an
     order-of-magnitude jump (relL2 ≥ 1e-1) that op is a real bug.

B. **Bucket-parity mitigation** (one gate boot, env-only, no code change):
   `VLLM_DECODE_BS_BUCKET_MIN=2 VLLM_DECODE_BS_BUCKET_STEP=2`
   -> decode warmup buckets {2,4,8,16,32}; bs=1 decode pads into the bs=2
   recipe, making B1 and B2 decode shapes identical. Expected: golden fork
   pairs become bit-identical and the co-batched flip disappears from greedy
   gates. Watch: +1 decode bucket, bs=1 decode runs the bigger GEMMs (small
   perf cost), conv/KDA pools see padded dummy rows (mask/verify state writes
   for dummy slots), KDA/hybrid layers under padding need the §5 prefill note.

C. **Prefill-parity gap**: if the gate flip persists (or spawn docs show
   co-batched (2,128) prefill), extend the capture driver to send the two
   prompts simultaneously so a (2,128) prefill ordinal is admitted (the
   admission rule accepts both reqs at pos 0 in one forward), then compare
   `prefill_q/k/v` + `latent_cache_write_input` + `cache_written_rows`.

## 5. Next-window capture protocol (turnkey)

```sh
# 1. boot with existing instrumentation (hooks merged in tree, tests green)
#    VLLM_DIAG_SAMPLER_DIR must not have ENABLED/LAYER_ENABLED at boot;
# 2. after readiness on every rank:
mkdir -p /tmp/glm53-mla-win && chmod 700 /tmp/glm53-mla-win
touch /tmp/glm53-mla-win/ENABLED           # 0600
touch /tmp/glm53-mla-win/LAYER_ENABLED     # 0600
# 3. driver: request A alone (wait), requests B+C staggered so B prefill > B
#    serial decode > B+C co-batched decodes (matches the layer0-sync pattern);
# then run:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python tools/diag_compare.py inventory /tmp/glm53-mla-win
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python tools/diag_compare.py export \
  --dir /tmp/glm53-mla-win --pair '1:0>6:1' --pair '2:0>6:0' --out /tmp/mla_comparison.json
```

Compare boundaries expected in payload: `model.layers.0.attention.*.{q,
kv_c_normed,k_pe,decode_q_absorbed,decode_query,decode_attention_latent,
decode_v_up_proj}`, wrapper module outputs (`fused_qkv_a_proj`, `q_a_layernorm`,
`q_b_proj`, `o_proj`, `pre_o_proj`), `slot_mapping`, `seq_lens`,
`cache_written_rows`. `block_list/block_mapping/block_groups/attn_bias/
cache_history` are deliberately `unavailable` (not captured); the golden-fork
comparison does not need them.

Sentinel caveats (unchanged): owner-only /tmp child, 0700 dir, 0600 sentinels,
created only after readiness, identical on all TP ranks, never toggled
mid-forward; synthetic admission budget 9 forwards / 3 reqs / 3 positions /
64 MiB per rank is one-shot per boot (no refund on failed attempts).

## 6. Addendum 2026-09-12 (CPU-only): turnkey scripts + a correction to §4B

**Scripts (in tree):** `tools/diag_window.sh` (ROLE=capture | ROLE=parity) and
`tools/diag_window_driver.py`. Both wait for all 8 cards via
`glm53-bench/wait_cards.sh`, claim `/tmp/gaudirpc_cards.state`, boot with the
validated `glm53_serve.sh` env + `--no-async-scheduling --no-enable-prefix-caching`
(the diagnostic gate requires async off), arm sentinels only after readiness,
run the driver, run `diag_compare.py`, and tear down only their own server.
Run nothing here while another project holds cards.

**Correction to §4B.** The `[fwd]` trace key is `(phase, bs, query, blocks)`
and the 09-11 capture shows the fork decodes ran `('decode', 1, 1, 1)` vs
`('decode', 2, 1, 2)`: the **block-count dimension differs too**, so pinning
only `VLLM_DECODE_BS_BUCKET_MIN=2` leaves bs=1 and bs=2 on different recipes.
Parity needs both dims pinned. `ROLE=parity` defaults to bs `(8,8,8)` → {8}
and blocks `MIN=32` (ramp 32,64,…; 8 seqs × ≤3 blocks = 24 pads to 32), so
every decode in the greedy probe runs one recipe. Prediction: 8/8 serial ==
concurrent. If flips persist under one recipe, the cause is not shape rounding.

**Capture driver pattern** (matches the 09-11 fork exactly): prompt = greedy
`prose`, thinking off, max_tokens=3; A alone, then B and C 50 ms apart. The
diagnostic budget is one-shot per boot, so do not send anything else before
the driver; the script sends one throwaway request *before* arming.

## 7. Correction 2026-09-12: layer 0 is a KDA layer, and the delta is not bounded

**Layer type.** `config.json` `layer_types[0] == "linear_attention"`. The §3/§4A
decision tree names MLA-only ops, and `layer_diagnostic.attention_hooks` skipped
linear layers, so the planned "MLA-interior capture" would have recorded layer 0
as `unavailable`. The seed of the fork therefore sits inside the **KDA decode
step** of layer 0, with identical inputs *and* identical incoming state: the
KDA state capture of 09-11 (`/tmp/glm53-debug-reset/state_comparison.json`)
shows layer 0's conv and recurrent pools equal between serial and co-batched,
and layers 1, 2, 4, ... onward differing.

**Growth, not plateau.** Re-reading `/tmp/boundary_delta_map.json` for the
A@dec1 (bs=1) vs C@dec1 (bs=2) pair:

| boundary | relL2 |
|---|---|
| layers.0.attention_raw_output | 4.0e-3 |
| layers.1-12 (attention/ffn mhc_post) | ~3e-2 |
| layers.20 | 1.1e-1 |
| layers.30 | 1.9e-1 |
| final_norm | 1.6e-1 (max_abs 0.73) |

The step-wise growth is consistent with a ~1 ULP seed amplified by MoE routing
flips at near-ties plus mHC mixing; it is not proof of that. A second bs>1 defect
in a deeper layer is not excluded by this data.

**Instrumentation now in tree** (`layer_diagnostic.kda_hooks/kda_boundary/
kda_sequence_boundary/kda_pool_rows`, called from `HpuGlm5NextKdaAttention.
forward_orig`; CPU tests `tests/unit_tests/worker/test_kda_diagnostic.py`).
Boundaries under `model.layers.0.kda.*`, in execution order:

`qkv_proj, f_a_proj, f_b_proj, forget_gate, b_proj, beta, conv_pool_in (physical
rows of the selected lanes' conv slots, before the update), conv_out,
conv_pool_out, ssm_state_in (loaded recurrent state, snapshot ordered before the
in-place update), kda_out, ssm_state_out, core, g_a_proj, g_b_proj, gate,
pre_o_proj, o_proj`. Prefill records `conv_out`, `ssm_state_in/out` per
sequence, `core`, and the projections.

**Decision tree for the KDA seed** (fork pair, bf16 boundaries; 1 ULP ~ relL2 3-4e-3):
- `qkv_proj` / `b_proj` / `f_*` / `g_*` differ -> fp8 linear recipe (M=1 vs M=2
  GEMM/GEMV kernel selection). Confirm: `forget_gate`/`beta` differ only as much
  as their inputs.
- projections identical, `conv_pool_in` identical, `conv_out` differs ->
  depthwise conv update kernel is batch-shape sensitive.
- `conv_out` identical, `ssm_state_in` identical, `kda_out`/`ssm_state_out`
  differ -> `kda_decode_step` (fp32 l2norm / matmul readouts) picks a different
  recipe for N=1 vs N=2. fp32 ops differing by >1e-6 relL2 here would be a real bug.
- `ssm_state_in` differs while `conv_pool_in` is identical -> state load/slot
  mapping bug (wrong slot for a lane): O(1) delta expected, a genuine defect.
- everything identical through `gate`, `pre_o_proj` differs -> `_gated_o_norm`
  fusion; `pre_o_proj` identical, `o_proj` differs -> o_proj GEMM or the TP
  all-reduce (message-size dependent HCCL algorithm; check that all ranks agree
  on `pre_o_proj` first).

Budget check: `ssm_state_*` rows are 8 heads x 128 x 128 fp32 = 512 KiB each;
two snapshots x <=3 rows x <=9 forwards ~ 27 MiB of the 64 MiB per-rank cap.

**Where the growth happens (added 2026-09-12).** Per-layer relL2 for the two
fork pairs is flat between jumps, and every jump lands on an FFN step of an
MoE layer (pos1 fork: layer 18 ffn 0.066 -> 0.120, layer 29 ffn 0.153 -> 0.204;
pos2 fork: layer 3 ffn 0.031 -> 0.049, layer 21 ffn 0.049 -> 0.083, layer 29
ffn 0.119 -> 0.180). Attention steps, KDA or MLA, do not jump; the plateau even
shrinks in places (layer 34). This is the signature of expert-routing flips at
near-ties amplifying a tiny seed, not of the KDA recurrence compounding it.
Test for the next capture: compare the MoE router top-k indices of the fork
rows at those layers (not yet instrumented; `boundary()` on the router's
selected-expert tensor would do it).

**Prefill-side evidence (from `/tmp/glm53-debug-reset/STATUS.md` §11 and
`first_logits.json`, verified 2026-09-12).** The "delta" JSON prompt flips its
first token between serial (B1) and co-batched (B2) prefill on bf16-mHC boots
and not on the fp32-mHC boot (36/36). Serial top-2 margin is 0.25 nats. Both
boots were MTP; no same-boot single-knob A/B exists. Also: the HPU logprob
reporting is corrupted in several first_logits entries (duplicate values, an
unsorted top list whose chosen token is not the max, a -9999 sentinel), so
reported logprobs are not usable as parity evidence.

## 8. State at end of 2026-09-12 (all CPU-side; nothing HPU-verified today)

Branch `fix/glm53-kda-transition`, six commits over 7a44eab6:

| commit | content | HPU status |
|---|---|---|
| 45505b1a | KDA chunk-transition transpose fix + oracle test | verified 09-10 (kdafix 8/8) |
| 672b3eb3 | spec-decode/MTP fixes (candidate conv, pad slots, prompt-cache fill, MTP reshape) | validated 09-11 PBSD2 boot |
| 129ee324 | diagnostics + comparator + window scripts (incl. KDA-interior hooks) | KDA hooks never run on HPU |
| 4444942c | sampled-logprob host ownership; blocking prompt-logprob copies | **unverified**: rerun the first-token probe with logprobs on |
| ea535226 | MoE routing capture + offline top-k margins; prompt-bucket parity; delta prompts | never run on HPU |
| b7036c08 | test hygiene (default dtype restore) | n/a |

**Window checklist (in order):**
1. `ROLE=parity ./tools/diag_window.sh` (nospec, bf16 mHC, PBSD=1). Expect 12/12
   serial == concurrent. Then `PBSD=2 ROLE=parity` for the prefill side.
2. `ROLE=capture ./tools/diag_window.sh`. In the compare output look for the
   first `[ulp]`/`[small]` row under `model.layers.0.kda.*` (seed op, §7 tree),
   then the `[ROUTE] ... experts FLIP` rows: the layer of the first flip should
   coincide with the first jump in the per-layer relL2 profile, and its margin
   should be of the order of the upstream delta.
3. Logprob check: `probe_first_logits.py` pattern (logprobs=True, top_logprobs=5)
   on any boot; every entry must have a sorted top list whose first value equals
   the chosen token's logprob, no duplicate tokens, no -9999.

## 9. HPU window 2026-09-13: shape is not the cause; graph replay is row-dependent

All boots: TP=8, nospec, bf16 mHC, `--no-async-scheduling`, decode buckets
pinned to one recipe `(8, 1, 32)`, prompt bs 1. Warmup takes ~2 min with a
single decode bucket (vs 30-40 min on the full ladder); a boot is ready in ~10 min.

| boot | knobs | result |
|---|---|---|
| parity #1 | defaults | serial vs concurrent 9/12 identical (code2, prose, j_delta diverge); every serving decode ran `('decode', 8, 1, 32)` per the fwd trace |
| capture | LOGPROB_PROBE=1 | fork pairs A@dec1/dec2 (alone) vs B/C (co-batched): **bit-identical on all 199 boundaries**, layer-0 KDA interior included, 0 routing flips (lazy diag forward, graphs bypassed) |
| parity #2 | VLLM_HPU_DECODE_TENSOR_CACHE=0, rowdep probe | rowdep: 8 identical `prose` copies at once -> **7 distinct texts**, 2/8 match serial (6 rows diverge at char 81, one at 138); 8 identical `j_delta` -> 5 distinct, 2/8 match; parity 8/12 with a *different* set of diverging prompts (prose2, struct, math) |

Conclusions:
1. The 09-11 layer-0 1-ULP delta was recipe shape (real bs 1 vs 2 in lazy
   mode). With shapes pinned, co-batching is bit-exact in lazy mode.
2. The serving-visible divergence is **graph-replay specific and row
   dependent**: identical inputs in different rows of one replayed batch give
   different outputs, and the set of affected prompts changes from boot to
   boot. That is a nondeterminism/race signature, not numerics. The decode
   tensor cache is not the cause.
3. Suspects, in order: the launch-boundary policy of 7a44eab6 (no mark_step
   before the TP all-reduce, no per-layer mark_step under replay; validated
   only on a 2-rank single-stream bench), then anything else that reads a
   producer's output inside the replayed graph without a dependency. Test:
   `VLLM_HPU_ALLREDUCE_MARKSTEP=1 VLLM_CONFIG_HIDDEN_LAYERS=1` + rowdep.

Logprobs: with the host-ownership fix in place the **first** logprobs request
of a boot still returned the 09-11 garbling (token ids = every other id then
zeros, values correct); all later requests, serial and concurrent, were clean
(23/24). So it is a first-execution defect of the gather recipe, not a
lifetime race. Commit 48770bd8 drops the int64->int32 narrowing of the ids in
both HPU gather paths; the probe's top_logprobs sweep tells first-in-process
from first-per-recipe.

Tooling notes: `diag_compare --auto` now picks the reference by fewest
admitted records (under padded buckets every forward is 8 wide);
`diag_window_driver.py rowdep` is the replay row-dependence probe;
`LOGPROB_PROBE=1` / `PRE_ARM_HOOK=` run on the live server before any capture.
