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

## 10. 2026-09-13 afternoon: the defect is row-position dependent under replay, entering at layer 0's output

Boots (all pinned buckets, nospec, bf16 mHC):

| boot | knob | rowdep (8 identical `prose`) | parity |
|---|---|---|---|
| TC off | VLLM_HPU_DECODE_TENSOR_CACHE=0 | 7 distinct, 2/8 == serial | 8/12 |
| mark_steps | ALLREDUCE_MARKSTEP=1, CONFIG_HIDDEN_LAYERS=1 | 7 distinct, 2/8 | 10/12 |
| acc_par 0 | PT_HPU_LAZY_ACC_PAR_MODE=0 | 7 distinct, 2/8 | (logits role) |

None of those knobs matter. What does:

- **Raw logits under replay, aligned by position** (`ROLE=logits`): batch rows
  0 and 1 are bit-identical to the serial run at every position; rows 2-7
  differ by 1-4 logit units from position 1 on, and only flip the argmax at a
  near-tie (position 12, serial top-2 margin 0.000).
- **Lazy capture at position 2 only** (`VLLM_DIAG_LAYER_POSITIONS=2`): the
  request moved from index 7 to index 0 by a batch condense has layer 0
  bit-identical to the correct row (incoming KDA state included, output
  included) and differs from layer 1 on; 18 expert-routing flips follow.
- **Per-layer KDA state under replay** (`ROLE=state`): at position 2 the
  correct row (index 1) and the wrong row (index 2) agree on layer 0's conv
  and recurrent pools and disagree on every other layer's, by ~1 bf16 ULP in
  the conv pools (which hold each layer's qkv projection of its input).

Reading: layer 0's recurrent update is exact for every row, so the replayed
step computed layer 0's **output** differently for rows >= 2: downstream of
the state update, i.e. gate / gated norm / o_proj / **TP all-reduce** / mHC.
A rounding-level difference that is deterministic by row index in an 8-row
batch, identical for rows 0 and 1, is what a reduce-scatter all-reduce gives:
the [8, hidden] bf16 buffer is split into per-rank chunks (one decode row
each) and each chunk is summed in a different rank order. Serial row 0 and
concurrent row 0 share chunk 0, hence "serial is correct". The hybrid model
keeps the ULP in its recurrent state and MoE routing flips amplify it.

Test in flight: `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (all_gather + fixed-order
fp32 sum, commit 01e15a3f). Prediction: rowdep 1 distinct 8/8 and parity 12/12.

## 11. Confirmed (2026-09-13 16:06): the TP/EP all-reduce is not row-invariant

`VLLM_HPU_ALLREDUCE_MODE=gather_sum` (commit 01e15a3f + 16 MiB working-buffer
cap 11b2489a), pinned buckets, nospec, bf16 mHC:

| probe | before | with gather_sum |
|---|---|---|
| 8 identical `prose` at once | 7 distinct texts, 2/8 == serial | **1 text, 8/8 == serial** |
| 8 identical `j_delta` at once | 5 distinct, 2/8 | **1 text, 8/8** |
| condense cases | mixed | **both match serial** |
| 12-prompt serial vs concurrent | 8-10/12 | **12/12** |

Root cause: HCCL's all-reduce sums each chunk of the buffer in a rank order
that depends on the chunk index; at [8, hidden] a chunk is one decode row, so
a row's residual-stream value after o_proj / MoE down-proj depends on its
position in the batch by ~1 bf16 ULP. A lone request always sits at row 0,
hence "serial is right". GLM-5.3 carries the ULP in the KDA recurrent state of
every layer after the first and MoE routing flips at near-ties turn it into
different greedy tokens. Nothing in KDA, MoE, the tensor cache, mark_step
policy, lazy accumulation mode or bucket shapes is at fault; bucket shape
only changed which chunk order a row received (the 09-11 layer-0 delta).

Unpinned-ladder parity and the decode-throughput cost of gather_sum are
measured in the boots that follow; the first gather_sum attempt without the
size cap died in prefill warmup (PT_DEVMEM: 8x buffers retained per
all-reduce site inside captured graphs).

## 12. Final numbers (2026-09-13 16:06-16:58) and the shipped recipe

Single-stream decode, `bench.py --reps 3 --max-tokens 300`, median tok/s, one
boot each, same day:

| boot | all-reduce | buckets | tok/s | 12-prompt parity |
|---|---|---|---|---|
| baseline | hccl | default ladder | 13.61 | 7/12 |
| A | gather_sum | default ladder | 14.68 | 7/12 |
| C | gather_sum (default now) | pinned (8,8,8)/(32,512,3200) | 15.02 | **12/12** |

Restart noise on this box is ~6%, so all three are the same speed. Decisions:

- `VLLM_HPU_ALLREDUCE_MODE` defaults to `gather_sum` (commit 789f75ce). It
  removes the row-position dependence (identical rows are identical, each
  batch composition is deterministic) at no measurable cost. `hccl` restores
  the previous behaviour.
- Bit-exact serial-vs-concurrent parity additionally needs one decode recipe:
  `VLLM_DECODE_BS_BUCKET_MIN=8 VLLM_DECODE_BS_BUCKET_STEP=8
  VLLM_DECODE_BLOCK_BUCKET_MIN=32` at `--max-num-seqs 8`. Free at bs 8, and
  warmup drops from ~13 to ~2 minutes (one decode bucket). Not made a plugin
  default: at larger max_num_seqs the padded lone decode may cost real time,
  measure before adopting. Documented in docs/features/glm53_flash.md.
- Everything else tested today is exonerated: bucket shape as a *bug* (it only
  selected the chunk order), decode tensor cache, all-reduce/per-layer
  mark_steps, lazy accumulation-parallel mode, KDA state handling, MoE
  routing, fp8 activation scaling (per-token).

Open items carried forward: MTP (nspec) parity was never re-measured with the
fix; the previous 4/8-vs-nospec numbers should be redone with gather_sum and
pinned buckets. Prefill co-batching (PBSD=2) parity with the fix is also
untested today (all boots ran prompt bs 1).

## 13. MTP-4 re-verified on the fixed tree (2026-09-13 17:31)

Same recipe as §12 boot C (gather_sum default, pinned buckets, prompt bs 1),
`--speculative-config {"method":"mtp","num_speculative_tokens":4}`, draft graph
on, draft attention bypassed (shipped defaults). Reference = same-tree nospec
capture `greedy_refs/nospec_gs_pinned.json` taken one boot earlier.

| metric | MTP-4 today | nospec today | previous MTP records |
|---|---|---|---|
| greedy vs nospec (8 prompts) | 3/8 identical; all 5 divergences at the known near-tie chars (81, 138, 105, 273, 647), coherent both sides | n/a | 4/8 (09-11), 3/8 (09-10) |
| serial vs concurrent parity (12 prompts) | **12/12** | 12/12 | never clean |
| mean accepted length | **2.405** | n/a | 2.06 (09-10), 2.27 (09-02) |
| per-position acceptance | 0.824 / 0.444 / 0.126 / 0.010 | n/a | 0.716 / 0.274 / 0.062 / 0.005 |
| aggregate mixed throughput (agg_probe) | **20.16 tok/s** | n/a | 15.4-19.5 |
| single-stream decode (bench.py, prose prompt) | **17.26 tok/s** (1.23x) | 14.01 | 16.8 prose (09-02, vs 10.5 base) |

Reading: the MTP path is deterministic across batch composition now, and
acceptance is the best measured on this port (a corrupted residual stream in
rows >= 2 was also degrading the draft's inputs). The remaining MTP-vs-nospec
divergences are the verify step's 5-position kernel shape vs a 1-position
decode, the same class as the default-ladder 7/12, not a speculation defect.

Still open: draft self-attention stays bypassed (its KV slot wiring, STATUS §7
item 3); co-batched prefill (PBSD=2) parity untested on the fixed tree.

## 14. Co-batched prefill (PBSD=4) on the fixed tree (2026-09-13 18:50)

Pinned decode buckets, prompt bs pinned to 4 (`VLLM_PROMPT_BS_BUCKET_MIN=4`),
gather_sum default with the 16 MiB cap:

- rowdep `prose`: 4 distinct texts, 3/8 == serial, two rows diverge at **char 0**
  (first token); `j_delta`: 8/8 identical.
- parity 9/12 (code @52, prose @81, math @132). Prompt forwards ran (4,128),
  (4,256), (4,512) plus chunked continuations.

Cause: a 4-prompt prefill at the 128-token bucket is 512 tokens = 32 MiB of
gathered working buffer, above the cap, so those sites use the plain HCCL
reduction and a token's position in the co-batched buffer decides its
rounding. Every all-reduce site in a captured graph retains its working
buffers, which is why the cap exists (uncapped gather_sum died in prefill
warmup). Per-site working memory: gather 8x, fp32 or transpose 2x, plain 0.

Decision for the launcher: `PBSD=1`. Co-batched prefill bought the previous
agent ~5-12% aggregate prefill throughput (c1 ~2000 vs c4 2105-2247 tok/s;
prefill is MME-bound) and costs first-token determinism. Follow-up candidate,
committed but untested: `VLLM_HPU_ALLREDUCE_LARGE_MODE=transpose`
(hidden-major all-reduce; position-invariant if HCCL chunks the flat buffer
by count; 2x memory). Test = a PBSD=4 parity boot with that env.

## 15. MTP draft self-attention re-tested on the fixed tree (2026-09-13 19:05)

Same recipe as §13, `VLLM_GLM_MTP_DRAFT_ATTN=1` (draft runs its own
self-attention; this forces the eager draft because the captured draft core
only covers the attention-bypassed path).

| metric | attention off (§13) | attention on, eager draft |
|---|---|---|
| mean accepted length | 2.405 | **3.429** (CUDA reference 3.70) |
| per-position acceptance | 0.82 / 0.44 / 0.13 / 0.01 | **0.91 / 0.71 / 0.48 / 0.32** |
| aggregate throughput (agg_probe) | 20.16 tok/s | 18.15 tok/s |
| single-stream (bench.py) | 17.26 tok/s | 12.90 tok/s |
| greedy vs nospec | 3/8 (near-tie chars) | 3/8 (same chars) |
| serial vs concurrent parity | 12/12 | 12/12 |

The 09-10 verdict "attention on hurts acceptance" was measured with rows >= 2
of every batch on a corrupted residual stream; with the all-reduce fix the
draft's attention is worth +1.0 accepted token per step. What it costs today
is the eager draft (the +46% graphed-draft win is lost). Next boot:
`VLLM_GLM_MTP_SPLIT_GRAPH=1` (captures the tensor work around the eager
attention). If that recovers the graphed-draft speed, attention on becomes
the default; otherwise the draft KV wiring needs to be made graph-safe.

### 15b. Draft attention + split draft graph (`VLLM_GLM_MTP_SPLIT_GRAPH=1`, 19:24)

| mode | accepted len | aggregate tok/s | single-stream tok/s | parity |
|---|---|---|---|---|
| A: bypass (shipped default) | 2.405 | 20.16 | 17.26 | 12/12 |
| B: attention on, eager draft | 3.429 | 18.15 | 12.90 | 12/12 |
| C: attention on, split graph | 3.429 | **22.10** | 16.16 | **11/12** (j_delta @337) |

C is the fastest under concurrency and identical to B in what the draft
computes, but its split path (captured pre/post cores around eager attention,
four synchronizes per draft step) produced one late serial-vs-concurrent
divergence that A and B never did in four boots. Not made default. Rerun to
tell deterministic from racy; the real fix is a graph-safe draft attention
(draft KV slot wiring), which would combine A's boundaries with B's
acceptance and is the top MTP performance item.

## 16. Shipped launcher validated end to end (2026-09-13 20:02-20:52)

`/root/llm/glm53-bench/glm53_serve.sh` as now shipped (pinned decode recipe,
one prompt per prefill forward, prefix caching off, MAXSEQ 32 nospec / 8 under
speculation), driven by `launcher_gate.sh`:

| launcher mode | parity (12 prompts) | acceptance | aggregate tok/s | single-stream tok/s |
|---|---|---|---|---|
| nospec, 32 seqs, prefix caching ON (first attempt) | 11/12 | n/a | n/a | 14.39 |
| nospec, 32 seqs, prefix caching off | **12/12** | n/a | n/a | 14.14 |
| MTP-4, 32 seqs (first attempt) | **hung**: decode config (160, 1, 256) not warmed up, compiled on the fly with heartbeat timeouts | | | |
| MTP-4, 8 seqs | **12/12** | 2.405 | 20.89 | 17.56 |

Launcher changes made today: `PBSD` default 4 -> 1; single decode recipe
(`DECODE_LADDER=1` restores the ladder); `--no-enable-prefix-caching` by
default (`PREFIX_CACHE=1` opts in, refused under speculation because the tree
raises there); `MAXSEQ` defaults to 8 when `NSPEC>0`. Backup of the previous
launcher: `glm53_serve.sh.bak_20260913`.

Prefix caching is the one setting that still costs parity on the launcher
(11/12 vs 12/12): a request that hits a cached prefix runs a different
prefill shape than one that does not. It is off by default now; treating it
is a separate item (mamba cache mode 'align' also changes KDA slot handling).

## 17. Prefill: TTFT sweep on the shipped launcher (2026-09-13 22:13)

`tools/diag_prefill_probe.py sweep` through `glm53-bench/prefill_gate.sh`
(nospec, 32 seqs pinned, one prompt per forward, max_tokens=1, median of 3):

| prompt tokens | bucket | TTFT s | tok/s | note |
|---|---|---|---|---|
| 123 | 128 | 0.139 | 885 | fixed cost ~0.11 s dominates |
| 139 | 256 | 0.166 | 835 | |
| 251 | 256 | 0.172 | 1456 | |
| 267 | 512 | 0.244 | 1093 | +42% for crossing the edge |
| 507 | 512 | 0.249 | 2037 | |
| 523 | 1024 | 0.378 | 1383 | +52% for crossing the edge |
| 1003 | 1024 | 0.377 | 2659 | |
| 1035 | 2048 | 1.015 | 1019 | **+170%** for crossing the edge |
| 1995 | 2048 | 1.046 | 1907 | |
| 2059 | 3200 (un-warmed, added on the fly) | 0.931 | 2212 | faster than the 2048 graph |
| 3099 | 3200 | 0.915 | 3387 | best per-token rate |

Two things stand out. (1) The query-bucket ladder 128/256/512/1024/2048/3200
has an edge at 1024 that costs 2.7x for the next token, and the 2048 bucket
itself is superlinear: 0.51 ms/token vs 0.37 at 1024 and 0.29 at 3200.
(2) The 3200 shape was not in the warmup list ("Generated 10 prompt
buckets" stops at 2048; the first 2059-token request added (1, 3200, 0) as
an unprepared bucket) and still beats the captured 2048 graph per token, so
HPU-graph replay is not obviously the fast path for large prefills here.
The MME budget at 3387 tok/s is ~3% of peak: prefill remains host/launch or
TPC bound. Profile of steps at 1024/2048/3200 in the next section.

## 18. Prefill profile (2026-09-14 08:28) and the KDA decay-dot fix

Rank-0 torch-profiler trace of one 512-token prefill step (profiler confined
to rank 0 without stacks; the 8-worker with-stack variant OOM-killed the
host twice, see commit 4d7e8ecc):

- step wall 156 ms, device union 160 ms of a 185 ms window: **86% device
  busy**. Prefill is device-bound, not host-bound.
- 981k kernel events in the step, mean 1.4 us; summed kernel time 1.38 s
  across engines. Top: `reduce_sum_fwd_f32` 434 ms (140k), a fused fp32
  elementwise kernel 191 ms, `GEMM` 71 ms (11.5k). The MME is a minority.
- Under HPU-graph replay all kernels belong to one recipe, so attribution
  came from a one-card microbenchmark (`tools/diag_kernel_counts.py`): one
  KDA layer's `hpu_chunk_kda` at 512 tokens = 13.5k kernels, 57.6 ms summed,
  2.6 ms device-union; x33 layers ~ 86 ms of the step's 160 ms (54%). At 2048
  tokens 9.7 ms/layer (320 ms of ~1000). The mHC pre is ~120 kernels per
  call (whole-tensor Sinkhorn), not a factor. The parallel chunk scan does
  not change device time (the loop over 8 chunks is not the cost).
- The cost was `_decay_dot`: sum_d a_i,d b_j,d exp(ga_i,d - gb_j,d) done as a
  [B, tc, tc, 32] elementwise product per D-tile (~170 M fp32 elements per
  layer at 512 tokens), reduced on the TPCs.

Fix (commit c860b8ea): per 16-row block, split the exponent around the
block's first-row gate and contract over D with a matmul on the MME; the
gate lower bound (-5/token) keeps every factor inside fp32 (e^{+-80}),
non-causal entries are clamped and masked. Per layer: 57.6 -> 3.7 ms summed,
2.6 -> 0.3 ms union (512 tokens); 224 -> 14.3 ms, 9.7 -> 0.9 ms (2048).
Oracle gates: transition tests 12/12 (CPU), test_hpu_kda 14/14 on HPU.
Serving gates (TTFT sweep, greedy vs launcher_nospec_v2, rowdep, parity,
bench) in the boot that follows.

Still open from the sweep: the 2048 query bucket is superlinear (0.51
ms/token vs 0.37 at 1024 and 0.29 at 3200) and the KDA share there is only
~30%, so another component misbehaves at exactly that shape.

## 19. KDA decay-dot fix on the shipped launcher (2026-09-14 08:52)

`prefill_gate.sh` with `GATES=1`, same recipe as §17, median-of-3 TTFT:

| prompt tokens | TTFT before (s) | TTFT after (s) | speedup | tok/s after |
|---|---|---|---|---|
| 123 | 0.139 | 0.125 | 1.11x | 985 |
| 139 | 0.166 | 0.129 | 1.29x | 1076 |
| 251 | 0.172 | 0.137 | 1.26x | 1828 |
| 267 | 0.244 | 0.165 | 1.48x | 1614 |
| 507 | 0.249 | 0.168 | 1.48x | 3014 |
| 523 | 0.378 | 0.221 | 1.71x | 2371 |
| 1003 | 0.377 | 0.218 | 1.73x | 4590 |
| 1035 | 1.015 | 0.332 | 3.06x | 3120 |
| 1995 | 1.046 | 0.324 | 3.23x | 6164 |
| 2059 | 0.931 | 0.536 | 1.74x | 3845 |
| 3099 | 0.915 | 0.479 | 1.91x | 6476 |
The 2048 cliff was the decay-dot's temporaries, not attention: 1035 tokens
went from 1.015 s to 0.332 s. Gates on the same boot: rowdep 8/8 identical
rows on both prompts and both condense cases, serial-vs-concurrent parity
12/12, decode bench 13.9 tok/s (unchanged). Greedy vs the pre-change
launcher capture 5/8 with divergences only at the known near-tie characters:
the contraction order changed at the fp32-rounding level (the oracle bounds
it at ~1e-6), which is the batch-shape class, not a defect.
