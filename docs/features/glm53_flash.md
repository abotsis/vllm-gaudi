# GLM-5.3-Flash (glm5_next) on HPU

Support for the GLM-5.3-Flash model family on Intel Gaudi accelerators: a hybrid
decoder mixing multi-head latent attention (MLA), gated delta net / KDA
recurrent layers, and a routed mixture-of-experts block with a **clamped SwiGLU**
activation, plus MTP (multi-token prediction) speculative decoding.

Launch example (TP=8):

```bash
vllm serve zai-org/GLM-5.3-Flash --tensor-parallel-size 8 --port 8000
```

MTP speculative decoding (the draft head ships inside the checkpoint as layer
`n_layers + 1`; no separate draft model is downloaded):

```bash
vllm serve zai-org/GLM-5.3-Flash --tensor-parallel-size 8 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
```

## What is reused from upstream vLLM, and why the rest is plugin-local

GLM-5.3-Flash support was merged into vLLM itself (vllm#53906). That
implementation only ships NVIDIA/AMD compute kernels; the platform dispatch at
`vllm/models/glm5next` has no HPU backend. The vllm-gaudi plugin therefore
carries its own model implementation, deliberately reusing every upstream piece
that is platform-neutral:

| Piece | Source | Status on HPU |
|---|---|---|
| `Glm5NextConfig` / config plumbing, `hf_config_override` MTP rewrite | upstream vLLM | reused as-is |
| KDA state shapes/dtypes/copy funcs (`MambaStateShapeCalculator.kda_state_shape`, `kda_state_dtype`, `gated_delta_net_state_copy_func`) | upstream vLLM `mamba_utils` | reused as-is |
| `GatedDeltaNetAttention` scaffolding (conv/state/norm interfaces) | upstream vLLM `mamba/gdn` | interfaces reused; state stepping replaced |
| KV-cache group machinery, `MambaSpec`, hybrid cache manager | upstream vLLM + vllm-gaudi | reused as-is |
| Un-fused `swigluoai` expert activation (`_unfused_swigluoai_moe` with `alpha`/`beta`/`limit`) | upstream vllm-gaudi (MiniMax-M3 path) | reused as-is; bit-identical to the fused path below |
| Fused clamped-SwiGLU MoE (`VLLM_GLM_FUSED_CLAMP_MOE`, default on) | vllm-gaudi plugin | HPU-only — Synapse's fused-MoE activation enum ends at `{silu, gelu, relu}` and cannot represent the clamp |

### Why a custom KDA

The upstream KDA compute stack (`vllm/models/glm5next/nvidia/kda.py`,
`nvidia/ops/third_party/kda/*`, the vendored flash-linear-attention ops) is
Triton/CuteDSL CUDA code. There is no pure-torch reference implementation
upstream to fall back to, the kernel attachment is unconditional at layer
construction, and HPU is not a considered target anywhere in that path
(only XPU is explicitly rejected; CUDA/ROCm are selected otherwise). Building
the model class from upstream would re-implement every construction subtree it
needs overridden. The plugin instead:

1. keeps the upstream structural contracts (state shapes, dtypes, copy
   functions, `MambaBase` registration, KV-cache groups) so a future upstream
   platform dispatch can absorb the HPU kernels directly;
2. owns only the compute layer: `vllm_gaudi/ops/hpu_kda_eager.py` (reference
   chunk/recurrent kernels), `vllm_gaudi/ops/hpu_kda_pytorch.py` (fast chunk
   kernel), and the state-slot management for speculative decoding (per-request
   private candidate slots appended to the recurrent-state pool, addressed by
   ids that stay stable across batch condense and swap operations).

Attention is plain NoPE MLA (`qk_rope_head_dim == 0`): no indexer, no rope. The
HPU path guards the zero-width rope cases (`oot_mla.py`,
`attention/backends/hpu_attn.py`), which upstream never exercises.

## Serving recipe (how to launch)

A minimal launch that reaches the measured throughput:

```bash
VLLM_USE_V1=1 vllm serve /path/to/GLM-5.3-Flash \
    --served-model-name glm-5.3-flash \
    --tensor-parallel-size 8 --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.50 \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
    --chat-template /path/to/GLM-5.3-Flash/chat_template.enable-thinking-switch.jinja \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}'   # optional
```

What the launcher must set (before `vllm` starts) vs.
what the plugin supplies automatically:

**Environment the launcher itself must export** — the Synapse backend reads
these at torch-import/backend-load time, before any plugin default can fire:

| Variable | Value | Why |
| --- | --- | --- |
| `PT_HPU_LAZY_MODE` | `1` | HPU graphs (lazy) are the supported serving mode; a partial/eager activation produced oversized KV pools (about 2x the correct size) and replay-scratch allocation deaths on the dev host |
| `PT_HPU_MEMORY_POOL` | `none` | validated-recipe value (Synapse-side pool off) |
| `PT_HPU_LAZY_ACC_PAR_MODE` | `1` | validated-recipe value (accumulation-parallel lazy mode) |

Then everything below is supplied automatically (any explicit user value wins):

| fused clamped-SwiGLU MoE (`VLLM_GLM_FUSED_CLAMP_MOE`) | **auto** (default on since it is worth ~2x on GLM-5.3 decode) | unfused fallback is silent; default-on avoids half-speed surprises when the var fails to propagate |
| `PT_HPU_GPT_MOE_WT_INTERLEAVED=0` | **auto** (all archs, `if unset`) | the wrapper's unset behavior is the interleaved layout while vLLM packs `w13` concatenated; pinning the layout prevents silently wrong expert math and lets the fused arm arm itself (read at graph-compile time, so a plugin-time pin suffices) |
| bucketing (lin, decode-block 2048→3200/512, prompt-ctx 3200/512, prompt-query 2048, prompt-bs 1) | **auto** for `glm5_next` (plugin default, `if unset`) | this is the bucket set the warmup coverage and perf measurements were taken with |
| `VLLM_COMPACT_GDN` | upstream default (`0`) is correct for GLM | KDA is not supported under the compact GDN path |
| `--enable-expert-parallel --enable-ep-weight-filter` | user-supplied CLI | MoE weight split across TP ranks and the weight-filter pass are load decisions, not plugin defaults |
| `--block-size 128` | **auto** (plugin default) | vLLM-gaudi sets the Gaudi page size to 128 tokens when the user does not specify one |
| `PT_HPU_WEIGHT_SHARING=0`, `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` | **auto** (all archs, `if unset`) | required for multi-rank HPU-graph serving; set by the plugin at import |

Any value the user sets explicitly wins over every plugin default.

## Speculative decoding notes

- The draft head is not loaded twice: `HpuEagleProposer.load_model` binds the
  target's already-loaded MTP block (saves a second full pass over the
  checkpoint and a duplicate of the layer weights per rank).
- The draft head runs as its own HPU graph (`VLLM_GLM_MTP_DRAFT_GRAPH=0`
  disables capture; drafting then runs the draft head eagerly).
- Prefill batches that do not produce logits (intermediate chunks of a chunked
  or mamba-block-aligned prefill) produce no draft tokens themselves; drafts
  are paired with the batch that produced the logits.

## Determinism under concurrency

Two separate effects make a co-batched request's greedy output differ from
the same request served alone. They were separated on 2026-09-13 with
`tools/diag_window.sh` (see `tools/LAYER0_DELTA_FINDINGS.md`):

1. **All-reduce row order (a bug, fixed by default).** HCCL's all-reduce sums
   each chunk of the `[bs, hidden]` buffer in a chunk-dependent rank order.
   At decode a chunk is one row, so a row's residual stream after `o_proj`
   and the MoE down-projection depended on its position in the batch by
   about one bf16 ULP. GLM-5.3 keeps that ULP in the KDA recurrent state of
   every layer after the first, and expert routing at near-ties turns it into
   different tokens: eight identical prompts decoded together produced seven
   different texts. `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (default) all-gathers
   the partials and sums them in rank order in fp32, which is position
   independent; with it the eight prompts produce one text. No measurable
   decode cost (single-stream 14.7 vs 13.6 tok/s, within restart noise).
   Buffers above `VLLM_HPU_ALLREDUCE_ALT_MAX_BYTES` (16 MiB) fall back to
   `hccl`, because every all-reduce site inside a captured prefill graph
   retains its working buffers. `hccl` restores the old behaviour.
2. **Batch-shape numerics (inherent).** A lone decode runs the bs=1 recipe, a
   full batch the bs=8 one; GEMM and attention kernels round differently at
   different shapes, exactly as on other accelerators. With gather_sum alone,
   serial vs concurrent greedy parity on the 12-prompt probe was 7/12 on the
   default bucket ladder. For bit-exact parity pin every decode to one
   recipe: `VLLM_DECODE_BS_BUCKET_MIN=8 VLLM_DECODE_BS_BUCKET_STEP=8
   VLLM_DECODE_BLOCK_BUCKET_MIN=32` (and `VLLM_PROMPT_BS_BUCKET_MIN` equal to
   the prompt bucket max when co-batched prefill is on): 12/12 identical.
   Measured single-stream: 15.0 tok/s pinned at 8 sequences, 14.0 pinned at
   32, vs 13.6-14.7 on the default ladder, all within restart noise (decode
   is host-launch-bound; padded rows are free), and warmup drops from ~13 to
   ~2 minutes because there is one decode bucket. The reference launcher
   (`glm53_serve.sh`) ships this recipe; `DECODE_LADDER=1` restores the ladder.
3. **Co-batched prefill (inherent at the current cap).** With
   `VLLM_PROMPT_BS_BUCKET_MAX>1` a prefill of four prompts at the smallest
   bucket is already 512 tokens, above the 16 MiB working-buffer cap, so those
   all-reduces take the position-dependent path and a prompt's first tokens
   depend on where it sat in the prefill batch (parity 9/12, first-token
   flips). Co-batched prefill bought ~5-12% aggregate prefill throughput, so
   the launcher defaults to one prompt per prefill forward.
   `VLLM_HPU_ALLREDUCE_LARGE_MODE=transpose` (hidden-major all-reduce, 2x
   working memory) is the untested candidate for having both.

MTP note: with the all-reduce fixed, the draft's own self-attention
(`VLLM_GLM_MTP_DRAFT_ATTN=1`) raises mean accepted length from 2.41 to 3.43
(per-position 0.91/0.71/0.48/0.32). It currently forces the eager draft
(12.9 tok/s single-stream vs 17.3 bypassed); with `VLLM_GLM_MTP_SPLIT_GRAPH=1`
it reaches 22.1 tok/s aggregate (best measured) at 16.2 single-stream but
showed one late serial-vs-concurrent divergence in 12 prompts, so the bypass
stays the default until the draft attention is graph-safe.

Tools: `tools/diag_window_driver.py rowdep` (identical prompts must give
identical rows), `ROLE=parity`, `ROLE=logits`, `ROLE=state`, `ROLE=capture`.

## Performance notes on this port

- `VLLM_HPU_DECODE_TENSOR_CACHE` keeps the HPU-graph tensor cache for decode
  graphs; on for `glm5_next` without speculative decoding, off under it
  (the doubled graph count exceeds host memory at TP=8).
- `VLLM_HPU_GRAPH_ASYNC_REPLAY=1` replays captured graphs asynchronously.
- The per-DecoderLayer `mark_step` hook is skipped under HPU-graph replay for
  `glm5_next` (each replayed boundary costs host time per decode step);
  `VLLM_CONFIG_HIDDEN_LAYERS` set explicitly restores it.
- FP8 linears hand the all-reduce contiguous tensors (no views) and do not
  round-trip `orig_M`/`orig_N` through the device.

See `docs/configuration/env_variables.md` for the environment variables.
