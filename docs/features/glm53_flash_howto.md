# GLM-5.3-Flash on 8x Gaudi2 with vllm-gaudi: quick how-to

Tested 2026-09-15 on an HL-225 (8x Gaudi2) box, SynapseAI 1.24.1 (habana-torch-plugin 1.24.1.482),
Python 3.12, 125 GB host RAM.

## 1. Software

```bash
# Gaudi stack: SynapseAI 1.24.1 drivers/firmware installed (hl-smi works), then a Python 3.12 venv
python3.12 -m venv ~/venv-gaudi2 && source ~/venv-gaudi2/bin/activate
pip install habana-torch-plugin==1.24.1.482 habana-torch-dataloader==1.24.1.482 \
            habana_gpu_migration==1.24.1.482 habana-pyhlml==1.24.1.482 neural_compressor_pt==3.6

# vLLM at the commit this plugin tree was built against (v0.27.2rc0 + 1041 commits)
git clone https://github.com/vllm-project/vllm.git ~/vllm
git -C ~/vllm checkout 3dc7a68ce4
VLLM_TARGET_DEVICE=empty pip install -e ~/vllm      # CPU/"empty" build; the plugin supplies the HPU backend

# the plugin: this fork, branch carve/5-tpc = core fixes + sampler + KDA kernels + GLM-5.3 model + TPC clamp kernel
git clone https://github.com/abotsis/vllm-gaudi.git ~/vllm-gaudi
git -C ~/vllm-gaudi checkout carve/5-tpc
pip install -e ~/vllm-gaudi

# weights (fp8 checkpoint, ~100 GB)
huggingface-cli download zai-org/GLM-5.3-Flash --local-dir ~/zai-org/GLM-5.3-Flash
```

Branch notes: `pr/core`, `pr/sampler`, `pr/kda` are the same code as standalone PRs; `carve/4-glm` is the model
without the TPC kernel (correct, ~10% lower MTP throughput). `fix/glm53-kda-transition` is the full working
branch including diagnostics and tools.

## 2. Serve (the validated recipe)

Save as `serve_glm53.sh` and run it. Defaults: TP 8, MTP-4 speculative decode, 32k context, 8 sequences.

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-$HOME/zai-org/GLM-5.3-Flash}
NSPEC=${NSPEC:-4}          # 0 disables speculative decode (then MAXSEQ 32 is fine)
MAXSEQ=${MAXSEQ:-8}        # under MTP: 8 (16 works, ~15 tok/s per stream but ~190 tok/s aggregate)
MAXLEN=${MAXLEN:-32768}    # 262144 also works: see section 4
GMU=${GMU:-0.35}           # KV-cache share; prompt graphs with context take ~13 GiB
PORT=${PORT:-8000}
export PYTHONPATH="$HOME/vllm-gaudi${PYTHONPATH:+:$PYTHONPATH}"
export PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=1 PT_HPU_MEMORY_POOL=none
export PT_HPU_GPT_MOE_WT_INTERLEAVED=0 VLLM_USE_V1=1 VLLM_COMPACT_GDN=0
export VLLM_GLM_FUSED_CLAMP_MOE=1 VLLM_BUCKETING_STRATEGY=lin
export VLLM_KDA_SCAN=parallel
# pinned decode shapes (one graph per block bucket, no batch ladder): deterministic under concurrency
export VLLM_DECODE_BS_BUCKET_MIN=$MAXSEQ VLLM_DECODE_BS_BUCKET_STEP=$MAXSEQ VLLM_DECODE_BS_BUCKET_MAX=$MAXSEQ
export VLLM_DECODE_BLOCK_BUCKET_MIN=128 VLLM_DECODE_BLOCK_BUCKET_STEP=512 VLLM_DECODE_BLOCK_BUCKET_MAX=3200
# one prompt per prefill forward; query buckets 128..2048,3200; context buckets 0/64/../256 blocks
export VLLM_PROMPT_BS_BUCKET_MIN=1 VLLM_PROMPT_BS_BUCKET_MAX=1 VLLM_PROMPT_QUERY_BUCKET_STEP=2048
export VLLM_PROMPT_CTX_BUCKET_MIN=0 VLLM_PROMPT_CTX_BUCKET_STEP=64 VLLM_PROMPT_CTX_BUCKET_MAX=$((MAXLEN/128))
export VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS=20480   # chunks with up to 16k context replay HPU graphs
SPEC=(); [ "$NSPEC" -gt 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}")
cd "$HOME/vllm-gaudi"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name glm-5.3-flash --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 8 --enable-expert-parallel --enable-ep-weight-filter \
  --block-size 128 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQ" --max-num-batched-tokens 3200 \
  --gpu-memory-utilization "$GMU" --dtype bfloat16 \
  --chat-template "$MODEL/chat_template/chat_template.enable-thinking-switch.jinja" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --no-enable-prefix-caching "${SPEC[@]}"
```

Plugin defaults you do not need to set: `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (order-invariant TP all-reduce),
`VLLM_GLM_MTP_DRAFT_ATTN=1`, `VLLM_GLM_MTP_ATTN_GRAPH=1` (draft attention captured as one graph per step),
`VLLM_GLM_TPC_CLAMP=auto` (TPC clamp kernel, loaded in lazy mode when present).

Warmup takes 4-5 minutes (~17 GiB of HPU graphs per card) and the server answers `/health` after it.
Prefix caching must stay off under speculative decode on this port.

## 3. First requests after boot

The MTP draft compiles one graph per decode shape and one prompt-cache fill per prompt bucket on first
use (2-8 s each). Send a few requests of different lengths (100, 500, 1k, 2k, 3k, 7k tokens) and a burst of
8 short ones before measuring; after that everything replays. (`warm_client.py` in the bench dir does this.)

Expected on 8x Gaudi2, greedy: TTFT 0.13 s (128 tok) / 0.22 s (1k) / 0.45 s (3k) / 1.2 s (7.7k);
decode 21-24 tok/s single stream, ~75 tok/s aggregate at 4 streams; sampled (T=0.7, top_p 0.9) ~19.7 tok/s.

## 4. Long context (262144 per sequence)

```bash
cat > buckets_256k.txt <<'B'
(1, [128, 256, 512, 1024, 2048, 3200], [0, 64, 128, 192, 256, 512, 1024, 1536, 2048])
(8, 1, [128, 256, 512, 1024, 2048, 4096, 8192, 10240])
B
VLLM_BUCKETING_FROM_FILE=$PWD/buckets_256k.txt MAXLEN=262144 GMU=0.25 ./serve_glm53.sh
```
KV pool 507k tokens (two full-length requests). Measured: 87k-token prompt 47 s to first token, 219k 141 s;
decode at those contexts ~6 tok/s (the 5 speculative lanes each carry the full block table), 23 tok/s at
short contexts. Prefill beyond 16k of context runs eager (~1.5-2k tok/s).

## 5. Checks

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hello in five words."}],
       "max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```
Look for `was not warmed-up` in the server log: none should appear after the warm-up requests.
CPU-safe unit tests: `pytest tests/unit_tests/test_bucketing.py tests/unit_tests/ops/test_hpu_kda_transition.py
tests/unit_tests/ops/test_hpu_kda_conv.py`; on a card: `tests/unit_tests/sampler/`, `tests/unit_tests/ops/test_hpu_kda.py`.
