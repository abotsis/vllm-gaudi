#!/usr/bin/env bash
# GLM-5.3-Flash batch-shape investigation: one HPU window, one command.
#
#   ROLE=capture ./tools/diag_window.sh   # layer-0 KDA-interior capture (serial vs co-batched fork)
#   ROLE=parity  ./tools/diag_window.sh   # bucket-parity A/B: pin decode (bs, blocks) buckets, greedy serial-vs-concurrent
#
# Shared-box protocol (same as glm53-bench/boot_gate.sh): waits for all 8
# compute nodes free via wait_cards.sh, records a claim in
# /tmp/gaudirpc_cards.state, tears down ONLY its own server, releases the claim.
# NEVER run this while another project holds cards; wait_cards.sh enforces it.
#
# Boot recipe = /root/llm/glm53-bench/glm53_serve.sh env (validated), plus what
# the layer diagnostic gate requires (hpu_model_runner._diag_sampler_directory):
# TP=8, PP=1, DP=1, no speculation, async scheduling OFF, and the sentinel dir
# armed only AFTER readiness. Prefix caching off (KDA spec-state contract).
set -u
set -o pipefail

ROLE="${ROLE:-capture}"
PORT="${PORT:-8000}"
MAXSEQ="${MAXSEQ:-8}"
MBT="${MBT:-3200}"
PBSD="${PBSD:-1}"           # 1 = standalone prefills (matches the 09-11 fork capture); 2 = admit co-batched (2,ctx) prefill
GMU="${GMU:-0.50}"
MAXLEN="${MAXLEN:-32768}"
MODEL="${MODEL:-/root/zai-org/GLM-5.3-Flash}"
REPO=/root/vllm-gaudi-fresh
BENCH=/root/llm/glm53-bench
VENV=/root/venv-gaudi2
PY="$VENV/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="${RUNDIR:-/root/llm/glm53-bench/run}"
LOG="$RUNDIR/diag_${ROLE}_${STAMP}.server.log"
REPORT="$RUNDIR/diag_${ROLE}_${STAMP}.report.txt"
mkdir -p "$RUNDIR"
exec > >(tee -a "$REPORT") 2>&1
echo "=== diag_window ROLE=$ROLE branch=$(git -C "$REPO" branch --show-current) $(date) ==="

# Decode bucket ladder. Default (capture) mirrors glm53_serve.sh: bs (1,8,MAXSEQ)
# -> {1,2,4,8}; blocks (1,512,3200) -> {1,2,4,...,256,512,...}. The [fwd] trace
# key is (bs, query, blocks): bs=1/1-block decode runs (1,1,1), bs=2 runs (2,1,2).
# Parity pins BOTH dims so every decode in the probe hits the same recipe:
# bs MIN=STEP=8 -> {8}; blocks MIN=32 -> {32,64,128,256,512,...} (8 seqs x ~3
# blocks = 24 < 32). Pinning bs alone (the findings doc's first proposal) is NOT
# enough because the block dimension changes the recipe too.
# Prompt side: with PBSD>1 the scheduler may co-batch two prefills into a
# (2, ctx) forward while a lone prompt runs (1, ctx). For parity, pin the
# prompt bs bucket too (MIN=PBSD) so a lone prompt pads into the same recipe
# as a co-batched pair. PBSD=2 ROLE=parity is the prefill-side A/B.
if [ "$ROLE" = "parity" ]; then
  BS_MIN="${BS_MIN:-8}"; BS_STEP="${BS_STEP:-8}"; BLK_MIN="${BLK_MIN:-32}"; PROMPT_BS_MIN="${PROMPT_BS_MIN:-$PBSD}"
else
  BS_MIN="${BS_MIN:-1}"; BS_STEP="${BS_STEP:-8}"; BLK_MIN="${BLK_MIN:-1}"; PROMPT_BS_MIN="${PROMPT_BS_MIN:-1}"
fi

DIAG_DIR="${DIAG_DIR:-/tmp/glm53-mla-win-$STAMP}"   # direct child of /tmp (gate requirement)

echo "--- waiting for cards ---"
if ! "$BENCH/wait_cards.sh" 3 20 43200; then echo "cards never freed"; exit 1; fi
CLAIM_ID="glm53-diag_$ROLE (branch=$(git -C "$REPO" branch --show-current))"
echo "$(date +%s) BUSY 0 $CLAIM_ID claimed" >> /tmp/gaudirpc_cards.state

export LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:/opt/habanalabs/openmpi-5.0.8/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"
export PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=1 PT_HPU_MEMORY_POOL=none
export PT_HPU_GPT_MOE_WT_INTERLEAVED=0 VLLM_USE_V1=1 VLLM_COMPACT_GDN=0
export VLLM_GLM_FUSED_CLAMP_MOE=1 VLLM_BUCKETING_STRATEGY=lin
export VLLM_DECODE_BLOCK_BUCKET_MIN="$BLK_MIN" VLLM_DECODE_BLOCK_BUCKET_MAX=3200 VLLM_DECODE_BLOCK_BUCKET_STEP=512
export VLLM_DECODE_BS_BUCKET_MIN="$BS_MIN" VLLM_DECODE_BS_BUCKET_STEP="$BS_STEP" VLLM_DECODE_BS_BUCKET_MAX="$MAXSEQ"
export VLLM_PROMPT_CTX_BUCKET_MAX=3200 VLLM_PROMPT_CTX_BUCKET_STEP=512
export VLLM_PROMPT_QUERY_BUCKET_STEP=2048 VLLM_PROMPT_BS_BUCKET_MAX="$PBSD" VLLM_PROMPT_BS_BUCKET_MIN="$PROMPT_BS_MIN"
export VLLM_DEBUG=fwd                      # [fwd] (phase, bs, query, blocks) trace per forward on every worker
if [ "$ROLE" = "capture" ]; then
  export VLLM_DIAG_SAMPLER_DIR="$DIAG_DIR"  # dir must NOT exist at boot: a stale sentinel disables capture for the runner
  rm -rf "$DIAG_DIR"
fi
echo "buckets: decode bs=($BS_MIN,$BS_STEP,$MAXSEQ) blocks=($BLK_MIN,512,3200) prompt bs=($PROMPT_BS_MIN,1,$PBSD) diag_dir=${VLLM_DIAG_SAMPLER_DIR:-none}"

PLUGIN_DIR="$($PY -c 'import vllm_gaudi, os; print(os.path.dirname(os.path.dirname(vllm_gaudi.__file__)))')"
case "$PLUGIN_DIR" in */vllm-gaudi-fresh) : ;; *) echo "FATAL: vllm_gaudi resolves to '$PLUGIN_DIR'"; exit 1 ;; esac

sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
cd "$REPO" || exit 1
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 8 --enable-expert-parallel --enable-ep-weight-filter \
  --block-size 128 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQ" --max-num-batched-tokens "$MBT" \
  --gpu-memory-utilization "$GMU" --dtype bfloat16 \
  --no-async-scheduling --no-enable-prefix-caching \
  --chat-template "$MODEL/chat_template/chat_template.enable-thinking-switch.jinja" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  > "$LOG" 2>&1 &
SERVER=$!
cleanup() {
  echo "--- teardown: killing own server pid $SERVER ---"
  { kill -TERM "$SERVER" 2>/dev/null; sleep 15; kill -9 "$SERVER" 2>/dev/null; } || true
  for p in $(ps -eo pid,ppid,comm | awk -v s="$SERVER" '$2==s {print $1}'); do kill -9 "$p" 2>/dev/null; done
  echo "$(date +%s) FREE 0 $CLAIM_ID released" >> /tmp/gaudirpc_cards.state
}
trap cleanup EXIT

echo "--- health wait (max 45 min) ---"
ready=0
for i in $(seq 1 270); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; echo "READY at $(date +%T) (~${i}0s)"; break; fi
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "SERVER DIED during startup:"; tr '\r' '\n' < "$LOG" | grep -E "Error|died|Killed|Traceback|assert" | tail -12; exit 1
  fi
  sleep 10
done
[ "$ready" = 1 ] || { echo "not ready after 45 min"; tail -5 "$LOG"; exit 1; }
# one throwaway request so any post-readiness lazy work is out of the way before arming
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say OK."}],"max_tokens":2,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
sleep 3

case "$ROLE" in
  capture)
    echo "--- arming diagnostic sentinels (after readiness, identical on all ranks: single host) ---"
    mkdir -p "$DIAG_DIR" && chmod 700 "$DIAG_DIR"
    (umask 077; : > "$DIAG_DIR/ENABLED"; : > "$DIAG_DIR/LAYER_ENABLED")
    ls -la "$DIAG_DIR"
    echo "--- driver: A alone, then B+C staggered ---"
    "$PY" "$REPO/tools/diag_window_driver.py" capture --base "http://127.0.0.1:$PORT/v1" \
      --out "$RUNDIR/diag_capture_${STAMP}.json"
    sleep 5
    echo "--- [fwd] trace on rank 0 since arming ---"
    tr '\r' '\n' < "$LOG" | grep -E "Worker_TP0.*\[fwd\]" | tail -12
    echo "--- inventory ---"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" "$REPO/tools/diag_compare.py" inventory "$DIAG_DIR"
    echo "--- export (auto-discovered fork + control pairs by prompt md5 and position) ---"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" "$REPO/tools/diag_compare.py" export --auto --dir "$DIAG_DIR" \
      --out "$RUNDIR/kda_comparison_${STAMP}.json" \
      || echo "export failed: check inventory and re-run diag_compare by hand (pair syntax orda:idxa>ordb:idxb)"
    echo "--- layer-0 KDA interior, first fork pair ---"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" "$REPO/tools/diag_compare.py" compare --auto --dir "$DIAG_DIR" \
      | grep -E "^====|layers\.0\.|final_norm|unavailable|^    - " | head -80
    ;;
  parity)
    echo "--- greedy serial-vs-concurrent probe under pinned buckets ---"
    "$PY" "$REPO/tools/diag_window_driver.py" parity --base "http://127.0.0.1:$PORT/v1" \
      --out "$RUNDIR/diag_parity_${STAMP}.json"
    echo "parity rc=$?  (0 = all prompts identical serial vs concurrent)"
    echo "--- forward shapes seen on rank 0 (prompt and decode) ---"
    tr '\r' '\n' < "$LOG" | grep -E "Worker_TP0.*\[fwd\]" | sed 's/.*\[fwd\] //' | sort | uniq -c
    ;;
  *) echo "unknown ROLE=$ROLE"; exit 2 ;;
esac
echo "=== diag_window $ROLE COMPLETE $(date) ==="
exit 0
