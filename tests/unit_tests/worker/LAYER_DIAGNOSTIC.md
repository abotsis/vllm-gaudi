# Layer boundary diagnostic (CPU-validated only)

This diagnostic is independent of `STATE_ENABLED` and the state/raw-sampler
counters. It shares the existing validated `VLLM_DIAG_SAMPLER_DIR` and
`ENABLED` gate: owner-only direct child of `/tmp`, directory 0700, sentinel
0600, synchronous TP8/PP1/DP1, NSPEC0, no warmup. It additionally requires
regular owner-owned `LAYER_ENABLED` mode 0600, created **only after readiness**.
A sentinel present at runner construction permanently disables this diagnostic
for that runner. Keep configuration/sentinels identical across all TP ranks;
do not toggle during an in-flight forward. This implementation supports the
lazy HPU path only, not torch.compile/eager execution.

Admission allows three requests first observed at output position zero, their
first three output positions, at most nine selected records and nine attempted
forwards per rank. It validates the exact prompt/output history, last input
token, absolute position and flattened logits mapping before bypassing graphs.
Failed attempts are not refunded; no reset/retry mechanism is provided.

Only a selected forward sets `bypass_hpu_graphs=True`. The original `use_graphs`
continues to control convolution pool writes and tensor-cache policy. The normal
HpuModelAdapter and sole forward remain intact: no second cache update. Python
boundary capture is installed only around this synchronous forward, removed
before blocking CPU flush, and cleared in `finally`, including failure paths.
Unselected forwards follow the normal graph path and perform no boundary tensor
work. No hooks are dynamically installed into a normal captured graph.

Each rank clones only selected token rows from initial mHC streams, every
attention mHC post and FFN mHC post, and final norm. Exclusive mode-0600 files
`layer-rankN-XX.pt` record rank, exact request mapping, baseline graph eligibility,
and diagnostic bypass. Boundary tensor payload is capped at **64 MiB per rank
across all attempts**; a boundary exceeding the remaining budget is skipped and
`truncated=True` is recorded. Inspect truncation before interpreting missing
boundaries. Mapping/setup, forward, or flush errors propagate rather than
silently retrying a partially executed forward. Files contain sensitive tokens
and activations; retain locally and remove manually after review.

Layer 0 additionally records (in execution order):

- `attention_mhc_pre_input`: mHC weighted input before RMSNorm, `[T, hidden]`.
- `attention_mhc_pre_post_a`: post weights, `[T, hc, 1]`.
- `attention_mhc_pre_comb_a`: combination weights, `[T, hc, hc]`.
- `attention_norm_input`: normalized input supplied to attention, `[T, hidden]`.
- `attention_raw_output`: self-attention return before `_mhc_post`, `[T, hidden]`.

These reuse the collector's dimension-0 flattened-token mapping, without reshaping
at call sites. Later layers retain only the original two post-mHC boundaries.
`unavailable_boundaries` explicitly records `attention_pre_o_proj`: the plugin's
`HpuGlm5NextSparseAttention` delegates to `HPUMultiHeadLatentAttentionWrapper`,
which inherits `MultiHeadLatentAttentionWrapper.forward` from upstream. The active
upstream implementation applies `o_proj` inside that inherited forward, so this
local-model-only extension does not intercept or monkeypatch it. The raw return
is **after** output projection, not the pre-projection attention core. If layer 0
is configured as KDA instead, its internal boundary is also explicitly marked
uninstrumented. No upstream or attention implementation was changed.

Raw sampler logits capture remains unchanged and available under its existing
gate and bounds for comparison if the graph-bypassed execution still diverges.
No HPU, server, restart, or inference request was used to validate this change.

CPU validation:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .venv/bin/python -m pytest -q --confcutdir=tests/unit_tests/worker \
  tests/unit_tests/worker/test_layer_diagnostic_ast.py \
  tests/unit_tests/worker/test_kda_state_diagnostic_ast.py \
  tests/unit_tests/worker/test_sampler_diagnostic_ast.py
```
