# Incoming KDA state diagnostic (review-only)

No server was started for this change. No sentinel was created.

The existing `VLLM_DIAG_SAMPLER_DIR` safety gate must pass (direct child of
`/tmp`, owner-only directory mode 0700, regular owner-only `ENABLED` mode
0600, synchronous scheduling, TP8/PP1/DP1, no speculation, not warmup).
State capture additionally requires a regular owner-owned `STATE_ENABLED`
file with mode 0600. Create that sentinel only after service readiness.
A sentinel already present when the runner is constructed permanently
blocks state capture for that runner. Removing either sentinel disables
capture without changing sampler behavior.

Limits are per rank: nine attempted forwards, three admitted requests,
positions 0–2 per request, nine request-position attempts, and 256 MiB of
persisted tensor payload. Invalid mappings and failed writes do not refund
attempts. Requests must be observed at output position zero to be admitted.
Use one serial request followed by two concurrent requests; unrelated work
or long intermediate chunked prefills can consume the attempt budget.
There is deliberately no retry/reset mechanism.

Files are exclusive `state-rankN-XX.pt`, mode 0600. Each rank snapshots its
actual KDA modules' `kv_cache[0]` convolution and `kv_cache[1]` recurrent
pools. Selected rows are independently owned and synchronously copied to
CPU before the sole normal forward. No graph bypass or second forward is
introduced. Full padded state maps and flags are retained for collision
analysis. Raw prefill state is not zeroed/masked by `has_initial_states_p`.
Request records include prompt/output history, output position, actual
selected input token and absolute position, row and flat logit index.
Unmerged rectangular inputs and exact history/position/logit agreement are
required; mismatches fail closed with a warning, not guessed mapping.

These files contain sensitive request text as token IDs and model state;
keep them local with the enforced owner-only permissions. Remove them
manually when review is complete.

CPU validation (no plugin or HPU backend autoload):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .venv/bin/python -m pytest -q --confcutdir=tests/unit_tests/worker \
  tests/unit_tests/worker/test_kda_state_diagnostic_ast.py \
  tests/unit_tests/worker/test_sampler_diagnostic_ast.py
```
