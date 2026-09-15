# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPC clamp-SwiGLU dispatch wrapper
(vllm_gaudi.ops.hpu_clamp_swiglu).

Validates that the pure-torch fallback reproduces the exact transformers
Glm5Next clamped-SwiGLU semantics (clamp gate BEFORE silu, symmetric up
clamp, fp32 internal, bf16 out) against an fp32 oracle, and that the TPC
kernel path — when its loader is available — matches the same oracle and the
fallback within 2 bf16 ulps. Kernel-absent cases skip cleanly.

Run (CPU-only cases run anywhere; TPC cases need a Gaudi node with the
kernel built, see vllm_gaudi/ops/tpc_clamp_swiglu/):
  python -m pytest tests/unit_tests/ops/test_hpu_clamp_swiglu.py -q
"""

import pytest
import torch
import torch.nn.functional as F

import vllm_gaudi.ops.hpu_clamp_swiglu as clamp_mod
from vllm_gaudi.ops.hpu_clamp_swiglu import clamp_swiglu

DEV = "hpu" if torch.hpu.is_available() else "cpu"
LIMIT = 7.0
PEAK = 60.0  # |h| reaches 60 -> clamp at 7 verifiably binds on both halves


def _make_h(T: int, i_dim: int) -> torch.Tensor:
    """bf16 [T, 2I] chunk with amax pinned at PEAK (deterministic)."""
    torch.manual_seed(1234 + T * 31 + i_dim)
    raw = torch.randn(T, 2 * i_dim, device=DEV, dtype=torch.float32)
    return (raw * (PEAK / raw.abs().amax())).bfloat16()


def _reference(h: torch.Tensor, limit: float) -> torch.Tensor:
    """fp32 oracle with transformers clamp-then-silu semantics."""
    d = h.shape[-1] // 2
    g = h[..., :d].float().clamp(max=limit)
    u = h[..., d:].float().clamp(min=-limit, max=limit)
    return F.silu(g) * u


def _assert_clamp_binds(h: torch.Tensor, out: torch.Tensor, i_dim: int) -> None:
    """Sanity: the clamp is actually active for this input."""
    assert h.float().abs().amax() >= 2 * LIMIT  # premise: inputs way past limit
    assert h[..., :i_dim].float().amax() > LIMIT  # gate side exceeds the clamp
    assert h[..., i_dim:].float().abs().amax() > LIMIT  # up side exceeds the clamp
    bound = F.silu(torch.tensor(LIMIT)).item() * LIMIT
    bound_bf16 = float(torch.tensor(bound).bfloat16())  # fp32 bound rounds up in bf16
    assert out.float().abs().amax() <= bound_bf16 + 1e-6  # bounded, not exploded
    # unclamped math would differ materially -> clamp demonstrably binding
    unclamped = F.silu(h[..., :i_dim].float()) * h[..., i_dim:].float()
    ref = _reference(h, LIMIT)
    assert (unclamped - ref).norm() > 0.05 * ref.norm()


def _bf16_ulp(x32: torch.Tensor) -> torch.Tensor:
    """Per-element bf16 spacing (as fp32) of values held in fp32 x32."""
    ax = x32.abs()
    subnormal_floor = torch.full_like(ax, 2.0**-133)
    e = torch.floor(torch.log2(ax.clamp(min=2.0**-126)))
    return torch.where(ax >= 2.0**-126, torch.exp2(e - 7.0), subnormal_floor)


def _assert_within_bf16_ulps(a: torch.Tensor, b: torch.Tensor, n: int) -> float:
    a32, b32 = a.float(), b.float()
    assert torch.equal(torch.isnan(a32), torch.isnan(b32))
    diff = (a32 - b32).abs()
    tol = n * _bf16_ulp(b32) + 8 * 2.0**-133
    worst = (diff / tol).max().item()
    assert bool((diff <= tol).all()), f"exceeded {n} bf16 ulps (worst ratio {worst:.2f})"
    return worst


@pytest.mark.parametrize("i_dim", [1024, 2048])
@pytest.mark.parametrize("T", [1, 128])
def test_fallback_matches_fp32_reference(T, i_dim, monkeypatch):
    monkeypatch.setenv("VLLM_GLM_TPC_CLAMP", "0")  # force the torch path
    h = _make_h(T, i_dim)
    out = clamp_swiglu(h, LIMIT)
    assert out.shape == (T, i_dim) and out.dtype == torch.bfloat16
    ref = _reference(h, LIMIT)
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten(), dim=0).item()
    assert cos >= 0.9999, f"cos={cos} (T={T} i_dim={i_dim})"
    _assert_clamp_binds(h, out, i_dim)
    print(f"PASS fallback T={T} i_dim={i_dim} cos={cos:.6f}")


@pytest.mark.parametrize("i_dim", [1024, 2048])
@pytest.mark.parametrize("T", [1, 128])
def test_tpc_matches_fp32_reference(T, i_dim, monkeypatch):
    monkeypatch.setenv("VLLM_GLM_TPC_CLAMP", "auto")
    if not clamp_mod.refresh_tpc_clamp() or not clamp_mod.TPC_CLAMP_AVAILABLE:
        pytest.skip("TPC clamp-swiglu kernel not available (loader missing/broken)")
    h = _make_h(T, i_dim)
    out = clamp_swiglu(h, LIMIT)
    assert out.shape == (T, i_dim) and out.dtype == torch.bfloat16
    ref = _reference(h, LIMIT)
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten(), dim=0).item()
    assert cos >= 0.9999, f"cos={cos} (T={T} i_dim={i_dim})"
    _assert_clamp_binds(h, out, i_dim)
    print(f"PASS tpc T={T} i_dim={i_dim} cos={cos:.6f}")


@pytest.mark.parametrize("i_dim", [1024, 2048])
@pytest.mark.parametrize("T", [1, 128])
def test_tpc_vs_fallback_ulp(T, i_dim, monkeypatch):
    if not clamp_mod.refresh_tpc_clamp() or not clamp_mod.TPC_CLAMP_AVAILABLE:
        pytest.skip("TPC clamp-swiglu kernel not available (loader missing/broken)")
    h = _make_h(T, i_dim)
    monkeypatch.setenv("VLLM_GLM_TPC_CLAMP", "0")
    fb = clamp_swiglu(h, LIMIT)
    monkeypatch.setenv("VLLM_GLM_TPC_CLAMP", "auto")
    tpc = clamp_swiglu(h, LIMIT)
    worst = _assert_within_bf16_ulps(tpc, fb, 2)
    print(f"PASS ulp T={T} i_dim={i_dim} worst-tolerance-ratio={worst:.3f}")


def test_env_zero_forces_fallback(monkeypatch):
    """VLLM_GLM_TPC_CLAMP=0 must take the torch path even with the kernel
    loaded: output bitwise-equal to the fallback math on the same input."""
    monkeypatch.setenv("VLLM_GLM_TPC_CLAMP", "0")
    h = _make_h(128, 1024)
    out = clamp_swiglu(h, LIMIT)
    ref = clamp_mod._torch_fallback(h, LIMIT)
    assert torch.equal(out, ref), "env=0 did not reproduce the fallback bitwise"
    assert isinstance(clamp_mod.TPC_CLAMP_AVAILABLE, bool)  # introspection knob
    # an unset env defaults to "auto" without raising
    monkeypatch.delenv("VLLM_GLM_TPC_CLAMP", raising=False)
    out_default = clamp_swiglu(h, LIMIT)
    assert out_default.shape == ref.shape and not torch.isnan(out_default).any()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
