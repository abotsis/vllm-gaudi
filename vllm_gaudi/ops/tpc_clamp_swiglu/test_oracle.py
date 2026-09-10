# SPDX-License-Identifier: Apache-2.0
"""Oracle for clamp_swiglu_fwd_bf16 TPC kernel vs references.

References (computed on CPU from the identical bf16 input, mirroring
vllm_gaudi/ops/hpu_fused_moe.py::_silu_clamp_expert_act):
  fp32 ref  : g=h[:,:i_dim].float().clamp(max=L); u=h[:,i_dim:].float().clamp(-L,L);
              out32 = silu(g)*u                        (no bf16 rounding)
  torch epi : (silu(g)*u).to(bf16)                     (the 5-op eager epilogue)

Acceptance per case:
  cos(TPC out, fp32 ref) >= 0.9999   AND   max ulp(TPC out, torch epi) <= 2
Also asserts the clamp verifiably binds (output differs from unclamped silu).
Run: PT_HPU_LAZY_MODE=1 ... python test_oracle.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import load_clamp_swiglu  # noqa: E402 — MUST precede habana import (GC_KERNEL_PATH)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import habana_frameworks.torch.core as htcore  # noqa: E402

LIMIT = 10.0


def torch_epilogue_cpu(h_bf16: torch.Tensor, i_dim: int) -> torch.Tensor:
    g = h_bf16[..., :i_dim].float().clamp(max=LIMIT)
    u = h_bf16[..., i_dim:].float().clamp(min=-LIMIT, max=LIMIT)
    return (F.silu(g) * u).to(torch.bfloat16)


def fp32_ref_cpu(h_bf16: torch.Tensor, i_dim: int) -> torch.Tensor:
    g = h_bf16[..., :i_dim].float().clamp(max=LIMIT)
    u = h_bf16[..., i_dim:].float().clamp(min=-LIMIT, max=LIMIT)
    return F.silu(g) * u


def unclamped_ref_cpu(h_bf16: torch.Tensor, i_dim: int) -> torch.Tensor:
    g = h_bf16[..., :i_dim].float()
    u = h_bf16[..., i_dim:].float()
    return F.silu(g) * u


def bf16_mono(t: torch.Tensor) -> torch.Tensor:
    """Map bf16 bit patterns to a sign-magnitude monotonic integer scale.
    +0.0 and -0.0 are numerically equal — collapse both to 0 (TPC mul yields
    +0.0 where IEEE torch yields -0.0; identical value, must not count as
    ulp distance)."""
    bits = t.view(torch.int16).to(torch.int32)
    mono = torch.where(bits < 0, 0x8000 - bits, bits)
    return torch.where(bits & 0x7FFF == 0, torch.zeros_like(mono), mono)


def max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((bf16_mono(a) - bf16_mono(b)).abs().max().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # manual f64, no eps clamp: F.cosine_similarity clamps the denominator at
    # eps=1e-8, which turns ~1e-22-magnitude vectors (neg-skew case) into
    # meaningless ~1e-22 'similarities'. f32 would also underflow the dot.
    af, bf = a.double().flatten(), b.double().flatten()
    den = af.norm() * bf.norm()
    return 0.0 if den == 0 else float(af.dot(bf) / den)


def main() -> int:
    op = load_clamp_swiglu()

    rows = []
    all_pass = True

    def run_case(name: str, T: int, i_dim: int, h32: torch.Tensor) -> None:
        nonlocal all_pass
        h_cpu = h32.to(torch.bfloat16)
        h = h_cpu.to("hpu")
        out = op(h, LIMIT)
        htcore.mark_step()
        torch.hpu.synchronize()
        out_cpu = out.cpu()

        epi = torch_epilogue_cpu(h_cpu, i_dim)
        ref32 = fp32_ref_cpu(h_cpu, i_dim)

        c = cosine(out_cpu, ref32)
        u = max_ulp(out_cpu, epi)
        ok = (c >= 0.9999) and (u <= 2)
        all_pass &= ok
        rows.append((name, T, i_dim, c, u, ok))

    gen = torch.Generator().manual_seed(1234)

    for i_dim in (1024, 2048):
        for T in (1, 8, 128, 2048):
            # magnitudes up to 60: gaussian*20 clipped to +-60 (clips at 3 sigma)
            h32 = (torch.randn(T, 2 * i_dim, generator=gen) * 20.0).clamp(-60.0, 60.0)
            # force exact boundary/adversarial values in row 0:
            h32[0, :i_dim] = torch.linspace(-60.0, 60.0, i_dim)
            h32[0, i_dim:] = torch.linspace(60.0, -60.0, i_dim)
            run_case(f"rand+lin T={T}", T, i_dim, h32)

    # all-negative / all-positive / heavy-clamp mass distributions
    for i_dim in (2048, ):
        h32 = torch.full((8, 2 * i_dim), -55.0) + torch.randn(8, 2 * i_dim, generator=gen)
        run_case("neg-skew", 8, i_dim, h32)
        h32 = torch.full((8, 2 * i_dim), 55.0) + torch.randn(8, 2 * i_dim, generator=gen)
        run_case("pos-skew", 8, i_dim, h32)
        h32 = torch.randn(8, 2 * i_dim, generator=gen) * 0.01  # tiny values, no clamp
        run_case("tiny", 8, i_dim, h32)

    # tail-exercising shape: i_dim = 8*128 + 40
    i_dim = 1096
    h32 = (torch.randn(4, 2 * i_dim, generator=gen) * 20.0).clamp(-60.0, 60.0)
    h32[0, :i_dim] = torch.linspace(-60.0, 60.0, i_dim)
    run_case("tail i_dim=1096", 4, i_dim, h32)

    # ---- clamp must verifiably bind ----
    i_dim = 2048
    h32 = torch.zeros(4, 2 * i_dim)
    h32[:, :i_dim] = 50.0  # gate far above limit everywhere
    h32[:, i_dim:] = torch.linspace(-60.0, 60.0, i_dim)
    h_cpu = h32.to(torch.bfloat16)
    out = op(h_cpu.to("hpu"), LIMIT)
    htcore.mark_step()
    torch.hpu.synchronize()
    out_cpu = out.cpu()

    uncl = unclamped_ref_cpu(h_cpu, i_dim)
    clamped = torch_epilogue_cpu(h_cpu, i_dim)
    diff_unclamped = (out_cpu.float() - uncl).abs().max().item()
    u_vs_epi = max_ulp(out_cpu, clamped)
    binds = diff_unclamped > 1.0 and u_vs_epi <= 2
    rows.append(("clamp-binds", 4, i_dim, float("nan"), u_vs_epi, binds))
    all_pass &= binds
    print(f"clamp-binds: max|TPC - unclamped_silu| = {diff_unclamped:.3f} (must be > 1)")
    print(f"clamp-binds: TPC vs clamped epilogue  = {u_vs_epi} ulp (must be <= 2)")

    print(f"\n{'case':<16} {'T':>5} {'i_dim':>5} {'cos(vs f32)':>12} {'ulp(vs epi)':>12}  verdict")
    for name, T, i_dim, c, u, ok in rows:
        cs = f"{c:.6f}" if c == c else "   n/a  "
        print(f"{name:<16} {T:>5} {i_dim:>5} {cs:>12} {u:>12}  {'PASS' if ok else 'FAIL'}")

    print("\nORACLE:", "ALL PASS" if all_pass else "FAILURES PRESENT")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
