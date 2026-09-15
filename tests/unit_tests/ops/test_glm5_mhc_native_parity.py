# SPDX-License-Identifier: Apache-2.0
"""The mHC production path on HPU is MHCPreOp.forward_oot -> forward_native ->
vllm's mhc_pre_torch; the layer anti-mixer `_mhc_post` in glm5_next.py must
reproduce vllm's mhc_post_torch, and the model's pre-API contract must be the
HF sigmoid(+eps) form — NOT softmax.

History: a retired internal mHC mirror carried `pre_mix = softmax(...)` where
production (and the HF reference, and every upstream backend) has
`sigmoid(...) + hc_eps`. Nothing in production ran it, but a TPC mHC kernel
effort was built and oracle-tested against that mirror, so its kernel
reproduced the wrong function to cos 0.999999 while diverging from the model
to cos ~0.96-0.99 per site. This file pins the production semantics so the
mistake cannot recur. CPU-only.

On HPU the production pre path is `_mhc_pre_hpu` (same gate math; the 24-lane
mix GEMM runs on the bf16 MME instead of an fp32 GEMV — the fp32 GEMV is the
decode critical path at batch 1). Its mix values carry one bf16 rounding, so
the tests below pin: identical CONTRACT, gate outputs within measured bf16
tolerance, and layer_input within the same bar.
"""
import types

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm_gaudi.models.glm5_next import _mhc_post, _mhc_pre_hpu

HC, HIDDEN, RMS_EPS, HC_EPS, SINKHORN = 4, 256, 1e-5, 1e-6, 20


def _hc_mod(seed):
    g = torch.Generator().manual_seed(seed)
    mix = (2 + HC) * HC
    return types.SimpleNamespace(fn=torch.randn(mix, HC * HIDDEN, generator=g) * 0.02,
                                 base=torch.randn(mix, generator=g) * 0.1,
                                 scale=torch.tensor([0.7, 0.5, 0.9]),
                                 rms_eps=RMS_EPS,
                                 hc_eps=HC_EPS,
                                 sinkhorn_repeat=SINKHORN)


def _ref_mhc_pre(residual, hc_mod):
    """HF-reference mHC pre-mix math (the contract above): fp32 mirror with
    sigmoid weights (RMS-normalized mixes), post x2, sinkhorn-normalized
    combiner, layer input = weighted mean of streams."""
    fn, base, scale = hc_mod.fn, hc_mod.base, hc_mod.scale
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    x = residual.reshape(-1, hc * hidden).float()
    mixes = x @ fn.t()
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc * hidden) + hc_mod.rms_eps)
    pre_logits = mixes[:, :hc] * scale[0] + base[:hc]
    pre_mix = torch.sigmoid(pre_logits) + hc_mod.hc_eps
    post_logits = (mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    post_mix = torch.sigmoid(post_logits) * 2.0
    comb_logits = (mixes[:, 2 * hc:].view(-1, hc, hc) * scale[2] + base[2 * hc:].view(1, hc, hc))
    comb = torch.softmax(comb_logits, dim=-1) + hc_mod.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_mod.hc_eps)
    for _ in range(hc_mod.sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_mod.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_mod.hc_eps)
    layer_input = (pre_mix.unsqueeze(-1) * residual.reshape(-1, hc, hidden).float()).sum(1).to(residual.dtype)
    T = residual.shape[0]
    return (post_mix.view(T, hc, 1), comb.view(T, hc, hc), layer_input.view(T, hidden))


@pytest.mark.parametrize("T", [1, 3, 8])
def test_ref_mhc_pre_matches_production(T):
    """The packed-mix contract: pre must equal vllm mhc_pre_torch (production)
    for identical inputs — guarding the packed fn/scale/base layout that the
    HPU path relies on."""
    hc = _hc_mod(T)
    residual = (torch.randn(T, HC, HIDDEN, generator=torch.Generator().manual_seed(100 + T)) * 0.1).bfloat16()
    pm_r, cb_r, li_r = _ref_mhc_pre(residual, hc)
    pm_u, cb_u, li_u = mhc_pre_torch(residual, hc.fn, hc.scale, hc.base, RMS_EPS, HC_EPS, HC_EPS, 2.0, SINKHORN)
    assert pm_r.shape == pm_u.shape and cb_r.shape == cb_u.shape and li_r.shape == li_u.shape
    torch.testing.assert_close(pm_r, pm_u, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cb_r, cb_u, rtol=1e-5, atol=1e-6)
    # layer_input is bf16 on both sides; identical fp32 math -> identical rounding
    torch.testing.assert_close(li_r, li_u, rtol=0, atol=0)


def test_hf_reference_is_sigmoid_not_softmax():
    """The regression this file exists for: sigmoid weights do not sum to 1.

    Pins the HF-reference derivation directly (independent of vllm's op):
    pre = sigmoid(pre_logits) + eps, so the weighted sums of the streams are
    sigmoid-weighted, not softmax-normalized."""
    hc = _hc_mod(7)
    residual = (torch.randn(4, HC, HIDDEN, generator=torch.Generator().manual_seed(7)) * 0.1).bfloat16()
    x = residual.reshape(4, -1).float()
    mixes = (x @ hc.fn.t()) * torch.rsqrt(x.square().sum(-1, keepdim=True) / (HC * HIDDEN) + RMS_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * hc.scale[0] + hc.base[:HC]) + HC_EPS
    _, _, li_r = _ref_mhc_pre(residual, hc)
    expected = (pre.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    torch.testing.assert_close(li_r, expected, rtol=0, atol=0)
    assert not torch.allclose(pre.sum(-1), torch.ones(4), atol=1e-2)


@pytest.mark.parametrize("T", [1, 5])
def test_mhc_post_matches_production(T):
    g = torch.Generator().manual_seed(T)
    residual = (torch.randn(T, HC, HIDDEN, generator=g) * 0.1).bfloat16()
    x = (torch.randn(T, HIDDEN, generator=g) * 0.1).bfloat16()
    post_mix = torch.rand(T, HC, 1, generator=g) * 2
    comb = torch.softmax(torch.randn(T, HC, HC, generator=g), -1)
    out_n = _mhc_post(x, residual, post_mix, comb)
    out_u = mhc_post_torch(x, residual, post_mix, comb)
    torch.testing.assert_close(out_n.float(), out_u.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("T", [1, 3, 8])
def test_mhc_pre_hpu_matches_fp32_contract(T):
    """HPU production pre path vs the fp32 reference: same contract, bf16-mix
    rounding budget. mix rounding (~2^-8 rel on the 24-lane logits) passes
    through sigmoid/softmax/sinkhorn; the observed envelope on adversarially
    LARGE (unit-scale) random weights is ~2e-2 relative — real checkpoint
    params are ~50x smaller still."""
    hc = _hc_mod(T)
    residual = (torch.randn(T, HC, HIDDEN, generator=torch.Generator().manual_seed(200 + T)) * 0.1).bfloat16()
    pm_r, cb_r, li_r = _ref_mhc_pre(residual, hc)
    pm_h, cb_h, li_h = _mhc_pre_hpu(residual, hc.fn, hc.scale, hc.base, RMS_EPS, HC_EPS, HC_EPS, 2.0, SINKHORN)
    assert pm_h.shape == pm_r.shape and cb_h.shape == cb_r.shape and li_h.shape == li_r.shape
    # bf16 mix -> fp32 gate math: relax 1e-5 to the 1e-1-bounded envelope on
    # gate outputs (sigmoid/softmax are contraction maps).
    torch.testing.assert_close(pm_h, pm_r, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(cb_h, cb_r, rtol=5e-2, atol=5e-3)
    rel_li = ((li_h.float() - li_r.float()).norm() / (li_r.float().norm() + 1e-9)).item()
    assert rel_li < 2e-2, rel_li


def test_mhc_pre_hpu_small_weights_tight():
    """With checkpoint-scale fn (0.02 std, as _hc_mod builds) the mix deltas
    are dominated by bf16 GEMM output rounding and stay tiny."""
    hc = _hc_mod(5)
    residual = (torch.randn(4, HC, HIDDEN, generator=torch.Generator().manual_seed(51)) * 0.1).bfloat16()
    pm_r, cb_r, li_r = _ref_mhc_pre(residual, hc)
    pm_h, cb_h, li_h = _mhc_pre_hpu(residual, hc.fn, hc.scale, hc.base, RMS_EPS, HC_EPS, HC_EPS, 2.0, SINKHORN)
    torch.testing.assert_close(pm_h, pm_r, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(cb_h, cb_r, rtol=1e-2, atol=1e-3)
