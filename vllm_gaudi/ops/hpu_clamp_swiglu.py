# SPDX-License-Identifier: Apache-2.0
"""TPC clamp-SwiGLU dispatch for the GLM clamped-SwiGLU MoE activation.

GLM silu+swiglu_limit models need the transformers Glm5Next clamped SwiGLU:
clamp the gate BEFORE silu (one-sided, max=limit), clamp the up projection
symmetrically (±limit), compute in fp32, emit bf16. The pure-torch reference
lives in ``vllm_gaudi.ops.hpu_fused_moe._silu_clamp_expert_act``; a dedicated
TPC-C kernel (``vllm_gaudi/ops/tpc_clamp_swiglu``, PR2) computes the same
function on a single TPC engine pass.

Dispatch rules for :func:`clamp_swiglu`:

- ``VLLM_GLM_TPC_CLAMP`` (default ``"auto"``): the TPC op is used when it is
  ``"1"`` or ``"auto"`` AND the op loaded successfully. Any other value
  (``"0"``, ``"off"``, ...) forces the pure-torch fallback regardless of
  kernel availability. The env var is read per call (host-side constant, no
  tensor-data dependence), so flipping it takes effect immediately.
- The TPC op additionally requires ``h`` to be bf16, non-empty, and contiguous
  (the kernel's contract); anything else (e.g. fp32 chunks, empty slices)
  silently falls back to the bit-identical torch math.
- The fallback reproduces ``_silu_clamp_expert_act`` exactly: fp32 internal,
  gate ``clamp(max=limit)`` before silu, up ``clamp(-limit, limit)``, cast
  back to ``h.dtype`` — so behavior is identical with or without the kernel.

Graph capture: both resolved paths are static-shape, host-branch-free element
regions — no ``.item()``/``.tolist()``/data-dependent control flow — and are
safe to capture in HPU graphs once the module-level loader state is settled
(it is, after the first call outside capture).

Loader discovery: ``VLLM_GLM_TPC_CLAMP_LOADER`` names a Python file exposing
``load_clamp_swiglu() -> op(h, limit)`` (default: the packaged
``vllm_gaudi/ops/tpc_clamp_swiglu/loader.py``). The import is lazy and cached; any
failure simply marks the TPC path unavailable. Call
:func:`refresh_tpc_clamp` to retry after the loader lands.
"""

import importlib.util
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_LOADER_PATH = os.path.join(os.path.dirname(__file__), "tpc_clamp_swiglu", "loader.py")
_LOADER_MODULE_NAME = "vllm_gaudi.ops._tpc_clamp_swiglu_loader"

# True iff the TPC op loaded successfully on the most recent load attempt.
# This is plain kernel availability: it does NOT include the VLLM_GLM_TPC_CLAMP
# env gate, which is evaluated per call in clamp_swiglu().
TPC_CLAMP_AVAILABLE: bool = False

_TPC_OP: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None
_TPC_LOAD_ATTEMPTED = False


def _loader_path() -> str:
    path = os.environ.get("VLLM_GLM_TPC_CLAMP_LOADER", "").strip()
    return path or _DEFAULT_LOADER_PATH


def refresh_tpc_clamp(force: bool = True) -> bool:
    """(Re)attempt to load the TPC clamp-SwiGLU op from the loader path.

    Called lazily on the first :func:`clamp_swiglu` dispatch; call it
    explicitly (e.g. after the loader file lands) to retry. Updates
    :data:`TPC_CLAMP_AVAILABLE`. Any exception while importing/executing the
    loader or calling ``load_clamp_swiglu()`` counts as "unavailable".
    """
    global TPC_CLAMP_AVAILABLE, _TPC_OP, _TPC_LOAD_ATTEMPTED
    if _TPC_LOAD_ATTEMPTED and not force:
        return TPC_CLAMP_AVAILABLE
    _TPC_LOAD_ATTEMPTED = True
    TPC_CLAMP_AVAILABLE = False
    # The kernel extension's op registration ABORTS the process (C++
    # terminate, uncatchable) outside lazy mode — never attempt the load
    # unless PT_HPU_LAZY_MODE=1; eager processes use the torch fallback.
    if os.environ.get("PT_HPU_LAZY_MODE", "0") != "1":
        return TPC_CLAMP_AVAILABLE
    _TPC_OP = None
    path = _loader_path()
    try:
        spec = importlib.util.spec_from_file_location(_LOADER_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build import spec for loader file {path!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        op = module.load_clamp_swiglu()
        if not callable(op):
            raise TypeError(f"load_clamp_swiglu() returned non-callable {op!r}")
        _TPC_OP = op
        TPC_CLAMP_AVAILABLE = True
        logger.info("[GLM] TPC clamp-swiglu kernel loaded from %s", path)
    except Exception as exc:  # loader may fail in arbitrary ways; degrade
        # Loud on purpose, same reasoning as the fused-MoE arming diagnostic in
        # hpu_fp8.py: the torch fallback is bit-identical, so a failed load is
        # invisible at runtime and silently gives up the kernel's speedup.
        logger.warning("[GLM] TPC clamp-swiglu kernel unavailable (%s); using the "
                       "pure-torch fallback", exc)
    return TPC_CLAMP_AVAILABLE


def _tpc_mode_allows() -> bool:
    mode = os.environ.get("VLLM_GLM_TPC_CLAMP", "auto").strip().lower()
    return mode in ("1", "auto")


def _torch_fallback(h: torch.Tensor, limit: float) -> torch.Tensor:
    """Exact _silu_clamp_expert_act math: clamp-before-silu gate, symmetric
    up clamp, fp32 internal, cast back to h.dtype."""
    d = h.shape[-1] // 2
    g = h[..., :d].float().clamp(max=limit)
    u = h[..., d:].float().clamp(min=-limit, max=limit)
    return (F.silu(g) * u).to(h.dtype)


def clamp_swiglu(h: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU on ``h`` of shape ``[..., 2I]`` -> ``[..., I]``.

    Dispatches to the TPC op when allowed (see module docstring) and when
    ``h`` is a non-empty contiguous bf16 tensor; otherwise runs the exact
    pure-torch fallback. The dispatch branches only on host-side constants
    (env var, dtype, contiguity flags) — never on tensor *values* — so the
    selected path is capturable end-to-end in HPU graphs.
    """
    if not _TPC_LOAD_ATTEMPTED:
        refresh_tpc_clamp(force=False)
    if (_TPC_OP is not None and _tpc_mode_allows() and h.dtype == torch.bfloat16 and h.device.type == "hpu"
            and h.numel() > 0 and h.is_contiguous()):
        lead = h.shape[:-1]
        # Under compiled lazy graphs the custom op may receive a reshape VIEW
        # whose storage the bridge cannot bind ("Neither storage attached to
        # input tensor"). clone() materializes own storage; the copy is a few
        # us against the 25 us kernel.
        h2 = h.reshape(-1, h.shape[-1]).clone()
        out = _TPC_OP(h2, float(limit))
        return out if h.dim() == 2 else out.reshape(*lead, out.shape[-1])
    return _torch_fallback(h, limit)
