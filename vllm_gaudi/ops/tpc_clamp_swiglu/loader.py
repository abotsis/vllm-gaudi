# SPDX-License-Identifier: Apache-2.0
"""Loader for the clamp_swiglu_fwd_bf16 TPC kernel on Gaudi 2.

Usage:
    from loader import load_clamp_swiglu
    op = load_clamp_swiglu()           # builds glue .so + torch ext, sets GC_KERNEL_PATH
    out = op(h, limit=10.0)            # h: [T, 2I] bf16 contiguous on HPU

Contract:
  - h must be bf16, contiguous, last dim even and >= 2. Any leading dims are
    flattened into T (the kernel is a flat rowwise op).
  - limit is a host-side fp32 scalar kernel parameter (default 10.0,
    production value; passed via the custom-op params struct, NOT a tensor).
  - Fails with ClampSwigluLoadError if any stage of the build/load fails.
"""

import os
import subprocess
from pathlib import Path

import torch
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
GLUE_SO = HERE / "libclamp_swiglu_kernels.so"
# Torch plugin lib dir (ext_src.cpp links against libhabana_pytorch_plugin.so).
habana_lib = Path(torch.__file__).resolve().parent.parent / "habana_frameworks" / "torch" / "lib"

# Set GC_KERNEL_PATH at import time — BEFORE any habana_frameworks import in
# the host process (the graph compiler captures its kernel search path at init;
# setting it later has no effect and GUID resolution fails with synStatus 26).
_existing = os.environ.get("GC_KERNEL_PATH", "")
if GLUE_SO.exists() and str(GLUE_SO) not in _existing.split(":"):
    os.environ["GC_KERNEL_PATH"] = str(GLUE_SO) + (":" + _existing if _existing else "")


class ClampSwigluLoadError(RuntimeError):
    """Raised when the clamp_swiglu TPC kernel cannot be built or loaded."""


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, cwd=HERE, **kw)


def _build_all():
    # 1) kernel + glue .so
    habana_inc = Path(torch.__file__).resolve().parent.parent / "habana_frameworks" / "torch" / "include"
    if not habana_inc.joinpath("hpu_custom_op.h").exists():
        raise ClampSwigluLoadError(f"hpu_custom_op.h not found under {habana_inc}")

    try:
        _run(["bash", str(HERE / "build.sh")])
    except subprocess.CalledProcessError as e:
        raise ClampSwigluLoadError(f"kernel/glue build failed (rc={e.returncode})") from e

    try:
        (HERE / "build_ext").mkdir(exist_ok=True)
        cpp_extension.load(
            name="clamp_swiglu_ext",
            sources=[str(HERE / "ext_src.cpp")],
            extra_include_paths=[str(habana_inc)],
            extra_ldflags=[f"-L{habana_lib}", "-l:libhabana_pytorch_plugin.so", "-Wl,-rpath," + str(habana_lib)],
            extra_cflags=["-std=c++17", "-O2"],
            verbose=False,
            build_directory=str(HERE / "build_ext"),
        )
    except Exception as e:  # noqa: BLE001 — surface any failure as the specific error
        raise ClampSwigluLoadError(f"torch extension build/load failed: {e}") from e


_OPS = None

_REGISTRY_KEY = ("vllm_gaudi.tpc_clamp_swiglu.loader", str(GLUE_SO))


def _ops_registry():
    """Process-global store for ops loaded by ANY loader-module instance.

    The loader file is exec'd under several spec module names (the package
    early-loader and :mod:`vllm_gaudi.ops.hpu_clamp_swiglu` each build their
    own module object); module globals do not dedupe them. Without a shared
    registry, a second instance re-runs ``cpp_extension.load`` and the
    TORCH_LIBRARY registration aborts the process ('already registered').
    """
    import sys as _sys
    import types as _types
    store = _sys.modules.get("vllm_gaudi._tpc_clamp_ops_registry")
    if store is None:
        store = _types.ModuleType("vllm_gaudi._tpc_clamp_ops_registry")
        store.ops = {}
        _sys.modules["vllm_gaudi._tpc_clamp_ops_registry"] = store
    return store.ops


def load_clamp_swiglu(force_rebuild=False):
    """Build (if needed) and load the op. Returns a python callable."""
    global _OPS
    if _OPS is not None and not force_rebuild:
        return _OPS
    cached = _ops_registry().get(_REGISTRY_KEY)
    if cached is not None and not force_rebuild:
        _OPS = cached
        return _OPS

    if force_rebuild or not GLUE_SO.exists():
        _build_all()

    # GC_KERNEL_PATH must be set BEFORE habana loads its kernel libraries:
    # put our lib first so its GUID wins over any stock kernel of same name.
    existing = os.environ.get("GC_KERNEL_PATH", "")
    mine = str(GLUE_SO)
    if mine not in existing.split(":"):
        os.environ["GC_KERNEL_PATH"] = mine + (":" + existing if existing else "")

    try:
        import torch  # noqa: F401
        import habana_frameworks.torch.core as htcore  # noqa: F401 — loads HPU plugin
    except Exception as e:  # noqa: BLE001
        raise ClampSwigluLoadError(f"habana_frameworks import failed: {e}") from e

    if not GLUE_SO.exists():
        raise ClampSwigluLoadError(f"glue library missing: {GLUE_SO}")

    # build (cached) and import the extension via cpp_extension.load — it both
    # compiles-if-needed and dlopys the module exactly once per process.
    import torch
    from torch.utils import cpp_extension

    habana_inc = Path(torch.__file__).resolve().parent.parent / "habana_frameworks" / "torch" / "include"
    habana_lib = habana_inc.parent / "lib"
    plugin_so = habana_lib / "libhabana_pytorch_plugin.so"
    if not plugin_so.exists():
        raise ClampSwigluLoadError(f"habana plugin not found: {plugin_so}")

    # Pre-load the HPU plugin RTLD_GLOBAL by its FULL PATH (same object habana
    # itself loads later — dlopen is idempotent by inode) so the extension's
    # undefined habana::custom_op symbols resolve at import time.
    import ctypes

    try:
        ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise ClampSwigluLoadError(f"cannot pre-load {plugin_so}: {e}") from e

    try:
        (HERE / "build_ext").mkdir(exist_ok=True)
        # NOTE: we deliberately do NOT link libhabana_pytorch_plugin.so — habana
        # loads it via ctypes with a full path (RTLD_GLOBAL); adding it as
        # DT_NEEDED makes ld resolve the SONAME to a *second* copy of the lib
        # and its static registrations run twice ("already registered" abort).
        # Undefined symbols here resolve against the in-process plugin at dlopen.
        cpp_extension.load(
            name="clamp_swiglu_ext",
            sources=[str(HERE / "ext_src.cpp")],
            extra_include_paths=[str(habana_inc)],
            extra_ldflags=["-Wl,--allow-shlib-undefined"],
            extra_cflags=["-std=c++17", "-O2"],
            verbose=False,
            build_directory=str(HERE / "build_ext"),
        )
    except Exception as e:  # noqa: BLE001
        raise ClampSwigluLoadError(f"torch extension load failed: {e}") from e

    if not hasattr(torch.ops.custom_op, "clamp_swiglu"):
        raise ClampSwigluLoadError("torch.ops.custom_op.clamp_swiglu missing after load")

    def _op(h, limit=10.0):
        if h.device.type != "hpu":
            raise ClampSwigluLoadError(f"input must be on hpu, got {h.device}")
        if h.dtype != torch.bfloat16:
            raise ClampSwigluLoadError(f"input must be bf16, got {h.dtype}")
        if not h.is_contiguous():
            raise ClampSwigluLoadError("input must be contiguous")
        if h.dim() < 1 or h.shape[-1] % 2 != 0:
            raise ClampSwigluLoadError(f"last dim must be even, got shape {tuple(h.shape)}")
        return torch.ops.custom_op.clamp_swiglu(h, float(limit))

    _OPS = _op
    _ops_registry()[_REGISTRY_KEY] = _OPS
    return _OPS


if __name__ == "__main__":
    op = load_clamp_swiglu()
    import torch

    h = torch.randn(8, 4096, device="hpu", dtype=torch.bfloat16) * 5
    out = op(h)
    htcore = __import__("habana_frameworks.torch.core", fromlist=["x"])
    htcore.mark_step()
    print("smoke OK:", out.shape, out.dtype)
