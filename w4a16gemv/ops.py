"""The compiled kernels, loaded from the CMake build (W4A16GEMV_BUILD_DIR, default build/release)."""

import os
import sys
from pathlib import Path

import torch  # noqa: F401  loads libtorch before w4a16gemv_ops, which links against it

REPO_ROOT = Path(__file__).resolve().parents[1]

_build_dir = Path(os.environ.get("W4A16GEMV_BUILD_DIR", "build/release"))
if not _build_dir.is_absolute():
    _build_dir = REPO_ROOT / _build_dir
sys.path.insert(0, str(_build_dir / "python"))

import w4a16gemv_ops  # noqa: E402

sequential = w4a16gemv_ops.sequential
naive = w4a16gemv_ops.naive
coalesced = w4a16gemv_ops.coalesced
vectorized = w4a16gemv_ops.vectorized
load_only = w4a16gemv_ops.load_only

# The three GEMV kernels, slowest first. All share the signature in csrc/w4a16.h.
KERNELS = {
    "naive": naive,
    "coalesced": coalesced,
    "vectorized": vectorized,
}

# info(m, **block_config) -> registers, spills and occupancy.
KERNEL_INFO = {
    "naive": w4a16gemv_ops.naive_info,
    "coalesced": w4a16gemv_ops.coalesced_info,
    "vectorized": w4a16gemv_ops.vectorized_info,
}
