"""Backend/device helpers for NumPy/CuPy execution."""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

Device = Literal["auto", "cpu", "gpu"]
EngineBackend = Literal["native", "python"]


def _normalize_device(device: Optional[str]) -> Device:
    value = (device or "auto").strip().lower()
    if value not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'gpu'")
    return value  # type: ignore[return-value]


def default_device() -> Device:
    return _normalize_device(os.getenv("NEURAL_ENGINE_DEVICE", "auto"))


def default_engine_backend() -> EngineBackend:
    value = (os.getenv("NEURAL_ENGINE_BACKEND", "native") or "native").strip().lower()
    if value not in {"native", "python"}:
        warnings.warn(
            f"Unknown NEURAL_ENGINE_BACKEND='{value}', using 'native'.",
            RuntimeWarning,
        )
        return "native"
    return value  # type: ignore[return-value]


def cupy_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_backend(device: Optional[str] = None, warn: bool = True) -> Tuple[Any, Device, str, bool]:
    requested = _normalize_device(device or default_device())

    if requested == "cpu":
        return np, "cpu", "numpy", False

    if requested == "gpu":
        if cupy_available():
            return cp, "gpu", "cupy", True
        if warn:
            warnings.warn("GPU requested but CuPy/CUDA unavailable. Falling back to CPU.", RuntimeWarning)
        return np, "cpu", "numpy", False

    if cupy_available():
        return cp, "gpu", "cupy", True
    return np, "cpu", "numpy", False


def array_module(arr: Any) -> Any:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_device(value: Any, xp: Any, dtype: Optional[Any] = None) -> Any:
    if xp is np:
        return np.asarray(value, dtype=dtype)
    if cp is None:
        return np.asarray(value, dtype=dtype)
    return cp.asarray(value, dtype=dtype)


def to_cpu(value: Any) -> np.ndarray:
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def as_numpy(value: Any) -> np.ndarray:
    return to_cpu(value)
