"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib
import warnings
from typing import Any

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-cudf-kernels.so"

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list[Any], Any]] = {
    "ck_scan": ([I, I, I, I, I, I, I], None),
    "ck_reduce": ([I, I, I, I, I, I], I),
    "ck_ewm": ([I, I, I, F, I], None),
    "ck_ewm_adjusted": ([I, I, I, F, I], None),
    "ck_rolling_window": ([I, I, I, I, I, I, I, I, I, I], None),
    "ck_rolling_range": (
        [I, I, I, I, F, F, I, I, I, I, I, I, I],
        None,
    ),
    "ck_bools_to_mask": ([I, I, I], I),
    "ck_mask_to_bools": ([I, I, I, I], I),
    "ck_count_set_bits": ([I, I], I),
    "ck_set_null_mask": ([I, I, I, I], None),
    "ck_nans_to_nulls": ([I, I, I], I),
    "ck_nans_to_nulls_gpu": ([I, I, I], I),
    "ck_bools_to_mask_gpu": ([I, I, I], I),
    "ck_count_set_bits_gpu": ([I, I, I], I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"{_LIB_PATH} does not exist; run `bash build/build.sh` first"
            )
        _library = ctypes.CDLL(str(_LIB_PATH))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value: Any) -> np.ndarray:
    """An owned, C-contiguous float64 view of `value`."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("column kernels take one-dimensional columns")
    return np.ascontiguousarray(array)


def addr(array: np.ndarray) -> int:
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address


_ALL_VALID: np.ndarray = np.zeros(0, dtype=np.uint8)


def mask_addr(bitmask: np.ndarray | None, size: int = 0) -> int:
    """Address of a packed validity bitmask.

    A Mojo pointer is non-nullable, so address 0 must never be dereferenced.
    cudf spells "no nulls" as a null mask pointer, which is fine in C++ but not
    across this ABI, so a null mask is passed as a shared all-ones buffer
    instead. The kernels only ever read it.
    """
    global _ALL_VALID
    if bitmask is None:
        needed = (size + 7) >> 3
        if _ALL_VALID.size < needed:
            _ALL_VALID = np.full(max(needed, 1 << 12), 0xFF, dtype=np.uint8)
        return int(_ALL_VALID.ctypes.data)
    if bitmask.dtype != np.uint8 or not bitmask.flags.c_contiguous:
        raise TypeError("validity bitmasks must be C-contiguous uint8 arrays")
    return int(bitmask.ctypes.data)



def baddr(array: np.ndarray) -> int:
    """Address of a C-contiguous uint8 buffer (the bitmask kernels)."""
    if array.dtype != np.uint8 or not array.flags.c_contiguous:
        raise TypeError("bitmask buffers must be C-contiguous uint8 arrays")
    address = int(array.ctypes.data)
    if array.size == 0:
        # A zero-length array may still hand back a valid pointer, but the
        # kernels are never called with an empty buffer.
        raise ValueError("bitmask kernels are not called with empty buffers")
    return address

def gpu_fallback(name: str) -> None:
    warnings.warn(
        f"Mojo GPU execution of {name} was unavailable; using the CPU kernel",
        RuntimeWarning,
        stacklevel=3,
    )
