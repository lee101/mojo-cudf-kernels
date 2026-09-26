"""Validity bitmask kernels, mirroring `cudf::detail::null_mask` and the
`cudf::transform` mask conversions.

These are the bit-exact part of the port: a packed bitmask is bitwise work, so
these functions are asserted with `==` rather than a tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _lib
from .column import Column, pack_validity, unpack_validity

__all__ = [
    "bools_to_mask",
    "mask_to_bools",
    "count_set_bits",
    "set_null_mask",
    "nans_to_nulls",
]


def _as_valid(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError("validity flags must be a one-dimensional uint8 array")
    return np.ascontiguousarray(array)


def bools_to_mask(flags: Any, *, device: str = "cpu") -> np.ndarray:
    """Pack a uint8 validity array into cudf's `bitmask_type` layout.

    With `device="gpu"` the packing runs on the GPU when a device is available
    and there is enough free memory; otherwise it falls back to the CPU kernel.
    Returns `(bitmask, set_bits)`.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    source = _as_valid(flags)
    n = source.size
    out = np.zeros((n + 7) >> 3, dtype=np.uint8)
    if n == 0:
        return out, 0
    if device == "gpu":
        if _lib.lib().ck_bools_to_mask_gpu(_lib.baddr(source), n, _lib.baddr(out)) == 1:
            return out, int(unpack_validity(out, n).sum())
        _lib.gpu_fallback("bools_to_mask")
    set_bits = _lib.lib().ck_bools_to_mask(_lib.baddr(source), n, _lib.baddr(out))
    return out, int(set_bits)


def mask_to_bools(bitmask: Any, size: int) -> np.ndarray:
    """Unpack a cudf validity bitmask into a uint8 flag array.

    Rows beyond the bitmask's `size()` are invalid, matching cudf's rule that a
    short null mask is padded with nulls.
    """
    mask = np.ascontiguousarray(bitmask, dtype=np.uint8)
    if mask.ndim != 1:
        raise TypeError("validity bitmask must be one-dimensional")
    out = np.zeros(size, dtype=np.uint8)
    if size:
        _lib.lib().ck_mask_to_bools(
            _lib.baddr(mask), _lib.baddr(out), size, int(mask.size)
        )
    return out


def count_set_bits(bitmask: Any, size: int, *, device: str = "cpu") -> int:
    """Popcount over the first `size` bits of a validity bitmask.

    This is `cudf::detail::count_set_bits`, and it is what `Column.nulls` is
    built from, so the null count never costs a per-row Python scan.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    mask = np.ascontiguousarray(bitmask, dtype=np.uint8)
    if size <= 0:
        return 0
    if device == "gpu":
        total = np.zeros(1, dtype=np.int64)
        if _lib.lib().ck_count_set_bits_gpu(
            _lib.baddr(mask), size, total.ctypes.data
        ) == 1:
            return int(total[0])
        _lib.gpu_fallback("count_set_bits")
    return int(_lib.lib().ck_count_set_bits(_lib.baddr(mask), size))


def set_null_mask(bitmask: Any, begin: int, end: int, valid: bool) -> np.ndarray:
    """`cudf::detail::set_null_mask`: write `valid` into `[begin, end)`."""
    mask = np.ascontiguousarray(bitmask, dtype=np.uint8)
    if end < begin:
        raise ValueError("end must not precede begin")
    if end:
        _lib.lib().ck_set_null_mask(
            _lib.baddr(mask), int(begin), int(end), 1 if valid else 0
        )
    return mask


def nans_to_nulls(column: Any, *, device: str = "cpu") -> Column:
    """`cudf::transform::nans_to_nulls`: mark every NaN entry invalid.

    Infinities survive, because cudf tests for NaN specifically rather than for
    finiteness. With `device="gpu"` the elementwise pass runs on the GPU when
    one is available.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    source = column if isinstance(column, Column) else Column(column)
    n = source.size
    out = np.zeros((n + 7) >> 3, dtype=np.uint8)
    if n == 0:
        return Column.from_mask(np.zeros(0, dtype=np.float64), out)
    ran = False
    if device == "gpu":
        ran = _lib.lib().ck_nans_to_nulls_gpu(
            _lib.addr(source.data), _lib.baddr(out), n
        ) == 1
        if not ran:
            _lib.gpu_fallback("nans_to_nulls")
    if not ran:
        _lib.lib().ck_nans_to_nulls(_lib.addr(source.data), _lib.baddr(out), n)
    return Column.from_mask(source.data.copy(), out)
