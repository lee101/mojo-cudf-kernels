"""Rolling windows, mirroring `cudf::detail::rolling`.

Two shapes are covered, because libcudf has two:

* `rolling_window` -- the fixed-index window of `rolling_fixed_window.cu`, where
  row `i` covers `[i - preceding, i + following]` clipped to the column.
* `rolling_range` -- the value-range window of `range_utils.cuh`, where the
  window is found by bisecting a separately sorted `orderby` column. This is
  the shape libcudf's `test_rolling.py` asserts against.

Both take `min_periods` and skip null entries, and both return a null whenever
the window holds no valid entry or fewer than `min_periods` of them. `var` and
`std` use libcudf's two-moment formula, as in the column reductions.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from . import _lib
from .column import Column
from .reductions import MAX, MEAN, MIN, STD, SUM, VAR

__all__ = ["rolling_window", "rolling_range"]

_ROLLING_OPS = {
    "sum": SUM,
    "min": MIN,
    "max": MAX,
    "mean": MEAN,
    "var": VAR,
    "std": STD,
}


def _op(aggregation: str) -> int:
    if aggregation not in _ROLLING_OPS:
        raise ValueError(f"unknown rolling aggregation {aggregation!r}")
    return _ROLLING_OPS[aggregation]


def rolling_window(
    column: Any,
    window: int | tuple[int, int] = 3,
    *,
    min_periods: int = 1,
    aggregation: str = "sum",
    ddof: int = 1,
    center: bool = True,
) -> Column:
    """Fixed-index rolling aggregation.

    `window` is either a total width or an explicit `(preceding, following)`
    pair. With an integer and `center=True` the window is split as cudf's
    `Series.rolling` splits it: `preceding = window // 2` and
    `following = window - window // 2 - 1`.
    """
    if isinstance(window, tuple):
        preceding, following = int(window[0]), int(window[1])
    else:
        width = int(window)
        if width < 1:
            raise ValueError("window must be at least 1")
        if center:
            preceding = width // 2
            following = width - width // 2 - 1
        else:
            preceding, following = width - 1, 0
    if preceding < 0 or following < 0:
        raise ValueError("window halves must be non-negative")
    op = _op(aggregation)
    source = column if isinstance(column, Column) else Column(column)
    n = source.size
    values = np.zeros(n, dtype=np.float64)
    out_mask = np.zeros((n + 7) >> 3, dtype=np.uint8)
    if n:
        _lib.lib().ck_rolling_window(
            _lib.addr(source.data),
            _lib.mask_addr(source.mask, n),
            n,
            preceding,
            following,
            int(min_periods),
            int(ddof),
            op,
            _lib.addr(values),
            _lib.mask_addr(out_mask),
        )
    return Column.from_mask(values, out_mask)


def rolling_range(
    column: Any,
    orderby: Any,
    *,
    preceding: float = 0.0,
    following: float = 0.0,
    preceding_open: bool = False,
    following_open: bool = False,
    min_periods: int = 1,
    aggregation: str = "sum",
    ddof: int = 1,
) -> Column:
    """Value-range rolling aggregation over a sorted `orderby` column.

    Row `i` aggregates every row whose `orderby` value lies in
    `(orderby[i] - preceding, orderby[i] + following]`, or with the open flags
    the half-open variants libcudf calls `BoundedOpen`.
    """
    if preceding < 0 or following < 0:
        raise ValueError("window deltas must be non-negative")
    op = _op(aggregation)
    source = column if isinstance(column, Column) else Column(column)
    order = _lib.f64(orderby)
    if order.size != source.size:
        raise ValueError("orderby must have the same length as the column")
    n = source.size
    values = np.zeros(n, dtype=np.float64)
    out_mask = np.zeros((n + 7) >> 3, dtype=np.uint8)
    if n:
        _lib.lib().ck_rolling_range(
            _lib.addr(order),
            _lib.addr(source.data),
            _lib.mask_addr(source.mask, n),
            n,
            ctypes.c_double(preceding),
            ctypes.c_double(following),
            1 if preceding_open else 0,
            1 if following_open else 0,
            int(min_periods),
            int(ddof),
            op,
            _lib.addr(values),
            _lib.mask_addr(out_mask),
        )
    return Column.from_mask(values, out_mask)
