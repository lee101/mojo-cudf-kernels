"""Prefix scan and whole-column reduction, mirroring libcudf's `cudf::reduction`.

The API shape follows cudf: `cumsum`, `cumprod`, `cummin`, `cummax` for the
inclusive scans that `cudf.core.scan.scan` exposes, plus the libcudf-only
exclusive scan, and `sum`/`product`/`min`/`max`/`mean`/`var`/`std`/`count` for
the single-value reductions.

`var` and `std` default to `ddof=1`, matching `cudf.Series.var`.
"""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np

from . import _lib
from .column import Column, pack_validity, unpack_validity

__all__ = [
    "SUM",
    "PRODUCT",
    "MIN",
    "MAX",
    "MEAN",
    "VAR",
    "STD",
    "COUNT",
    "center_of_mass",
    "cumsum",
    "cumprod",
    "cummin",
    "cummax",
    "scan",
    "exclusive_scan",
    "reduce",
    "sum",
    "product",
    "minimum",
    "maximum",
    "mean",
    "variance",
    "std",
    "count",
    "ewm",
]

SUM = 0
PRODUCT = 1
MIN = 2
MAX = 3
MEAN = 4
VAR = 5
STD = 6
COUNT = 7

_OP_NAMES = {
    SUM: "sum",
    PRODUCT: "prod",
    MIN: "min",
    MAX: "max",
    MEAN: "mean",
    VAR: "var",
    STD: "std",
    COUNT: "count",
}


def _check_op(op: int) -> int:
    if op not in (SUM, PRODUCT, MIN, MAX):
        raise ValueError(f"{op!r} is not a valid scan aggregation")
    return int(op)


def _scan(column: Column, op: int, inclusive: bool) -> Column:
    _check_op(op)
    n = column.size
    values = np.zeros(n, dtype=np.float64)
    out_mask = np.zeros((n + 7) >> 3, dtype=np.uint8)
    if n:
        _lib.lib().ck_scan(
            _lib.addr(column.data),
            _lib.mask_addr(column.mask, n),
            n,
            op,
            1 if inclusive else 0,
            _lib.addr(values),
            _lib.mask_addr(out_mask),
        )
    return Column.from_mask(values, out_mask)


def scan(column: Any, method: str = "scan") -> Column:
    """Inclusive prefix scan, as `cudf.core.scan.scan` computes it.

    `method` is one of `scan` (sum), `cumsum`, `cumprod`, `cummin`, `cummax`.
    """
    mapping = {
        "scan": SUM,
        "cumsum": SUM,
        "cumprod": PRODUCT,
        "cummax": MAX,
        "cummin": MIN,
    }
    if method not in mapping:
        raise ValueError(f"unknown scan method {method!r}")
    return _scan(Column(column) if not isinstance(column, Column) else column,
                 mapping[method], inclusive=True)


def exclusive_scan(column: Any, method: str = "sum") -> Column:
    """Exclusive prefix scan, as `plc.scan.exclusive` computes it.

    Unlike the inclusive form this has no `cudf.Series` spelling; it is exposed
    because libcudf ships it and the seeding rule (first value rather than the
    operation's identity) is the one that needs stating.
    """
    mapping = {"sum": SUM, "prod": PRODUCT, "max": MAX, "min": MIN}
    if method not in mapping:
        raise ValueError(f"unknown scan method {method!r}")
    source = column if isinstance(column, Column) else Column(column)
    return _scan(source, mapping[method], inclusive=False)


def cumsum(column: Any) -> Column:
    return _scan(column if isinstance(column, Column) else Column(column), SUM, True)


def cumprod(column: Any) -> Column:
    return _scan(column if isinstance(column, Column) else Column(column), PRODUCT, True)


def cummin(column: Any) -> Column:
    return _scan(column if isinstance(column, Column) else Column(column), MIN, True)


def cummax(column: Any) -> Column:
    return _scan(column if isinstance(column, Column) else Column(column), MAX, True)


def reduce(column: Any, op: int, ddof: int = 1) -> float | None:
    """Whole-column reduction. Returns None for an all-null column, which is
    what cudf returns for an empty reduction."""
    if op not in _OP_NAMES:
        raise ValueError(f"unknown aggregation {op!r}")
    if op == COUNT:
        source = column if isinstance(column, Column) else Column(column)
        return float(source.size - source.nulls)
    source = column if isinstance(column, Column) else Column(column)
    if source.size == 0:
        return None
    out = np.zeros(2, dtype=np.float64)
    valid = _lib.lib().ck_reduce(
        _lib.addr(source.data),
        _lib.mask_addr(source.mask, source.size),
        source.size,
        op,
        int(ddof),
        _lib.addr(out),
    )
    if valid == 0:
        return None
    return float(out[0])


def sum(column: Any) -> float | None:
    return reduce(column, SUM)


def product(column: Any) -> float | None:
    return reduce(column, PRODUCT)


def minimum(column: Any) -> float | None:
    return reduce(column, MIN)


def maximum(column: Any) -> float | None:
    return reduce(column, MAX)


def mean(column: Any) -> float | None:
    return reduce(column, MEAN)


def variance(column: Any, ddof: int = 1) -> float | None:
    return reduce(column, VAR, ddof)


def std(column: Any, ddof: int = 1) -> float | None:
    return reduce(column, STD, ddof)


def count(column: Any) -> float | None:
    source = column if isinstance(column, Column) else Column(column)
    return float(source.size - source.nulls)


def center_of_mass(
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> float:
    """`cudf.core.window.ewm.get_center_of_mass`, transcribed.

    The four parameters are mutually exclusive, exactly as cudf enforces, and
    the domain checks are cudf's own.
    """
    given = [value for value in (com, span, halflife, alpha) if value is not None]
    if len(given) > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")
    if com is not None:
        if com < 0:
            raise ValueError("comass must satisfy: comass >= 0")
        return float(com)
    if span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        return (span - 1) / 2
    if halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - math.exp(math.log(0.5) / halflife)
        return 1 / decay - 1
    if alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        return (1 - alpha) / alpha
    raise ValueError("Must pass one of comass, span, halflife, or alpha")


def ewm(
    column: Any,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    adjust: bool = True,
) -> Column:
    """Exponentially weighted moving average, as `cudf.Series.ewm(...).mean()`.

    cudf derives `beta = com / (com + 1)` from whichever of `com`, `span`,
    `halflife` or `alpha` is given, then runs either the INFINITE-history
    (adjusted) pair scan or the plain recurrence. It does not implement
    `ignore_na` or `min_periods`, so neither is offered here; the output has no
    nulls, matching the column cudf builds.
    """
    comass = center_of_mass(com, span, halflife, alpha)
    beta = comass / (comass + 1.0)
    source = column if isinstance(column, Column) else Column(column)
    n = source.size
    values = np.zeros(n, dtype=np.float64)
    if n:
        entry = _lib.lib().ck_ewm_adjusted if adjust else _lib.lib().ck_ewm
        entry(
            _lib.addr(source.data),
            _lib.mask_addr(source.mask, n),
            n,
            ctypes.c_double(beta),
            _lib.addr(values),
        )
    return Column(values)
