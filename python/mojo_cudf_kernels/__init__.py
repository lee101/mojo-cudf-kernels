"""mojo-cudf-kernels: the compute-oriented subset of libcudf's column kernels.

The numeric inner loops of RAPIDS cudf -- prefix scans, exponentially weighted
moving averages, column reductions, rolling windows and the validity bitmask
transforms -- compiled to one shared library and callable from Python.

The package name keeps the `_kernels` suffix so it imports alongside the real
`cudf`, which is a GPU-only library. Nothing here imports `cudf`: cudf needs
CUDA and a GPU, so the reference behaviour in the test suite is transcribed from
the libcudf sources in `cpp/src` rather than executed. See the README for the
exact source file each kernel came from.
"""

from __future__ import annotations

from . import _lib
from .column import Column, pack_validity, unpack_validity
from .reductions import (
    COUNT,
    MAX,
    MEAN,
    MIN,
    PRODUCT,
    STD,
    SUM,
    VAR,
    center_of_mass,
    count,
    cummax,
    cummin,
    cumprod,
    cumsum,
    ewm,
    exclusive_scan,
    maximum,
    mean,
    minimum,
    product,
    reduce,
    scan,
    std,
    sum,
    variance,
)
from .rolling import rolling_range, rolling_window
from .validity import (
    bools_to_mask,
    count_set_bits,
    mask_to_bools,
    nans_to_nulls,
    set_null_mask,
)

__version__ = "0.1.0"

__all__ = [
    "Column",
    "MAX",
    "MEAN",
    "MIN",
    "PRODUCT",
    "STD",
    "SUM",
    "VAR",
    "COUNT",
    "bools_to_mask",
    "center_of_mass",
    "count",
    "count_set_bits",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "ewm",
    "exclusive_scan",
    "lib",
    "mask_to_bools",
    "maximum",
    "mean",
    "minimum",
    "nans_to_nulls",
    "pack_validity",
    "product",
    "reduce",
    "rolling_range",
    "rolling_window",
    "scan",
    "set_null_mask",
    "std",
    "sum",
    "unpack_validity",
    "variance",
]


def lib():
    """The loaded `ctypes.CDLL`, so callers can check the build is present."""
    return _lib.lib()
