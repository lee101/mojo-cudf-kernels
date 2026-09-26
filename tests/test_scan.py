"""Parity tests for the prefix-scan kernels.

`cudf` cannot be imported here: it is a GPU-only wheel and the reference test
environment has no cudf. The expectations are therefore the libcudf semantics
transcribed independently in `tests/reference.py` -- the same algorithms written
a different way, not the same loops -- so a wrong stride, a wrong seed, a
swapped inclusive/exclusive, or a mask that is off by one all fail here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mojo_cudf_kernels as mck
from mojo_cudf_kernels import MAX, MIN, PRODUCT, SUM
from reference import scan_reference

RTOL = 1e-12


def _assert_scan(got, expect_values, expect_mask, context=""):
    assert np.array_equal(got.valid, expect_mask), f"validity mask {context}"
    np.testing.assert_allclose(
        got.data[expect_mask], expect_values[expect_mask], rtol=RTOL, atol=0.0,
        err_msg=f"scan values {context}",
    )


@pytest.mark.parametrize(
    "op,name", [(SUM, "cumsum"), (PRODUCT, "cumprod"), (MIN, "cummin"), (MAX, "cummax")]
)
def test_inclusive_scan_matches_reference(op, name):
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    expect, mask = scan_reference(data, None, op)
    got = getattr(mck, name)(data)
    _assert_scan(got, expect, mask, name)


def test_cumsum_on_random_data():
    rng = np.random.default_rng(11)
    data = rng.standard_normal(4096)
    expect = np.cumsum(data)
    got = mck.cumsum(data)
    # Mojo emits FMA, but a bare sum has no multiply to fuse, so the only
    # divergence is the order of the additions, which is still a few ulp.
    np.testing.assert_allclose(got.data, expect, rtol=1e-10, atol=1e-12)
    assert got.nulls == 0


def test_cummin_is_not_cummax():
    # A transposed min/max dispatch is the obvious bug here; the two must
    # disagree on this input and each must match its own reference.
    data = [5.0, 2.0, 9.0, 1.0, 7.0]
    lo = mck.cummin(data).data
    hi = mck.cummax(data).data
    np.testing.assert_array_equal(lo, np.minimum.accumulate(data))
    np.testing.assert_array_equal(hi, np.maximum.accumulate(data))
    assert not np.array_equal(lo, hi)


def test_scan_nulls_propagate_from_first_null():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    valid = np.array([1, 1, 0, 1, 1], dtype=bool)
    expect, mask = scan_reference(data, valid, SUM)
    got = mck.cumsum(mck.Column(data, valid))
    _assert_scan(got, expect, mask, "sum with nulls")
    # Everything at and after the first null is invalid, which is the rule
    # `mask_scan` writes into the output mask.
    assert got.valid.tolist() == [True, True, False, False, False]
    assert got.nulls == 3


def test_scan_leading_null_keeps_exclusive_row_valid():
    # cudf's mask_scan shifts the limit by one for an exclusive scan, so the
    # row at the first null's index is still valid. Losing that +1 is the
    # classic off-by-one in this kernel.
    data = [1.0, 2.0, 3.0]
    valid = np.array([0, 1, 1], dtype=bool)
    got = mck.exclusive_scan(mck.Column(data, valid), "sum")
    assert got.valid.tolist() == [True, False, False]
    assert got.nulls == 2
    expect, mask = scan_reference(data, valid, SUM, inclusive=False)
    _assert_scan(got, expect, mask, "exclusive with leading null")


def test_exclusive_scan_seeds_with_identity():
    data = [2.0, 3.0, 4.0, 5.0]
    got = mck.exclusive_scan(data, "sum")
    np.testing.assert_allclose(got.data, [0.0, 2.0, 5.0, 9.0], rtol=0, atol=0)
    prod = mck.exclusive_scan(data, "prod")
    np.testing.assert_allclose(prod.data, [1.0, 2.0, 6.0, 24.0], rtol=RTOL, atol=0)
    lo = mck.exclusive_scan(data, "min")
    assert lo.data[0] == math.inf
    np.testing.assert_allclose(lo.data[1:], [2.0, 2.0, 2.0], rtol=RTOL, atol=0)
    hi = mck.exclusive_scan(data, "max")
    assert hi.data[0] == -math.inf
    np.testing.assert_allclose(hi.data[1:], [2.0, 3.0, 4.0], rtol=RTOL, atol=0)


def test_inclusive_and_exclusive_differ_by_one_shift():
    rng = np.random.default_rng(3)
    data = rng.standard_normal(257)
    inclusive = mck.cumsum(data).data
    exclusive = mck.exclusive_scan(data, "sum").data
    np.testing.assert_allclose(exclusive[1:], inclusive[:-1], rtol=1e-12, atol=0)
    assert inclusive[-1] != exclusive[-1]


def test_scan_of_single_row_and_empty():
    one = mck.cumsum([7.5])
    assert one.data.tolist() == [7.5]
    empty = mck.cumsum(np.zeros(0))
    assert empty.size == 0 and empty.nulls == 0


def test_scan_over_31_rows_exercises_three_mask_bytes():
    # 31 rows spans four bitmask bytes; a mask sized `n // 8` instead of
    # `(n + 7) // 8` would drop the tail.
    data = list(np.arange(1.0, 32.0))
    got = mck.cumsum(data)
    np.testing.assert_allclose(got.data, np.cumsum(data), rtol=RTOL, atol=0)
    assert got.mask.size == 4


def test_scan_dispatch_rejects_non_scan_aggregations():
    with pytest.raises(ValueError):
        mck.scan([1.0, 2.0], "nope")
    with pytest.raises(ValueError):
        mck.exclusive_scan([1.0, 2.0], "mean")
