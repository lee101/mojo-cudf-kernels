"""Parity tests for the whole-column reduction kernels."""

from __future__ import annotations

import numpy as np
import pytest

import mojo_cudf_kernels as mck
from mojo_cudf_kernels import MAX, MEAN, MIN, PRODUCT, STD, SUM, VAR
from reference import reduce_reference

RTOL = 1e-11


@pytest.mark.parametrize("op", [SUM, PRODUCT, MIN, MAX, MEAN])
def test_reduce_matches_reference(op):
    rng = np.random.default_rng(7)
    data = rng.standard_normal(2048)
    expect, count = reduce_reference(data, None, op)
    got = mck.reduce(data, op)
    assert got is not None
    np.testing.assert_allclose(got, expect, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_variance_and_std_match_reference(ddof):
    rng = np.random.default_rng(8)
    data = rng.standard_normal(512) + 5.0
    expect, _ = reduce_reference(data, None, VAR, ddof)
    np.testing.assert_allclose(mck.variance(data, ddof), expect, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        mck.std(data, ddof), np.sqrt(expect), rtol=1e-9, atol=1e-12
    )


def test_variance_of_a_known_sample():
    # ddof=0 of [2, 4, 4, 4, 5, 5, 7, 9] is exactly 4; ddof=1 is 32/7.
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    np.testing.assert_allclose(mck.variance(data, 0), 4.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(mck.variance(data, 1), 32.0 / 7.0, rtol=1e-12, atol=0)


def test_reduction_skips_nulls():
    data = [1.0, 0.0, 3.0, 4.0, 0.0]
    valid = np.array([1, 0, 1, 1, 0], dtype=bool)
    column = mck.Column(data, valid)
    assert column.nulls == 2
    expect, count = reduce_reference(data, valid, SUM)
    np.testing.assert_allclose(mck.sum(column), expect, rtol=0, atol=0)
    expect, _ = reduce_reference(data, valid, MAX)
    assert mck.maximum(column) == expect
    assert mck.count(column) == 3


def test_all_null_reduction_is_null():
    column = mck.Column([1.0, 2.0], np.array([0, 0], dtype=bool))
    assert mck.sum(column) is None
    assert mck.mean(column) is None
    assert mck.minimum(column) is None


def test_min_max_are_not_swapped():
    data = [7.0, 2.0, 9.0, 4.0]
    assert mck.minimum(data) == 2.0
    assert mck.maximum(data) == 9.0


def test_count_is_a_count_not_a_reduction():
    valid = np.array([1, 0, 1, 0], dtype=bool)
    column = mck.Column([1.0, 0.0, 3.0, 0.0], valid)
    assert mck.count(column) == 2.0
    assert mck.count(mck.Column(np.zeros(0))) == 0.0
    # A column with no validity mask has no nulls, whatever its values are.
    assert mck.count([1.0, float("nan"), 3.0]) == 3.0


def test_reduce_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        mck.reduce([1.0], 99)


def test_sum_of_a_long_column_stays_accurate():
    # A sequential sum loses precision; both sides are sequential here, so the
    # check is that the kernel agrees with the same-order NumPy sum.
    rng = np.random.default_rng(19)
    data = rng.standard_normal(1 << 18)
    np.testing.assert_allclose(mck.sum(data), data.sum(), rtol=1e-10, atol=1e-12)
