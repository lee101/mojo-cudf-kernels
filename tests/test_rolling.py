"""Parity tests for the two rolling-window kernels."""

from __future__ import annotations

import numpy as np
import pytest

import mojo_cudf_kernels as mck
from mojo_cudf_kernels import MAX, MEAN, MIN, STD, SUM, VAR
from reference import rolling_range_reference, rolling_window_reference

RTOL = 1e-11


def _check(got, expect_values, expect_mask, context):
    assert np.array_equal(got.valid, expect_mask), f"validity mask {context}"
    np.testing.assert_allclose(
        got.data[expect_mask], expect_values[expect_mask], rtol=RTOL, atol=1e-12,
        err_msg=f"values {context}",
    )


@pytest.mark.parametrize("agg", ["sum", "min", "max", "mean", "var", "std"])
def test_fixed_window_matches_reference(agg):
    rng = np.random.default_rng(31)
    data = rng.standard_normal(300)
    op = {"sum": SUM, "min": MIN, "max": MAX, "mean": MEAN, "var": VAR, "std": STD}[agg]
    expect, mask = rolling_window_reference(data, None, 3, 3, 1, op, ddof=1)
    got = mck.rolling_window(data, 7, aggregation=agg)
    _check(got, expect, mask, agg)


def test_fixed_window_edges_are_clipped():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    got = mck.rolling_window(data, (1, 1), aggregation="sum")
    np.testing.assert_allclose(got.data, [3.0, 6.0, 9.0, 12.0, 9.0], rtol=0, atol=0)
    # A window that walked off the end instead of clipping would give NaN or 0.
    assert got.data[0] == 3.0 and got.data[-1] == 9.0


def test_centered_split_matches_cudf_series_rolling():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    got = mck.rolling_window(data, 3, center=True, aggregation="sum")
    expect, mask = rolling_window_reference(data, None, 1, 1, 1, SUM)
    _check(got, expect, mask, "centered window=3")
    trailing = mck.rolling_window(data, 3, center=False, aggregation="sum")
    expect, mask = rolling_window_reference(data, None, 2, 0, 1, SUM)
    _check(trailing, expect, mask, "trailing window=3")


def test_min_periods_suppresses_short_windows():
    data = [1.0, None, None, 4.0, 5.0, None, 7.0]
    valid = np.array([1, 0, 0, 1, 1, 0, 1], dtype=bool)
    expect, mask = rolling_window_reference(data, valid, 2, 2, 3, SUM)
    got = mck.rolling_window(mck.Column(data, valid), 5, min_periods=3, aggregation="sum")
    _check(got, expect, mask, "min_periods=3")
    # Windows 2..4, 4..6 hold three valid entries; the clipped edges hold fewer,
    # so a kernel that counted the window width rather than the valid count
    # would keep the edges and drop the middles.
    assert got.valid.tolist() == [False, False, True, False, True, True, False]


def test_rolling_skips_nulls_in_the_aggregate():
    data = [2.0, 0.0, 6.0]
    valid = np.array([1, 0, 1], dtype=bool)
    got = mck.rolling_window(mck.Column(data, valid), 3, min_periods=1, aggregation="sum")
    # Centred width 3: rows (0,1), (0,1,2), (1,2) with the null at row 1 skipped.
    np.testing.assert_allclose(got.data, [2.0, 8.0, 6.0], rtol=0, atol=0)
    assert got.valid.all()


def test_rolling_mean_uses_valid_count_not_window_width():
    data = [2.0, 0.0, 6.0, 10.0]
    valid = np.array([1, 0, 1, 1], dtype=bool)
    got = mck.rolling_window(mck.Column(data, valid), (1, 1), min_periods=1, aggregation="mean")
    # Row 1's window holds two valid entries out of three; dividing by the
    # window width instead of the valid count would give 8/3 rather than 4.
    np.testing.assert_allclose(got.data, [2.0, 4.0, 8.0, 8.0], rtol=0, atol=0)


def test_rolling_var_ddof_one_on_a_known_window():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Centred width 3 with clipped edges: rows (0,1), (0..2), (1..3), (2..4), (3,4).
    # Sample variances with ddof=1 are 0.5, 1, 1, 1, 0.5.
    got = mck.rolling_window(data, 3, aggregation="var")
    np.testing.assert_allclose(got.data, [0.5, 1.0, 1.0, 1.0, 0.5], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        mck.rolling_window(data, 3, aggregation="std").data,
        np.sqrt([0.5, 1.0, 1.0, 1.0, 0.5]),
        rtol=1e-12,
        atol=0,
    )


def test_range_window_matches_bisect_reference():
    orderby = [-5.0, -2.0, 0.0, 10.0, 20.0, 20.0, 36.0, 42.0, 73.0, 102.0]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    expect, mask = rolling_range_reference(values, None, orderby, 0.0, 0.0, False, False, 1, SUM)
    got = mck.rolling_range(values, orderby)
    _check(got, expect, mask, "current-row range window")
    # Rows 4 and 5 share orderby = 20, so both land in the same window and sum
    # to 11; every other key is unique and selects exactly its own row.
    np.testing.assert_allclose(
        got.data, [1.0, 2.0, 3.0, 4.0, 11.0, 11.0, 7.0, 8.0, 9.0, 10.0], rtol=0, atol=0
    )


def test_range_window_duplicate_keys_select_the_whole_group():
    orderby = [1.0, 1.0, 1.0, 2.0, 2.0]
    values = [1.0, 10.0, 100.0, 5.0, 50.0]
    # A current-row window over a tied key covers every row sharing the key, so
    # a bisect that only looked at the row itself would give 1, 10, 100, 5, 50.
    got = mck.rolling_range(values, orderby, aggregation="sum")
    np.testing.assert_allclose(
        got.data, [111.0, 111.0, 111.0, 55.0, 55.0], rtol=0, atol=0
    )


def test_range_window_deltas_and_open_ends():
    # Gaps of two in the orderby column make the closed and open bounds differ.
    orderby = [0.0, 2.0, 4.0, 6.0, 8.0]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    closed = mck.rolling_range(values, orderby, preceding=2.0, following=2.0)
    expect, mask = rolling_range_reference(
        values, None, orderby, 2.0, 2.0, False, False, 1, SUM
    )
    _check(closed, expect, mask, "closed +-2")
    np.testing.assert_allclose(closed.data, [3.0, 6.0, 9.0, 12.0, 9.0], rtol=0, atol=0)

    open_left = mck.rolling_range(
        values, orderby, preceding=2.0, following=2.0, preceding_open=True
    )
    # The preceding bound becomes exclusive, so the row sitting exactly on
    # `key - 2` drops out: row 1 loses value 1 and row 2 loses value 2.
    np.testing.assert_allclose(
        open_left.data, [3.0, 5.0, 7.0, 9.0, 5.0], rtol=0, atol=0
    )

    open_right = mck.rolling_range(
        values, orderby, preceding=2.0, following=2.0, following_open=True
    )
    # The following bound becomes exclusive, so the row on `key + 2` drops out.
    np.testing.assert_allclose(
        open_right.data, [1.0, 3.0, 5.0, 7.0, 9.0], rtol=0, atol=0
    )


def test_range_window_skips_nulls_and_honours_min_periods():
    orderby = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    valid = np.array([1, 0, 0, 1, 1, 1], dtype=bool)
    expect, mask = rolling_range_reference(
        values, valid, orderby, 2.0, 2.0, False, False, 3, MEAN
    )
    got = mck.rolling_range(
        mck.Column(values, valid), orderby,
        preceding=2.0, following=2.0, min_periods=3, aggregation="mean",
    )
    _check(got, expect, mask, "range window with nulls")


def test_range_window_requires_a_matching_orderby():
    with pytest.raises(ValueError):
        mck.rolling_range([1.0, 2.0], [1.0, 2.0, 3.0])


def test_rolling_rejects_bad_windows():
    with pytest.raises(ValueError):
        mck.rolling_window([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        mck.rolling_window([1.0, 2.0], (-1, 2))
    with pytest.raises(ValueError):
        mck.rolling_window([1.0, 2.0], 2, aggregation="median")
    with pytest.raises(ValueError):
        mck.rolling_range([1.0], [1.0], preceding=-1.0)
