"""Tests for the validity-bitmask kernels.

These are bitwise operations, so every assertion here is exact (`==`), not a
tolerance. That is the one place in this port where `np.array_equal` is the
right assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_cudf_kernels as mck


def test_pack_unpack_round_trip_over_every_length_1_to_40():
    # 1..40 rows covers the partial trailing byte, and lengths 1, 8, 9, 16, 17
    # and 32 are exactly where an off-by-one in the byte count shows up.
    rng = np.random.default_rng(41)
    for size in range(1, 41):
        flags = rng.integers(0, 2, size=size, dtype=np.uint8)
        mask, set_bits = mck.bools_to_mask(flags)
        assert mask.size == (size + 7) // 8
        assert mask.dtype == np.uint8
        assert set_bits == int(flags.sum())
        assert np.array_equal(mask, np.packbits(flags, bitorder="little"))
        assert np.array_equal(mck.mask_to_bools(mask, size), flags)


def test_mask_bit_order_is_lsb_first():
    mask, _ = mck.bools_to_mask(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
    assert mask[0] == 0b0000_0001
    mask, _ = mck.bools_to_mask(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8))
    assert mask[0] == 0b1000_0000


def test_mask_to_bools_treats_a_short_mask_as_null():
    # cudf pads a null mask shorter than the column with nulls; a kernel that
    # read past the end of the buffer would report those rows valid.
    short = np.array([0b0000_0011], dtype=np.uint8)
    assert mck.mask_to_bools(short, 12).tolist() == [1, 1] + [0] * 10


def test_count_set_bits_is_a_popcount():
    rng = np.random.default_rng(42)
    for size in (1, 7, 8, 9, 100, 1023, 4096):
        flags = rng.integers(0, 2, size=size, dtype=np.uint8)
        mask, _ = mck.bools_to_mask(flags)
        assert mck.count_set_bits(mask, size) == int(flags.sum())
    assert mck.count_set_bits(np.zeros(3, dtype=np.uint8), 24) == 0
    assert mck.count_set_bits(np.full(3, 0xFF, dtype=np.uint8), 24) == 24


def test_set_null_mask_fills_a_range_only():
    mask = np.zeros(2, dtype=np.uint8)
    mck.set_null_mask(mask, 2, 13, True)
    expected = np.zeros(13, dtype=np.uint8)
    expected[2:13] = 1
    assert np.array_equal(mask, np.packbits(expected, bitorder="little"))
    mck.set_null_mask(mask, 3, 6, False)
    assert not mck.mask_to_bools(mask, 13)[[3, 4, 5]].any()
    assert mck.mask_to_bools(mask, 13)[[2, 6]].all()
    with pytest.raises(ValueError):
        mck.set_null_mask(mask, 5, 4, True)


def test_nans_to_nulls_marks_only_nans():
    data = [1.0, float("nan"), float("inf"), -float("inf"), 2.0, float("nan")]
    got = mck.nans_to_nulls(data)
    assert got.nulls == 2
    assert got.valid.tolist() == [True, False, True, True, True, False]
    # Infinities are not nulls, which a finiteness test would get wrong.


def test_nans_to_nulls_without_nans():
    got = mck.nans_to_nulls([1.0, 2.0, 3.0])
    assert got.nulls == 0 and got.valid.all()


def test_nans_to_nulls_of_empty_column():
    got = mck.nans_to_nulls(np.zeros(0))
    assert got.size == 0 and got.nulls == 0


def test_column_null_count_uses_the_kernel_popcount():
    rng = np.random.default_rng(43)
    valid = rng.integers(0, 2, size=1000, dtype=np.uint8).astype(bool)
    column = mck.Column(rng.standard_normal(1000), valid)
    assert column.nulls == int(1000 - valid.sum())
    assert column.size == 1000


def test_column_rejects_a_mismatched_mask_length():
    with pytest.raises(ValueError):
        # 9 rows need 2 mask bytes, not 1.
        mck.Column(np.zeros(9), mask=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError):
        mck.Column(np.zeros(2), valid=np.ones(2, bool), mask=np.zeros(1, dtype=np.uint8))


def test_validity_helpers_agree_with_the_kernel():
    rng = np.random.default_rng(44)
    valid = rng.integers(0, 2, size=257, dtype=np.uint8).astype(bool)
    packed = mck.pack_validity(valid)
    assert np.array_equal(packed, np.packbits(valid.astype(np.uint8), bitorder="little"))
    assert np.array_equal(mck.unpack_validity(packed, 257), valid)
    assert mck.unpack_validity(None, 4).all()
    assert mck.unpack_validity(np.zeros(0, dtype=np.uint8), 4).all()


def test_validity_rejects_wrong_dtypes():
    with pytest.raises(TypeError):
        mck.bools_to_mask(np.array([1.0, 0.0]))
    with pytest.raises(TypeError):
        mck.bools_to_mask(np.array([[1, 0]], dtype=np.uint8))
