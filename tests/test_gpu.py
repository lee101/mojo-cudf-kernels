"""Tests for the guarded GPU paths.

The contract the brief requires is not "the GPU runs" but "the GPU runs when it
can and the CPU runs when it cannot, and the two agree". So these tests assert
the result is correct either way, and assert the fallback is reached with a
warning when the GPU entry point declines.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mojo_cudf_kernels as mck
from mojo_cudf_kernels import _lib


def _ran(result: int) -> bool:
    return result == 1


def test_nans_to_nulls_gpu_matches_cpu():
    rng = np.random.default_rng(51)
    data = rng.standard_normal(200_000)
    data[rng.random(200_000) < 0.1] = np.nan
    cpu = mck.nans_to_nulls(data, device="cpu")
    gpu = mck.nans_to_nulls(data, device="gpu")
    assert gpu.nulls == cpu.nulls
    assert np.array_equal(gpu.valid, cpu.valid)


def test_bools_to_mask_gpu_matches_cpu():
    rng = np.random.default_rng(52)
    flags = rng.integers(0, 2, size=500_000, dtype=np.uint8)
    cpu_mask, cpu_bits = mck.bools_to_mask(flags, device="cpu")
    gpu_mask, gpu_bits = mck.bools_to_mask(flags, device="gpu")
    assert gpu_bits == cpu_bits
    assert np.array_equal(gpu_mask, cpu_mask)


def test_count_set_bits_gpu_matches_cpu():
    rng = np.random.default_rng(53)
    flags = rng.integers(0, 2, size=1_000_000, dtype=np.uint8)
    mask, bits = mck.bools_to_mask(flags, device="cpu")
    assert mck.count_set_bits(mask, flags.size, device="cpu") == bits
    assert mck.count_set_bits(mask, flags.size, device="gpu") == bits


def test_gpu_entry_points_decline_an_over_budget_request():
    # The 2 GiB allocation guard runs before anything is allocated, so an
    # absurd length returns 0 without touching the device. That is the
    # "fall back to the CPU" signal the shim keys on.
    library = _lib.lib()
    flags = np.ones(8, dtype=np.uint8)
    out = np.zeros(8, dtype=np.uint8)
    assert library.ck_bools_to_mask_gpu(_lib.baddr(flags), 1 << 40, _lib.baddr(out)) == 0
    mask = np.full(8, 0xFF, dtype=np.uint8)
    total = np.zeros(1, dtype=np.int64)
    assert library.ck_count_set_bits_gpu(_lib.baddr(mask), 1 << 40, total.ctypes.data) == 0
    assert library.ck_nans_to_nulls_gpu(
        _lib.addr(np.zeros(8)), _lib.baddr(mask), 1 << 40
    ) == 0


def test_empty_column_short_circuits_without_touching_the_device():
    # No device can be created for a zero-length buffer, so the shim returns
    # before asking, rather than warning about a fallback it never needed.
    empty = np.zeros(0, dtype=np.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask, bits = mck.bools_to_mask(empty, device="gpu")
    assert mask.size == 0 and bits == 0
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert mck.nans_to_nulls(np.zeros(0), device="gpu").size == 0


def test_fallback_warning_names_the_kernel():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _lib.gpu_fallback("bools_to_mask")
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "bools_to_mask" in str(caught[0].message)


def test_gpu_entry_points_exist_in_the_shared_library():
    library = _lib.lib()
    for name in (
        "ck_nans_to_nulls_gpu",
        "ck_bools_to_mask_gpu",
        "ck_count_set_bits_gpu",
    ):
        assert hasattr(library, name), f"{name} is not exported"
    # The sequential kernels deliberately have no GPU variant: a scan and a
    # rolling window are recurrences, and faking them on a device would cost
    # O(n) work per thread, which is slower than the serial kernel.
    for name in ("ck_scan_gpu", "ck_rolling_window_gpu", "ck_rolling_range_gpu"):
        assert not hasattr(library, name), f"{name} should not exist"
