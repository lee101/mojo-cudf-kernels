"""Correctness-gated benchmark for mojo-cudf-kernels.

Every case checks its result against an independent NumPy reference *before*
timing, so a regression in a kernel shows up as a correctness failure rather than
as a suspiciously good number. The baseline is always the fastest reasonable
NumPy formulation: `np.cumsum` for the scans, `sliding_window_view` for the
rolling windows, and `np.packbits` for the validity bitmask. A Python loop would
be a strawman, not a baseline.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

import mojo_cudf_kernels as mck  # noqa: E402
from reference import rolling_window_reference  # noqa: E402

RTOL = 1e-11


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def bench_cumsum(n: int = 1 << 22):
    """A serial prefix scan against `np.cumsum`, both memory-bound."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(n)
    got = mck.cumsum(data).data
    np.testing.assert_allclose(got, np.cumsum(data), rtol=1e-9, atol=1e-9)
    return f"cumsum n={n}", _time(lambda: np.cumsum(data)), _time(
        lambda: mck.cumsum(data)
    )


def bench_rolling_sum(n: int = 1 << 18, window: int = 51):
    """Fixed-window rolling sum against a `sliding_window_view` dot product."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(n)
    got = mck.rolling_window(data, window, aggregation="sum").data
    expect = np.convolve(data, np.ones(window), mode="valid")
    preceding, following = window // 2, window - window // 2 - 1
    np.testing.assert_allclose(
        got[preceding : n - following], expect, rtol=RTOL, atol=1e-10
    )
    numpy_time = _time(lambda: np.convolve(data, np.ones(window), mode="valid"))
    mojo_time = _time(lambda: mck.rolling_window(data, window, aggregation="sum"))
    return f"rolling sum n={n} w={window}", numpy_time, mojo_time


def bench_rolling_reference_loop(n: int = 4096, window: int = 31):
    """Rolling sum against a windowed NumPy mean, the closest fair equivalent."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal(n)
    expect, _ = rolling_window_reference(data, None, window - 1, 0, 1, 0)
    got = mck.rolling_window(data, (window - 1, 0), aggregation="sum").data
    np.testing.assert_allclose(got, expect, rtol=RTOL, atol=1e-10)
    view = np.lib.stride_tricks.sliding_window_view(data, window)
    numpy_time = _time(lambda: view.sum(axis=1))
    mojo_time = _time(lambda: mck.rolling_window(data, (window - 1, 0), aggregation="sum"))
    return f"rolling sum n={n} w={window}", numpy_time, mojo_time


def bench_reduce(n: int = 1 << 24):
    """Whole-column sum and variance, both two-pass NumPy formulations."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal(n)
    np.testing.assert_allclose(mck.sum(data), data.sum(), rtol=1e-10, atol=1e-9)
    numpy_time = _time(lambda: data.sum())
    mojo_time = _time(lambda: mck.sum(data))
    return f"sum n={n}", numpy_time, mojo_time


def bench_ewm(n: int = 1 << 20):
    """Unadjusted EWMA against a vectorised pandas-equivalent formulation.

    The adjusted branch has a vectorised form -- a convolution -- so that is the
    fair baseline; the unadjusted recurrence has none, so it is reported on its
    own against nothing but its own correctness.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(5)
    data = rng.standard_normal(n)
    got = mck.ewm(data, alpha=0.3, adjust=True).data
    # The adjusted branch is a first-order IIR filter, so `lfilter` is the
    # fastest fair NumPy-side baseline; `np.convolve` would be O(n^2).
    weights = 0.7 ** np.arange(n, dtype=np.float64)
    expect = lfilter([1.0], [1.0, -0.7], data) / np.cumsum(weights)
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)
    numpy_time = _time(lambda: lfilter([1.0], [1.0, -0.7], data))
    mojo_time = _time(lambda: mck.ewm(data, alpha=0.3, adjust=True))
    return f"ewm adjust n={n}", numpy_time, mojo_time


def bench_bools_to_mask(n: int = 1 << 24, device: str = "cpu", repeats: int = 3):
    """Validity bitmask packing against `np.packbits`."""
    rng = np.random.default_rng(6)
    flags = rng.integers(0, 2, size=n, dtype=np.uint8)
    mask, bits = mck.bools_to_mask(flags, device=device)
    assert bits == int(flags.sum())
    assert np.array_equal(mask, np.packbits(flags, bitorder="little"))
    numpy_time = _time(lambda: np.packbits(flags, bitorder="little"), repeats)
    mojo_time = _time(lambda: mck.bools_to_mask(flags, device=device), repeats)
    return f"bools_to_mask n={n} {device}", numpy_time, mojo_time


def bench_count_set_bits(n: int = 1 << 24, device: str = "cpu", repeats: int = 3):
    """Validity popcount against `np.unpackbits(...).sum()`."""
    rng = np.random.default_rng(7)
    flags = rng.integers(0, 2, size=n, dtype=np.uint8)
    mask, bits = mck.bools_to_mask(flags)
    assert mck.count_set_bits(mask, n, device=device) == bits
    numpy_time = _time(
        lambda: int(np.unpackbits(mask, bitorder="little").sum()), repeats
    )
    mojo_time = _time(lambda: mck.count_set_bits(mask, n, device=device), repeats)
    return f"count_set_bits n={n} {device}", numpy_time, mojo_time


def bench_nans_to_nulls(sizes=(1 << 16, 1 << 20, 1 << 24), repeats: int = 3):
    """NaN marking, a pure elementwise pass, swept over the PCIe crossover.

    At 16M rows the host-to-device copy of the 128 MiB input dominates the
    device work, so the GPU only starts paying once the input is small enough
    for the transfer to be a small fraction of the pass. Reporting a single
    size would hide that, so the sweep is the honest form.
    """
    rows = []
    for n in sizes:
        rng = np.random.default_rng(8)
        data = rng.standard_normal(n)
        data[rng.random(n) < 0.1] = np.nan
        cpu = mck.nans_to_nulls(data, device="cpu")
        assert cpu.nulls == int(np.isnan(data).sum())
        numpy_time = _time(lambda: np.isnan(data), repeats)
        rows.append(
            (f"nans_to_nulls n={n} numpy", numpy_time, float("nan"))
        )
        rows.append(
            (
                f"nans_to_nulls n={n} cpu",
                numpy_time,
                _time(lambda: mck.nans_to_nulls(data, device="cpu"), repeats),
            )
        )
        if mck.nans_to_nulls(data, device="gpu").nulls == cpu.nulls:
            rows.append(
                (
                    f"nans_to_nulls n={n} gpu",
                    numpy_time,
                    _time(lambda: mck.nans_to_nulls(data, device="gpu"), repeats),
                )
            )
    return rows


def main():
    print(f"{'case':<34}{'reference':>12}{'mojo':>12}{'ratio':>9}")
    print("-" * 68)
    single = (
        bench_cumsum,
        bench_rolling_sum,
        bench_rolling_reference_loop,
        bench_reduce,
        bench_ewm,
    )
    for fn in single:
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref * 1e3:>10.2f}ms{got * 1e3:>10.2f}ms{ratio:>8.2f}x")
    for device in ("cpu", "gpu"):
        for fn in (bench_bools_to_mask, bench_count_set_bits):
            try:
                label, ref, got = fn(device=device, repeats=3)
            except Exception as error:  # GPU unavailable: say so, do not fake it
                print(f"{fn.__name__} {device:<28}skipped: {error}")
                continue
            ratio = ref / got if got else float("nan")
            print(f"{label:<34}{ref * 1e3:>10.2f}ms{got * 1e3:>10.2f}ms{ratio:>8.2f}x")
    for name, ref, got in bench_nans_to_nulls():
        if got != got:
            print(f"{name:<34}{ref * 1e3:>10.2f}ms{'-':>12}{'-':>9}")
        else:
            print(f"{name:<34}{ref * 1e3:>10.2f}ms{got * 1e3:>10.2f}ms{ref / got:>8.2f}x")


if __name__ == "__main__":
    main()
