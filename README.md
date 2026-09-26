# mojo-cudf-kernels

The compute-oriented subset of [RAPIDS cudf](https://github.com/rapidsai/cudf)'s
column kernels, with the inner loops written in Mojo and callable from Python.
Prefix scans, exponentially weighted moving averages, column reductions,
rolling windows and the validity-bitmask transforms all live in one compiled
shared library.

`cudf` is a GPU-only wheel and is **not** importable in this environment, so
nothing here depends on it. Every kernel was transcribed from the libcudf
sources named in the table below, and the test suite checks each one against an
independent NumPy reference written from the same specification. The
Python package is named `mojo_cudf_kernels`, so it sits alongside the real
`cudf` when both are installed.

```python
import numpy as np
import mojo_cudf_kernels as mck

s = mck.cumsum([3.0, 1.0, 4.0, 1.0, 5.0])          # [3, 4, 8, 9, 14]
mck.sum([3.0, 1.0, 4.0])                          # 8.0
mck.rolling_window(np.arange(7.0), 3).data         # [1, 3, 5, 9, 12, 15, 11]
mck.ewm([1.0, 2.0, 3.0], alpha=0.5, adjust=True).data
```

## Covered subset

| area | implemented API | libcudf source |
| --- | --- | --- |
| Prefix scan | `cumsum`, `cumprod`, `cummin`, `cummax`, `scan`, `exclusive_scan` | `cpp/src/reductions/scan/scan.cuh`, `scan_inclusive.cu` |
| Null-propagating scan mask | first-null rule, exclusive `+1` shift | `mask_scan` in `scan_inclusive.cu` |
| EWMA | `ewm(com/span/halflife/alpha, adjust)`, `center_of_mass` | `cpp/src/reductions/scan/ewm.cu` |
| Reductions | `sum`, `product`, `minimum`, `maximum`, `mean`, `variance`, `std`, `count`, `reduce` | `cpp/src/reductions/{sum,product,min,max,mean,var,std}.cu`, `compound.cuh` |
| Rolling, fixed index | `rolling_window(window, min_periods, ddof, center)` | `cpp/src/rolling/detail/rolling_fixed_window.cu` |
| Rolling, value range | `rolling_range(orderby, preceding, following, *open)` | `cpp/src/rolling/detail/range_utils.cuh` |
| Validity bitmask | `bools_to_mask`, `mask_to_bools`, `count_set_bits`, `set_null_mask` | `cpp/src/transform/{bools_to_mask,mask_to_bools}.cu`, `cudf/detail/null_mask.hpp` |
| NaN to null | `nans_to_nulls` | `cpp/src/transform/nans_to_nulls.cu` |
| Column model | `Column`, `pack_validity`, `unpack_validity` | `cudf::column`, `cudf::bitmask_type` |
| GPU | `device="gpu"` on `nans_to_nulls`, `bools_to_mask`, `count_set_bits` | same kernels, device-side |

### Not implemented, and why

* **Grouped rolling windows.** `grouped_range_rolling_window` with grouping
  keys. The ungrouped range kernel is here; the group bounds are a labelling
  pass, not arithmetic.
* **`row_bit_count`.** Looked at and deliberately skipped. For a table of
  fixed-width columns its own `hierarchy_info::simple_per_row_size` collapses
  the result to a constant per row, so a "kernel" for it would be a
  multiplication dressed up as compute.
* **All of `cpp/src/strings/`, `join/`, `groupby/hash/`, `quantiles/`,
  `io/`, `stream_compaction`.** Regex matching, hashing, parquet and CSV
  parsing. String and IO work, not a numeric inner loop.
* **`cudf::reduction`'s decimal128 and dictionary paths.** Only the `float64`
  column type and its validity mask are carried across the FFI. A wider type
  would be a separate ABI, not a parameter.
* **`cudf::transform` compile (`transform/jit/kernel.cu`) and the whole Python
  layer** — `Series`, `DataFrame`, `cudf.core.scan`. Those are expression
  plumbing over the kernels above; the kernels are the compute.
* **Column types other than `float64`.** An `int64` column is exactly
  representable but is not exposed, so nothing claims a dtype it does not
  carry. Values wider than `float64`, and complex dtypes, are rejected.

## Install

```bash
pixi run build     # -> dist/libmojo-cudf-kernels.so
pixi run test
pixi run bench
```

Outside a Pixi task, `PYTHONPATH=python` and a built
`dist/libmojo-cudf-kernels.so` are all the package needs.

## Performance

Best-of-N wall clock in one process. Every case asserts numerical agreement
with the reference *before* timing, so a broken kernel fails the benchmark
rather than reporting a good number. Baselines are the fastest reasonable NumPy
formulation, not a Python loop: `np.cumsum`, `np.convolve`,
`scipy.signal.lfilter`, `np.packbits`, `np.isnan`. Measured on a shared
Xeon E5-2697 v4 with an RTX 5090; the box is loaded, so treat these as
indicative.

| case | reference | mojo-cudf-kernels | result |
| --- | ---: | ---: | ---: |
| cumsum n=4194304 | 26.52 ms | 62.23 ms | 0.43x |
| rolling sum n=262144 w=51 | 10.62 ms | 62.91 ms | 0.17x |
| rolling sum n=4096 w=31 | 0.19 ms | 0.51 ms | 0.37x |
| sum n=16777216 | 16.26 ms | 97.98 ms | 0.17x |
| ewm adjust n=1048576 | 6.75 ms | 5.87 ms | 1.15x |
| bools_to_mask n=16777216 cpu | 2.54 ms | 58.94 ms | 0.04x |
| count_set_bits n=16777216 cpu | 111.14 ms | 2.21 ms | 50.27x |
| bools_to_mask n=16777216 gpu | 4.08 ms | 96.61 ms | 0.04x |
| count_set_bits n=16777216 gpu | 20.76 ms | 1.60 ms | 13.01x |
| nans_to_nulls n=65536 cpu | 0.03 ms | 0.40 ms | 0.08x |
| nans_to_nulls n=65536 gpu | 0.03 ms | 0.47 ms | 0.07x |
| nans_to_nulls n=1048576 cpu | 0.71 ms | 7.06 ms | 0.10x |
| nans_to_nulls n=1048576 gpu | 0.71 ms | 6.47 ms | 0.11x |
| nans_to_nulls n=16777216 cpu | 25.28 ms | 227.18 ms | 0.11x |
| nans_to_nulls n=16777216 gpu | 25.28 ms | 165.62 ms | 0.15x |

Read this honestly: **Mojo loses on almost everything except the popcount.**

* `count_set_bits` wins by an order of magnitude because the reference
  (`np.unpackbits(...).sum()`) has to materialise an 8x larger array before
  summing it, while the kernel is a straight byte popcount. That is a real
  structural difference, not a tuning win.
* `ewm` is the one arithmetic case that wins: it is a serial recurrence with a
  loop-carried dependency, which NumPy can only express through
  `scipy.signal.lfilter`, and the kernel does it in one pass with no temporaries.
* `cumsum`, `sum`, `rolling` and `bools_to_mask` lose. These are all
  bandwidth-bound loops over contiguous data, and NumPy's own implementations
  are vectorised while a scalar `for` loop in a `abi("C")` export is not. The
  kernels carry no per-element branch -- the aggregation kind is dispatched
  once, outside the loop, which is worth about 1.5x over the first draft -- but
  that is not enough to reach NumPy's SIMD. The honest conclusion is that for
  this workload NumPy is the better tool and the port's value is the semantics
  (cudf's null rules, its two-moment variance, its pair-scan EWMA), not the
  speed.
* The GPU rows lose everywhere, and the size sweep says why: the host-to-device
  copy of the input plus the copy back dominates until the input is large
  enough to amortise it, and even at 16M rows it does not. The GPU path is here
  because the brief requires it and because the three kernels it covers are
  genuinely device-shaped; it is not a speedup, and reporting it as one would be
  false.

Reproduce with `pixi run bench`.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-cudf-kernels.so`.

The Python layer owns every array. Buffers cross the C ABI as 64-bit addresses
and are rebuilt in Mojo as `Pointer[T, AnyOrigin[mut=True]]`, which keeps the
exported symbols non-parametric. A Mojo pointer is non-nullable, so a column
with no nulls is passed a shared all-ones validity buffer rather than cudf's
null mask pointer.

Three kernels have GPU variants: `nans_to_nulls`, `bools_to_mask` and
`count_set_bits`. Each is guarded -- no device, under 4000 MiB free, or an
allocation over 2 GiB returns 0 and the shim warns and runs the CPU kernel. They
are the only three that qualify, because each has no cross-thread dependence: a
thread can produce its own output element with no communication.

The prefix scan and both rolling windows deliberately have **no** GPU variant.
All three are sequential recurrences. A "parallel" scan here would give every
thread the whole array and cost `O(n)` work per thread, which is slower than
the serial loop it replaces. That would be a fake port, so it is not shipped.
The test suite asserts those three symbols are absent.

Mojo emits FMA and the two-moment variance formula cancels, so floating-point
results are compared with `assert_allclose` at a stated tolerance. The bitmask
kernels are bitwise and are compared with `==`.

## Tests

`bash build/build.sh && PYTHONPATH=python pytest tests -q` -> **83 passed**.

`cudf` cannot be imported here, so parity is against `tests/reference.py`, which
re-derives each algorithm a different way rather than transcribing the kernel:
scans use `np.cumsum` and `np.minimum.accumulate`, the adjusted EWMA is rebuilt
from a weight vector via `np.cumprod`, rolling windows use
`np.searchsorted`, and bitmask packing uses `np.packbits`. The tests then target
the bugs a transcription actually makes: a swapped min/max dispatch, the missing
`+1` on the exclusive scan's null limit, a null carried to the end of the
column instead of the last valid value, `mean` dividing by the window width
instead of the valid count, a reversed bit order, and a mask sized `n // 8`
instead of `(n + 7) // 8`. Both mutations were checked by hand against the
suite: dropping the exclusive `+1` fails 1 test, reversing `set_valid`'s bit
order fails 19.

## License

Apache-2.0, matching RAPIDS cudf.
