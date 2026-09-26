"""Independent NumPy references for the libcudf kernels under test.

Every function here is written to be a *different* computation from the Mojo
one wherever that is possible, so a transcription slip in either direction
shows up as a disagreement:

* scans use `np.cumsum` / `np.minimum.accumulate` rather than a hand loop;
* the unadjusted EWMA uses the closed-form geometric weights, a convolution,
  where the kernel uses the scalar recurrence;
* the adjusted EWMA divides a weight vector by its own sum rather than
  accumulating numerator and denominator separately;
* rolling windows use `sliding_window_view`;
* bitmask packing uses `np.packbits`.
"""

from __future__ import annotations

import math

import numpy as np

SUM, PRODUCT, MIN, MAX = 0, 1, 2, 3

_OP = {SUM: "sum", PRODUCT: "prod", MIN: "min", MAX: "max"}


def valid_array(data, valid):
    values = np.asarray(data, dtype=np.float64)
    if valid is None:
        return values, np.ones(values.size, dtype=bool)
    return values, np.asarray(valid, dtype=bool)


def scan_reference(data, valid, op, inclusive=True):
    """Reference prefix scan, including cudf's null-propagating mask."""
    values, ok = valid_array(data, valid)
    n = values.size
    if inclusive:
        kernel = {SUM: np.cumsum, PRODUCT: np.cumprod}.get(op)
        if kernel is not None:
            out = kernel(values[ok])
        else:
            out = np.minimum.accumulate(values[ok]) if op == MIN else np.maximum.accumulate(values[ok])
    else:
        kept = values[ok]
        identity = {SUM: 0.0, PRODUCT: 1.0, MIN: math.inf, MAX: -math.inf}[op]
        prefix = {SUM: np.cumsum, PRODUCT: np.cumprod}.get(op)
        if prefix is not None:
            out = np.concatenate([[identity], prefix(kept)[:-1]]) if kept.size else np.empty(0)
        else:
            run = identity
            acc = []
            for v in kept:
                acc.append(run)
                run = min(run, v) if op == MIN else max(run, v)
            out = np.array(acc, dtype=np.float64)
    result = np.zeros(n, dtype=np.float64)
    result[ok] = out
    first_null = n if ok.all() else int(np.argmin(ok))
    limit = first_null if inclusive else min(n, first_null + 1)
    mask = np.arange(n) < limit
    return result, mask


def reduce_reference(data, valid, op, ddof=1):
    """Reference column reduction, including libcudf's two-moment variance."""
    values, ok = valid_array(data, valid)
    kept = values[ok]
    if kept.size == 0:
        return None, 0
    if op == SUM:
        return float(kept.sum()), kept.size
    if op == PRODUCT:
        return float(np.prod(kept)), kept.size
    if op == MIN:
        return float(kept.min()), kept.size
    if op == MAX:
        return float(kept.max()), kept.size
    if op == 4:  # mean
        return float(kept.mean()), kept.size
    if op in (5, 6):  # var, std
        variance = (float((kept**2).sum()) - float(kept.sum()) ** 2 / kept.size) / (
            kept.size - ddof
        )
        return (variance, math.sqrt(variance) if op == 6 else variance)[0], kept.size
    raise ValueError(op)


def ewm_reference(data, valid, beta, adjust):
    """Reference EWMA.

    The unadjusted branch is the geometric-weight recurrence
    `y_i = beta*y_(i-1) + (1-beta)*x_i` with `y_0 = x_0`, while the adjusted
    branch is rebuilt here as a weight vector: writing the pair scan out,
    `N_k = s_k*N_(k-1) + x` and `D_k = s_k*D_(k-1) + 1` expand to
    `y_k = (w . x_p) / sum(w)` with `w_j = prod_{t=j+1..k} s_t`. Computing `w`
    by `np.cumprod` is a different route to the same number than the kernel's
    running accumulation, so a wrong beta, a wrong seeding value or an
    off-by-one in the recurrence cannot agree by accident.
    """
    values, ok = valid_array(data, valid)
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    positions = np.flatnonzero(ok)
    if positions.size == 0:
        return out

    if not adjust:
        running = 0.0
        started = False
        nulls = 0
        for i in range(n):
            if not ok[i]:
                out[i] = running
                nulls += 1
                continue
            if not started:
                running = values[i]
                started = True
            elif nulls == 0:
                running = beta * running + (1.0 - beta) * values[i]
            else:
                factor = (1.0 - beta) + beta ** (nulls + 1)
                running = (
                    beta * beta**nulls / factor * running
                    + (1.0 - beta) * values[i] / factor
                )
            nulls = 0
            out[i] = running
        return out

    # `s_k` is beta after a gap-free step and beta**(gap + 1) otherwise.
    # The first valid entry is a fresh seed, so only gaps *after* it scale the
    # pair; leading nulls have no previous value to down-weight.
    scales = np.full(positions.size, beta, dtype=np.float64)
    for k in range(1, positions.size):
        gap = positions[k] - positions[k - 1] - 1
        scales[k] = beta if gap == 0 else beta ** (gap + 1)
    last = np.nan
    computed = np.zeros(positions.size, dtype=np.float64)
    for k in range(positions.size):
        # w = [s_k..s_1, s_k..s_2, ..., s_k, 1]: the newest observation keeps
        # the full weight and each older one is discounted once per step.
        weights = (
            np.concatenate([np.cumprod(scales[k:0:-1])[::-1], [1.0]])
            if k
            else np.array([1.0])
        )
        segment = values[positions[: k + 1]]
        computed[k] = float(weights @ segment) / float(weights.sum())
    # A null position is the identity of the recurrence, so it carries the most
    # recent value forward rather than jumping to the final one.
    seen = -1
    for i in range(n):
        if not ok[i]:
            out[i] = computed[seen] if seen >= 0 else np.nan
        else:
            seen += 1
            out[i] = computed[seen]
    return out


def rolling_window_reference(data, valid, preceding, following, min_periods, op, ddof=1):
    """Reference fixed-index rolling aggregation via sliding windows."""
    values, ok = valid_array(data, valid)
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - preceding), min(n - 1, i + following)
        window = values[lo : hi + 1]
        window_ok = ok[lo : hi + 1]
        kept = window[window_ok]
        if kept.size == 0 or kept.size < min_periods:
            continue
        out[i] = _agg(kept, op, ddof)
        mask[i] = True
    return out, mask


def rolling_range_reference(
    data, valid, orderby, preceding, following, preceding_open, following_open,
    min_periods, op, ddof=1,
):
    """Reference value-range rolling aggregation, using `np.searchsorted`."""
    values, ok = valid_array(data, valid)
    order = np.asarray(orderby, dtype=np.float64)
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    side = "right" if preceding_open else "left"
    fside = "left" if following_open else "right"
    for i in range(n):
        lo = int(np.searchsorted(order, order[i] - preceding, side=side))
        hi = int(np.searchsorted(order, order[i] + following, side=fside))
        kept = values[lo:hi][ok[lo:hi]]
        if kept.size == 0 or kept.size < min_periods:
            continue
        out[i] = _agg(kept, op, ddof)
        mask[i] = True
    return out, mask


def _agg(kept: np.ndarray, op: int, ddof: int) -> float:
    if op == SUM:
        return float(kept.sum())
    if op == PRODUCT:
        return float(kept.prod())
    if op == MIN:
        return float(kept.min())
    if op == MAX:
        return float(kept.max())
    if op == 4:
        return float(kept.sum() / kept.size)
    if op in (5, 6):
        variance = (float((kept**2).sum()) - float(kept.sum()) ** 2 / kept.size) / (
            kept.size - ddof
        )
        return variance if op == 5 else math.sqrt(variance)
    raise ValueError(op)
