"""Column kernels ported from libcudf's compute layer.

This is the arithmetic core of RAPIDS cudf, taken from the `cudf::reduction`,
`cudf::detail::rolling` and `cudf::transform` translation units:

* prefix scan  -- `cpp/src/reductions/scan/`
* ewm          -- `cpp/src/reductions/scan/ewm.cu`
* reductions   -- `cpp/src/reductions/{sum,product,min,max,mean,var,std}.cu`
* rolling      -- `cpp/src/rolling/detail/` (fixed window and range window)
* bitmask      -- `cpp/src/transform/{bools_to_mask,mask_to_bools,nans_to_nulls}.cu`
                  and `cudf/detail/null_mask.hpp`

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric. A null-mask
address of 0 means "no nulls", which is exactly how cudf represents a column
whose validity buffer is empty.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.bit import pop_count
from std.math import exp, inf, nan, pow, sqrt

comptime DPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime NPtr = Pointer[Int64, AnyOrigin[mut=True]]

# Aggregation kinds for the subset libcudf implements for these kernels.
comptime OP_SUM = 0
comptime OP_PRODUCT = 1
comptime OP_MIN = 2
comptime OP_MAX = 3
comptime OP_MEAN = 4
comptime OP_VAR = 5
comptime OP_STD = 6

# GPU admission thresholds, as the project brief requires.
comptime GPU_MIN_FREE_MIB = 4000
comptime GPU_MAX_ALLOC = 2 * 1024 * 1024 * 1024
comptime NOT_RUN = 0
comptime RAN = 1
comptime UNAVAILABLE = -1


def fp(addr: Int) -> DPtr:
    return DPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def ip(addr: Int) -> NPtr:
    return NPtr(unsafe_from_address=addr)


@always_inline
def valid_at(mask: BPtr, i: Int) -> Bool:
    """cudf bitmask bit test: bit `i` lives in byte `i >> 3` at offset `i & 7`."""
    return (mask[unsafe_offset=i >> 3] & (UInt8(1) << UInt8(i & 7))) != UInt8(0)


@always_inline
def set_valid(mask: BPtr, i: Int, v: Bool):
    var byte = i >> 3
    var bit = UInt8(1) << UInt8(i & 7)
    if v:
        mask[unsafe_offset=byte] = mask[unsafe_offset=byte] | bit
    else:
        mask[unsafe_offset=byte] = mask[unsafe_offset=byte] & ~bit


@always_inline
def acc_neutral(op: Int) -> Float64:
    """The identity of a reduction, used to seed an exclusive scan."""
    if op == OP_PRODUCT:
        return 1.0
    if op == OP_MIN:
        return inf[DType.float64]()
    if op == OP_MAX:
        return -inf[DType.float64]()
    return 0.0


@export("ck_scan")
def ck_scan(
    in_addr: Int,
    mask_addr: Int,
    n: Int,
    op: Int,
    inclusive: Int,
    dst_addr: Int,
    dst_mask_addr: Int,
) abi("C"):
    """Prefix scan with cudf's null-propagating validity mask.

    cudf scans the valid entries only, then writes the output mask as
    `min(n, first_null + (exclusive ? 1 : 0))` valid bits followed by invalid
    bits, where `first_null` is the index of the first null. That is the rule
    in `mask_scan` in `scan/scan_inclusive.cu`, and it reproduces
    `Series.cumsum`/`cummin`/... including the rule that everything at and after
    the first null is NA.

    An inclusive scan emits the running value after folding in `x[i]`; an
    exclusive one emits it before, so the first row is the operation's identity
    and row `i` is the fold of `x[0..i-1]`. That is the difference
    `thrust::inclusive_scan` and `thrust::exclusive_scan` make.

    Each aggregation gets its own loop: the kind is loop-invariant, and a
    per-element branch on it costs a noticeable share of a multi-million-element
    scan. None of the four needs a "first element" case, because 0, 1 and
    +-inf are the identities of sum, product, min and max respectively.
    """
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)
    var dst_mask = bp(dst_mask_addr)

    var first_null = n
    for i in range(n):
        if not valid_at(mask, i):
            first_null = i
            break
    var limit = first_null
    if inclusive == 0 and limit < n:
        limit = limit + 1

    var acc = acc_neutral(op)
    if op == OP_SUM:
        for i in range(n):
            var before = acc
            if valid_at(mask, i):
                acc = acc + src[unsafe_offset=i]
            dst[unsafe_offset=i] = before if inclusive == 0 else acc
    elif op == OP_PRODUCT:
        for i in range(n):
            var ok = valid_at(mask, i)
            var before = acc
            if ok:
                acc = acc * src[unsafe_offset=i]
            dst[unsafe_offset=i] = before if inclusive == 0 else acc
    elif op == OP_MIN:
        for i in range(n):
            var before = acc
            if valid_at(mask, i):
                var v = src[unsafe_offset=i]
                if v < acc:
                    acc = v
            dst[unsafe_offset=i] = before if inclusive == 0 else acc
    else:
        for i in range(n):
            var ok = valid_at(mask, i)
            var before = acc
            if ok:
                var v = src[unsafe_offset=i]
                if v > acc:
                    acc = v
            dst[unsafe_offset=i] = before if inclusive == 0 else acc

    for i in range(n):
        set_valid(dst_mask, i, i < limit)


@export("ck_reduce")
def ck_reduce(
    in_addr: Int, mask_addr: Int, n: Int, op: Int, ddof: Int, dst_addr: Int
) abi("C") -> Int:
    """Whole-column reduction. `dst[0]` is the value, `dst[1]` the valid count.

    Variance uses libcudf's two-moment formula from `reductions/compound.cuh`
    (sum, sum of squares, count) rather than a two-pass Welford, because that is
    what `cudf::reduction::detail::variance` dispatches to; matching it means
    matching its cancellation behaviour too.

    The aggregation kind is dispatched once, outside the loop, so the inner loop
    carries no branch. Folding a per-element `if op == ...` into a 16M-element
    loop costs about a third of the runtime.
    """
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)

    var acc = acc_neutral(op)
    var total = Float64(0.0)
    var sq_total = Float64(0.0)
    var count = 0
    if op == OP_PRODUCT:
        for i in range(n):
            if valid_at(mask, i):
                acc = acc * src[unsafe_offset=i]
                count += 1
    elif op == OP_MIN:
        for i in range(n):
            if valid_at(mask, i):
                var v = src[unsafe_offset=i]
                if v < acc:
                    acc = v
                count += 1
    elif op == OP_MAX:
        for i in range(n):
            if valid_at(mask, i):
                var v = src[unsafe_offset=i]
                if v > acc:
                    acc = v
                count += 1
    else:
        # SUM, MEAN, VAR and STD all need the plain sum and sum of squares.
        for i in range(n):
            if valid_at(mask, i):
                var v = src[unsafe_offset=i]
                total += v
                sq_total += v * v
                count += 1
        acc = total

    if count == 0:
        dst[unsafe_offset=0] = Float64(0.0)
        dst[unsafe_offset=1] = Float64(0.0)
        return 0
    if op == OP_MEAN:
        dst[unsafe_offset=0] = total / Float64(count)
    elif op == OP_VAR or op == OP_STD:
        var variance = (sq_total - total * total / Float64(count)) / Float64(
            count - ddof
        )
        dst[unsafe_offset=0] = variance if op == OP_VAR else sqrt(variance)
    else:
        dst[unsafe_offset=0] = acc
    dst[unsafe_offset=1] = Float64(count)
    return count


@export("ck_ewm")
def ck_ewm(
    in_addr: Int, mask_addr: Int, n: Int, beta: Float64, dst_addr: Int
) abi("C"):
    """The `adjust=False` branch of `scan/ewm.cu`.

    cudf encodes the recurrence as a scan over pairs `(p, q)` combined by
    `(c1, c2) * (d1, d2) = (c1 * d1, c2 * d1 + d2)`, and the unadjusted branch
    divides by nothing -- its comment says the denominators are all one. So `p`
    only ever multiplies the running value and the pair collapses to the scalar
    recurrence this kernel evaluates. `beta = com / (com + 1)`, and the first
    element is seeded with `x[0]` rather than zero, which is the `index == 0`
    special case in `ewma_noadjust_nulls_functor`.

    Nulls follow `null_roll_up`: a null position is the identity `(1, 0)` and so
    carries the previous value forward, while the next valid position divides
    by `(1 - beta) + beta**(nulls + 1)` and scales the carried value by
    `beta**nulls`, which down-weights it once per missing observation.
    """
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)

    var acc = Float64(0.0)
    var started = False
    var nulls_since = 0
    for i in range(n):
        if not valid_at(mask, i):
            dst[unsafe_offset=i] = acc
            nulls_since += 1
            continue
        var v = src[unsafe_offset=i]
        if not started:
            acc = v
            started = True
        elif nulls_since == 0:
            acc = beta * acc + (1.0 - beta) * v
        else:
            var factor = (1.0 - beta) + pow(beta, Float64(nulls_since) + 1.0)
            acc = (
                beta * pow(beta, Float64(nulls_since)) / factor * acc
                + (1.0 - beta) * v / factor
            )
        nulls_since = 0
        dst[unsafe_offset=i] = acc


@export("ck_ewm_adjusted")
def ck_ewm_adjusted(
    in_addr: Int, mask_addr: Int, n: Int, beta: Float64, dst_addr: Int
) abi("C"):
    """The `adjust=True` (INFINITE history) branch of `scan/ewm.cu`.

    Runs the same pair scan twice, once carrying the observations and once
    carrying unit inputs, then divides. A null position is the identity, so the
    value carries forward; the next valid position scales the accumulated pair
    by `beta * beta**nulls` before adding its own term. Before the first valid
    entry the accumulator is `(0, 0)` and cudf's division yields NaN, which is
    what a leading run of nulls produces there.
    """
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)

    var num = Float64(0.0)
    var den = Float64(0.0)
    var seen_valid = False
    var nulls_since = 0
    for i in range(n):
        if not valid_at(mask, i):
            dst[unsafe_offset=i] = (
                num / den if seen_valid else nan[DType.float64]()
            )
            nulls_since += 1
            continue
        if not seen_valid:
            # The scan starts from the identity (0, 0), so the first element's
            # weight is never applied: it seeds the pair outright.
            num = src[unsafe_offset=i]
            den = 1.0
        else:
            var scale = beta
            if nulls_since > 0:
                scale = beta * pow(beta, Float64(nulls_since))
            num = scale * num + src[unsafe_offset=i]
            den = scale * den + 1.0
        seen_valid = True
        nulls_since = 0
        dst[unsafe_offset=i] = num / den


@export("ck_rolling_window")
def ck_rolling_window(
    in_addr: Int,
    mask_addr: Int,
    n: Int,
    preceding: Int,
    following: Int,
    min_periods: Int,
    ddof: Int,
    op: Int,
    dst_addr: Int,
    dst_mask_addr: Int,
) abi("C"):
    """Fixed-index rolling window, as in `rolling/detail/rolling_fixed_window.cu`.

    Row `i` covers `[i - preceding, i + following]` clipped to the column, null
    entries are skipped, and the result is null when the window holds no valid
    entry or fewer than `min_periods` of them.
    """
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)
    var dst_mask = bp(dst_mask_addr)

    for i in range(n):
        var lo = i - preceding
        if lo < 0:
            lo = 0
        var hi = i + following
        if hi > n - 1:
            hi = n - 1
        var acc = acc_neutral(op)
        var total = Float64(0.0)
        var sq_total = Float64(0.0)
        var count = 0
        if op == OP_MIN:
            for j in range(lo, hi + 1):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    if v < acc:
                        acc = v
                    count += 1
        elif op == OP_MAX:
            for j in range(lo, hi + 1):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    if v > acc:
                        acc = v
                    count += 1
        elif op == OP_PRODUCT:
            for j in range(lo, hi + 1):
                if valid_at(mask, j):
                    acc = acc * src[unsafe_offset=j]
                    count += 1
        else:
            for j in range(lo, hi + 1):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    total += v
                    sq_total += v * v
                    count += 1
            acc = total
        var keep = count > 0 and count >= min_periods
        if not keep:
            dst[unsafe_offset=i] = Float64(0.0)
        elif op == OP_MEAN:
            dst[unsafe_offset=i] = total / Float64(count)
        elif op == OP_VAR or op == OP_STD:
            var variance = (sq_total - total * total / Float64(count)) / Float64(
                count - ddof
            )
            dst[unsafe_offset=i] = variance if op == OP_VAR else sqrt(variance)
        else:
            dst[unsafe_offset=i] = acc
        set_valid(dst_mask, i, keep)


def lower_bound(ordr: DPtr, probe: Float64, count: Int) -> Int:
    """First index whose orderby value is `>= probe` (Python's `bisect_left`)."""
    var low = 0
    var high = count
    while low < high:
        var mid = (low + high) >> 1
        if ordr[unsafe_offset=mid] < probe:
            low = mid + 1
        else:
            high = mid
    return low


def upper_bound(ordr: DPtr, probe: Float64, count: Int) -> Int:
    """First index whose orderby value is `> probe` (Python's `bisect_right`)."""
    var low = 0
    var high = count
    while low < high:
        var mid = (low + high) >> 1
        if probe < ordr[unsafe_offset=mid]:
            high = mid
        else:
            low = mid + 1
    return low


@export("ck_rolling_range")
def ck_rolling_range(
    ord_addr: Int,
    in_addr: Int,
    mask_addr: Int,
    n: Int,
    prec_delta: Float64,
    foll_delta: Float64,
    prec_open: Int,
    foll_open: Int,
    min_periods: Int,
    ddof: Int,
    op: Int,
    dst_addr: Int,
    dst_mask_addr: Int,
) abi("C"):
    """Value-range rolling window, as in `rolling/detail/range_utils.cuh`.

    Row `i` covers the rows whose `orderby` value lies in
    `(orderby[i] - prec_delta, orderby[i] + foll_delta]`, located by bisecting
    the already-sorted `orderby` column. That is the bound rule libcudf's own
    `test_rolling.py` asserts, with `BoundedOpen` selecting the open side.
    """
    var ordr = fp(ord_addr)
    var src = fp(in_addr)
    var dst = fp(dst_addr)
    var mask = bp(mask_addr)
    var dst_mask = bp(dst_mask_addr)

    for i in range(n):
        var key = ordr[unsafe_offset=i]
        var start: Int
        var stop: Int
        if prec_open != 0:
            start = upper_bound(ordr, key - prec_delta, n)
        else:
            start = lower_bound(ordr, key - prec_delta, n)
        if foll_open != 0:
            stop = lower_bound(ordr, key + foll_delta, n)
        else:
            stop = upper_bound(ordr, key + foll_delta, n)

        var acc = acc_neutral(op)
        var total = Float64(0.0)
        var sq_total = Float64(0.0)
        var count = 0
        if op == OP_MIN:
            for j in range(start, stop):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    if v < acc:
                        acc = v
                    count += 1
        elif op == OP_MAX:
            for j in range(start, stop):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    if v > acc:
                        acc = v
                    count += 1
        elif op == OP_PRODUCT:
            for j in range(start, stop):
                if valid_at(mask, j):
                    acc = acc * src[unsafe_offset=j]
                    count += 1
        else:
            for j in range(start, stop):
                if valid_at(mask, j):
                    var v = src[unsafe_offset=j]
                    total += v
                    sq_total += v * v
                    count += 1
            acc = total
        var keep = count > 0 and count >= min_periods
        if not keep:
            dst[unsafe_offset=i] = Float64(0.0)
        elif op == OP_MEAN:
            dst[unsafe_offset=i] = total / Float64(count)
        elif op == OP_VAR or op == OP_STD:
            var variance = (sq_total - total * total / Float64(count)) / Float64(
                count - ddof
            )
            dst[unsafe_offset=i] = variance if op == OP_VAR else sqrt(variance)
        else:
            dst[unsafe_offset=i] = acc
        set_valid(dst_mask, i, keep)


# ---------------------------------------------------------------------------
# validity bitmask
# ---------------------------------------------------------------------------


@export("ck_bools_to_mask")
def ck_bools_to_mask(bools_addr: Int, n: Int, dst_addr: Int) abi("C") -> Int:
    """`cpp/src/transform/bools_to_mask.cu`: pack a validity byte array.

    One output byte is built from eight input bytes, so the inner loop never
    touches the packed buffer. Doing it the other way round -- a read-modify-
    write per bit -- is 50x slower at 16M rows, which is what the benchmark
    measures.

    Returns the number of set bits, which is the complement of the `null_count`
    cudf reports for the packed column.
    """
    var bools = bp(bools_addr)
    var dst = bp(dst_addr)
    var words = (n + 7) >> 3
    var count = 0
    for w in range(words):
        var lo = w * 8
        var hi = lo + 8
        if hi > n:
            hi = n
        var bits = UInt8(0)
        for k in range(lo, hi):
            if bools[unsafe_offset=k] != UInt8(0):
                bits = bits | (UInt8(1) << UInt8(k - lo))
        dst[unsafe_offset=w] = bits
        count += Int(pop_count(bits))
    return count


@export("ck_mask_to_bools")
def ck_mask_to_bools(
    mask_addr: Int, bools_addr: Int, n: Int, mask_bytes: Int
) abi("C") -> Int:
    """`cpp/src/transform/mask_to_bools.cu`: unpack a validity bitmask.

    Entries whose bit lives past the end of the mask are invalid, matching
    cudf's rule that a null mask shorter than the column is padded with nulls.
    `mask_bytes` is the real length of the buffer, so a caller can pass a short
    mask and get the padding without reading out of bounds.
    """
    var mask = bp(mask_addr)
    var bools = bp(bools_addr)
    var count = 0
    for i in range(n):
        var ok = (i >> 3) < mask_bytes and valid_at(mask, i)
        bools[unsafe_offset=i] = UInt8(1 if ok else 0)
        if ok:
            count += 1
    return count


@export("ck_count_set_bits")
def ck_count_set_bits(mask_addr: Int, valid_count: Int) abi("C") -> Int:
    """`cudf::detail::count_set_bits`: popcount over the whole validity mask."""
    var mask = bp(mask_addr)
    var words = (valid_count + 7) >> 3
    var total = 0
    for w in range(words):
        total += Int(pop_count(mask[unsafe_offset=w]))
    return total


@export("ck_set_null_mask")
def ck_set_null_mask(mask_addr: Int, begin: Int, end: Int, value: Int) abi("C"):
    """`cudf::detail::set_null_mask`: fill `[begin, end)` with 0 or 1."""
    var mask = bp(mask_addr)
    for i in range(begin, end):
        set_valid(mask, i, value != 0)


@export("ck_nans_to_nulls")
def ck_nans_to_nulls(in_addr: Int, mask_addr: Int, n: Int) abi("C") -> Int:
    """`cpp/src/transform/nans_to_nulls.cu`: mark NaN entries invalid.

    The output mask starts all-valid and every NaN clears its bit. Infinities
    are not nulls, which is why the test is `v != v` and not a finiteness check.
    """
    var src = fp(in_addr)
    var mask = bp(mask_addr)
    var words = (n + 7) >> 3
    for w in range(words):
        mask[unsafe_offset=w] = UInt8(0xFF)
    var found = 0
    for i in range(n):
        var v = src[unsafe_offset=i]
        if v != v:
            set_valid(mask, i, False)
            found += 1
    return found


# ---------------------------------------------------------------------------
# GPU paths.
#
# Every entry point is guarded: no device, under 4000 MiB free, or an allocation
# over 2 GiB returns 0 and the Python shim runs the CPU kernel instead. Only the
# three kernels here are genuinely GPU-shaped -- each has no cross-thread
# dependence, so a grid of threads can produce its own output element with no
# communication. The prefix scan and the rolling windows are deliberately NOT
# in this list: both are sequential recurrences, and faking a parallel version
# would be O(n) work per thread, which is slower than the serial kernel it
# would replace.
# ---------------------------------------------------------------------------


def nans_gpu_kernel(src: DPtr, mask: BPtr, n: Int64):
    """One thread per mask byte.

    A thread per element would have eight threads read-modify-write the same
    byte, and those are not atomic, so the writes race. Giving each thread
    exclusive ownership of one byte removes the race entirely.
    """
    var w = Int(global_idx.x)
    var words = (Int(n) + 7) >> 3
    if w >= words:
        return
    var lo = w * 8
    var hi = lo + 8
    if hi > Int(n):
        hi = Int(n)
    var bits = mask[unsafe_offset=w]
    for k in range(lo, hi):
        var v = src[unsafe_offset=k]
        if v != v:
            bits = bits & ~(UInt8(1) << UInt8(k - lo))
    mask[unsafe_offset=w] = bits


def fill_ones_gpu_kernel(mask: BPtr, words: Int64):
    """The device buffer from `enqueue_create_buffer` is uninitialised, so the
    all-valid start state has to be written before any bit is cleared."""
    var w = Int(global_idx.x)
    if w < Int(words):
        mask[unsafe_offset=w] = UInt8(0xFF)


def pack_gpu_kernel(bools: BPtr, dst: BPtr, n: Int64):
    var w = Int(global_idx.x)
    var words = (Int(n) + 7) >> 3
    if w >= words:
        return
    var lo = w * 8
    var hi = lo + 8
    if hi > Int(n):
        hi = Int(n)
    var bits = UInt8(0)
    for k in range(lo, hi):
        if bools[unsafe_offset=k] != UInt8(0):
            bits = bits | (UInt8(1) << UInt8(k - lo))
    dst[unsafe_offset=w] = bits


def popcount_gpu_kernel(mask: BPtr, partials: NPtr, words: Int64, nthreads: Int32):
    var t = Int(global_idx.x)
    var threads = Int(nthreads)
    if t >= threads:
        return
    var per = (Int(words) + threads - 1) // threads
    var lo = t * per
    var hi = lo + per
    if hi > Int(words):
        hi = Int(words)
    var total = 0
    for w in range(lo, hi):
        total += Int(pop_count(mask[unsafe_offset=w]))
    partials[unsafe_offset=t] = Int64(total)


def sum_gpu_kernel(partials: NPtr, total: NPtr, count: Int64):
    if Int(global_idx.x) == 0:
        var acc = Int64(0)
        for k in range(Int(count)):
            acc += partials[unsafe_offset=k]
        total[unsafe_offset=0] = acc


@export("ck_nans_to_nulls_gpu")
def ck_nans_to_nulls_gpu(in_addr: Int, mask_addr: Int, n: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var memory = ctx.get_memory_info()
        if memory[0] < UInt(GPU_MIN_FREE_MIB * 1024 * 1024):
            return NOT_RUN
        var words = (n + 7) >> 3
        if UInt(n) * UInt(16) + UInt(words) > UInt(GPU_MAX_ALLOC):
            return NOT_RUN
        var d_src = ctx.enqueue_create_buffer[DType.float64](n)
        var d_mask = ctx.enqueue_create_buffer[DType.uint8](words)
        comptime block_size = 256
        var grid_size = (words + block_size - 1) // block_size
        ctx.enqueue_function[fill_ones_gpu_kernel](
            d_mask, Int64(words), grid_dim=grid_size, block_dim=block_size
        )
        ctx.synchronize()
        ctx.enqueue_copy(d_src, fp(in_addr))
        ctx.enqueue_function[nans_gpu_kernel](
            d_src, d_mask, Int64(n), grid_dim=grid_size, block_dim=block_size
        )
        ctx.enqueue_copy(bp(mask_addr), d_mask)
        ctx.synchronize()
        return RAN
    except:
        return NOT_RUN


@export("ck_bools_to_mask_gpu")
def ck_bools_to_mask_gpu(bools_addr: Int, n: Int, dst_addr: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var memory = ctx.get_memory_info()
        if memory[0] < UInt(GPU_MIN_FREE_MIB * 1024 * 1024):
            return NOT_RUN
        var words = (n + 7) >> 3
        if UInt(n) * UInt(2) + UInt(words) * UInt(2) > UInt(GPU_MAX_ALLOC):
            return NOT_RUN
        var d_bools = ctx.enqueue_create_buffer[DType.uint8](n)
        var d_mask = ctx.enqueue_create_buffer[DType.uint8](words)
        ctx.enqueue_copy(d_bools, bp(bools_addr))
        comptime block_size = 256
        var grid_size = (words + block_size - 1) // block_size
        ctx.enqueue_function[pack_gpu_kernel](
            d_bools, d_mask, Int64(n), grid_dim=grid_size, block_dim=block_size
        )
        ctx.enqueue_copy(bp(dst_addr), d_mask)
        ctx.synchronize()
        return RAN
    except:
        return NOT_RUN


@export("ck_count_set_bits_gpu")
def ck_count_set_bits_gpu(
    mask_addr: Int, valid_count: Int, dst_addr: Int
) abi("C") -> Int:
    """Two-stage popcount; the total comes back through `dst_addr` as an int64.

    One thread per partial word, then a single thread sums the partials. The
    count is returned through device memory rather than the C return value
    because Mojo has no way to read a device scalar back into a host `Int`
    inside an `abi("C")` export.
    """
    try:
        var ctx = DeviceContext()
        var memory = ctx.get_memory_info()
        if memory[0] < UInt(GPU_MIN_FREE_MIB * 1024 * 1024):
            return NOT_RUN
        var words = (valid_count + 7) >> 3
        if words == 0:
            return NOT_RUN
        var d_mask = ctx.enqueue_create_buffer[DType.uint8](words)
        ctx.enqueue_copy(d_mask, bp(mask_addr))
        comptime nthreads = 1024
        var d_partials = ctx.enqueue_create_buffer[DType.int64](nthreads)
        comptime block_size = 256
        ctx.enqueue_function[popcount_gpu_kernel](
            d_mask, d_partials, Int64(words), Int32(nthreads),
            grid_dim=nthreads, block_dim=block_size,
        )
        var d_total = ctx.enqueue_create_buffer[DType.int64](1)
        ctx.enqueue_function[sum_gpu_kernel](
            d_partials, d_total, Int64(nthreads), grid_dim=1, block_dim=1
        )
        ctx.enqueue_copy(ip(dst_addr), d_total)
        ctx.synchronize()
        return RAN
    except:
        return NOT_RUN
