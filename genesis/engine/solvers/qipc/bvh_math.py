"""BVH math primitives: AABB operations, Morton codes, Karras 2012 helpers.

All functions are ``@qd.func`` for use inside graph kernels.  Faithfully ports
``bvh_types.cuh`` from cgq so numerical results are identical.
"""

from __future__ import annotations

import quadrants as qd

# ---------------------------------------------------------------------------
# AABB helpers  (stored as 6 contiguous f64: lower_x, lower_y, lower_z,
#                                             upper_x, upper_y, upper_z)
# We pass AABB data via the owning field/ndarray + a flat offset.
# ---------------------------------------------------------------------------


@qd.func
def aabb_init(
    aabbs: qd.template(),
    idx: qd.i32,
):
    """Reset aabbs[idx] to the empty sentinel (lower=+inf, upper=-inf).

    The +-1e32 bounds are sentinels for "no bound yet" that the min/max
    expansion reduces away, not a length scale -- nothing in the scene compares
    against them, so they carry no tunable meaning. They are spelled inline
    because a module constant read from device code never enters the fastcache
    key, so editing it would silently reuse a stale kernel.
    """
    for k in qd.static(range(3)):
        aabbs[idx, k] = 1e32
        aabbs[idx, 3 + k] = -1e32


@qd.func
def aabb_expand(
    aabbs: qd.template(),
    idx: qd.i32,
    r: qd.f64,
):
    """Expand aabbs[idx] by radius r on all sides. Matches cgq ``AABB::expand``."""
    for k in qd.static(range(3)):
        aabbs[idx, k] = aabbs[idx, k] - r
        aabbs[idx, 3 + k] = aabbs[idx, 3 + k] + r


@qd.func
def aabb_combine_point(
    aabbs: qd.template(),
    idx: qd.i32,
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
):
    """Expand aabbs[idx] to include point (px, py, pz)."""
    aabbs[idx, 0] = qd.min(aabbs[idx, 0], px)
    aabbs[idx, 1] = qd.min(aabbs[idx, 1], py)
    aabbs[idx, 2] = qd.min(aabbs[idx, 2], pz)
    aabbs[idx, 3] = qd.max(aabbs[idx, 3], px)
    aabbs[idx, 4] = qd.max(aabbs[idx, 4], py)
    aabbs[idx, 5] = qd.max(aabbs[idx, 5], pz)


@qd.func
def aabb_combine_aabb(
    dst: qd.template(),
    dst_idx: qd.i32,
    src: qd.template(),
    src_idx: qd.i32,
):
    """Expand dst[dst_idx] to include src[src_idx]."""
    for k in qd.static(range(3)):
        dst[dst_idx, k] = qd.min(dst[dst_idx, k], src[src_idx, k])
        dst[dst_idx, 3 + k] = qd.max(dst[dst_idx, 3 + k], src[src_idx, 3 + k])


@qd.func
def aabb_overlap(
    a: qd.template(),
    a_idx: qd.i32,
    b: qd.template(),
    b_idx: qd.i32,
):
    """Return 1 if AABB a and b overlap, 0 otherwise.

    Matches cgq ``aabb_overlap`` (plain overlap, no gap tolerance).
    """
    if b[b_idx, 0] > a[a_idx, 3]:
        return 0
    if a[a_idx, 0] > b[b_idx, 3]:
        return 0
    if b[b_idx, 1] > a[a_idx, 4]:
        return 0
    if a[a_idx, 1] > b[b_idx, 4]:
        return 0
    if b[b_idx, 2] > a[a_idx, 5]:
        return 0
    if a[a_idx, 2] > b[b_idx, 5]:
        return 0
    return 1


@qd.func
def aabb_overlap_gap(
    a: qd.template(),
    a_idx: qd.i32,
    b: qd.template(),
    b_idx: qd.i32,
    gap: qd.f64,
):
    """Return 1 if AABB a and b overlap within gap tolerance, 0 otherwise.

    Matches cgq ``aabb_overlap_gap``: boxes overlap iff separation along
    every axis is strictly less than *gap*.
    """
    if (b[b_idx, 0] - a[a_idx, 3]) >= gap:
        return 0
    if (a[a_idx, 0] - b[b_idx, 3]) >= gap:
        return 0
    if (b[b_idx, 1] - a[a_idx, 4]) >= gap:
        return 0
    if (a[a_idx, 1] - b[b_idx, 4]) >= gap:
        return 0
    if (b[b_idx, 2] - a[a_idx, 5]) >= gap:
        return 0
    if (a[a_idx, 2] - b[b_idx, 5]) >= gap:
        return 0
    return 1


@qd.func
def aabb_center(
    aabbs: qd.template(),
    idx: qd.i32,
):
    """Return the center of aabbs[idx]."""
    c = qd.Vector.zero(qd.f64, 3)
    for k in qd.static(range(3)):
        c[k] = (aabbs[idx, k] + aabbs[idx, 3 + k]) * 0.5
    return c


# ---------------------------------------------------------------------------
# Morton codes (30-bit, from cgq bvh_types.cuh)
# ---------------------------------------------------------------------------


@qd.func
def expand_bits(v: qd.u32):
    """Spread 10-bit value into 30-bit 3-interleaved pattern."""
    v = (v * qd.u32(0x00010001)) & qd.u32(0xFF0000FF)
    v = (v * qd.u32(0x00000101)) & qd.u32(0x0F00F00F)
    v = (v * qd.u32(0x00000011)) & qd.u32(0xC30C30C3)
    v = (v * qd.u32(0x00000005)) & qd.u32(0x49249249)
    return v


@qd.func
def morton_code_30bit(x: qd.f64, y: qd.f64, z: qd.f64):
    """Compute 30-bit Morton code from [0,1]-normalized coordinates."""
    resolution = 1024.0
    xc = qd.min(qd.max(x * resolution, 0.0), resolution - 1.0)
    yc = qd.min(qd.max(y * resolution, 0.0), resolution - 1.0)
    zc = qd.min(qd.max(z * resolution, 0.0), resolution - 1.0)
    xx = expand_bits(qd.u32(xc))
    yy = expand_bits(qd.u32(yc))
    zz = expand_bits(qd.u32(zc))
    return (xx << qd.u32(2)) | (yy << qd.u32(1)) | zz


# ---------------------------------------------------------------------------
# Karras 2012 helpers (from cgq bvh_types.cuh)
# ---------------------------------------------------------------------------


@qd.func
def common_upper_bits(a: qd.u64, b: qd.u64):
    """Count leading zeros of (a XOR b).  Equivalent to ``__clzll(a ^ b)``."""
    return qd.i32(qd.math.clz(a ^ b))


@qd.func
def determine_range(
    codes: qd.template(),
    n: qd.i32,
    idx: qd.i32,
):
    """Karras 2012 determine_range: find the range [first, last] for internal node *idx*.

    *codes* is a 1-D array of sorted u64 Morton keys (shape ``(n,)``).
    Returns ``(first, last)`` as an i32 Vector of length 2.
    """
    result = qd.Vector.zero(qd.i32, 2)

    # idx == 0 special case: range is the full array
    # (no early return allowed in quadrants runtime if)
    result[0] = 0
    result[1] = n - 1

    if idx > 0:
        self_code = codes[idx]
        L_delta = common_upper_bits(self_code, codes[idx - 1])
        R_delta = common_upper_bits(self_code, codes[idx + 1])
        d = 1
        if R_delta <= L_delta:
            d = -1

        delta_min = qd.min(L_delta, R_delta)
        l_max = 2
        delta = qd.i32(-1)
        i_tmp = idx + d * l_max
        if i_tmp >= 0:
            if i_tmp < n:
                delta = common_upper_bits(self_code, codes[i_tmp])
        while delta > delta_min:
            l_max = l_max << 1
            i_tmp = idx + d * l_max
            delta = qd.i32(-1)
            if i_tmp >= 0:
                if i_tmp < n:
                    delta = common_upper_bits(self_code, codes[i_tmp])

        ll = qd.i32(0)
        t = l_max >> 1
        while t > 0:
            i_tmp = idx + (ll + t) * d
            delta = qd.i32(-1)
            if i_tmp >= 0:
                if i_tmp < n:
                    delta = common_upper_bits(self_code, codes[i_tmp])
            if delta > delta_min:
                ll = ll + t
            t = t >> 1

        jdx = idx + ll * d

        first = idx
        last = jdx
        if d < 0:
            first = jdx
            last = idx

        result[0] = first
        result[1] = last

    return result


@qd.func
def find_split(
    codes: qd.template(),
    n: qd.i32,
    first: qd.i32,
    last: qd.i32,
):
    """Karras 2012 find_split: binary search for the split position."""
    first_code = codes[first]
    last_code = codes[last]

    # Equal codes: split at midpoint (no early return in runtime if)
    split = (first + last) >> 1

    if first_code != last_code:
        delta_node = common_upper_bits(first_code, last_code)
        split = first
        stride = last - first
        cont = qd.i32(1)
        while cont != 0:
            stride = (stride + 1) >> 1
            middle = split + stride
            if middle < last:
                delta = common_upper_bits(first_code, codes[middle])
                if delta > delta_node:
                    split = middle
            if stride <= 1:
                cont = 0

    return split
