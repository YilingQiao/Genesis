"""Point vs half-plane contact primitives (``@qd.func`` device layer).

Port of cgq ``contact_function/halfplane_contact.h``. A half-plane is analytic
-- a reference point ``P`` and an outward unit normal ``N``, with the half-space
being ``N . (x - P) >= 0`` -- so unlike every other channel there is no second
primitive to classify against: no distance flag, no stencil, no degenerate
sub-cases. The whole geometry is one signed distance.

That collapses the barrier too. Where PT and EE need an eigendecomposition to
project their 12x12 Hessian, the PH Hessian is a scalar multiple of ``N N^T``,
so the PSD projection is a clamp on that one scalar.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def halfplane_signed_distance_device(
    xx: qd.f64,
    xy: qd.f64,
    xz: qd.f64,
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
) -> qd.f64:
    """Signed distance ``N . (x - P)``; positive outside the half-space.

    ``N`` is taken as a unit vector -- ``ContactSystem.wire_halfplanes``
    rejects anything else, because a non-unit normal would rescale this
    distance and so silently retune ``d_hat`` for that plane.
    """
    return nx * (xx - px) + ny * (xy - py) + nz * (xz - pz)


@qd.func
def halfplane_barrier_gradient_device(
    g_b: qd.f64,
    d: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
    out: qd.template(),
):
    """Write ``(2 g_b d) N`` into ``out`` (3,).

    Chain rule through ``d^2``: ``dE/dx = (db/d(d^2)) * 2 d * N``.
    """
    s = g_b * 2.0 * d
    out[0] = s * nx
    out[1] = s * ny
    out[2] = s * nz


@qd.func
def halfplane_barrier_hessian_device(
    H_b: qd.f64,
    g_b: qd.f64,
    d_sq: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
    out: qd.template(),
):
    """Write the PD-projected ``(4 d^2 H_b + 2 g_b) N N^T`` into ``out`` (9,).

    ``d^2E/dx^2 = H_b (2d N)(2d N)^T + g_b 2 N N^T = (4 d^2 H_b + 2 g_b) N N^T``.
    The matrix is rank one with ``N N^T`` positive semi-definite, so projecting
    it onto the PSD cone is exactly clamping that scalar coefficient at zero --
    no eigendecomposition, unlike the PT and EE Hessians.
    """
    coeff = 4.0 * d_sq * H_b + 2.0 * g_b
    if coeff < 0.0:
        coeff = 0.0
    out[0] = coeff * nx * nx
    out[1] = coeff * nx * ny
    out[2] = coeff * nx * nz
    out[3] = coeff * ny * nx
    out[4] = coeff * ny * ny
    out[5] = coeff * ny * nz
    out[6] = coeff * nz * nx
    out[7] = coeff * nz * ny
    out[8] = coeff * nz * nz
