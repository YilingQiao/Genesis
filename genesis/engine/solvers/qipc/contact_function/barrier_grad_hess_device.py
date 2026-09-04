"""Composed barrier gradient + Hessian (@qd.func device version).

Ports cgq ``gipc_barrier_gradient_hessian.h``.
Computes grad (12-vector) and hess (12x12 flat) on GPU.

PFPx is rank-1 (only column 8 is nonzero for PT/EE),
so H_12x12 = PFPx * (lambda0 * q0 * q0^T) * PFPx^T
simplifies to lambda0 * pfpx_col8 * pfpx_col8^T.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.contact_function.pfpx_device import (
    pfpx_ee_device,
    pfpx_pe_device,
    pfpx_pp_device,
    pfpx_pt_device,
)


@qd.func
def _barrier_lambda0_device(
    I5: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    dis_sq: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
) -> qd.f64:
    """Rank-1 Hessian eigenvalue, clamped below ``I5 = gass_t``.

    ``gass_ln`` and ``gass_a`` are ``log(gass_t)`` and ``gass_ln + (gass_t-1)/gass_t``,
    computed host-side by the owning ``ContactSystem`` (road-map Standing rule 5:
    a clamp threshold is a tunable, so it may not be a baked module constant).
    """
    result = qd.f64(0.0)
    if dis_sq < gass_t * dHat_sq:
        g = gass_t
        result = 8.0 * kappa * (g * gass_a * gass_a + (g * g - 1.0) * gass_ln / g + (g - 1.0) * gass_ln * gass_a * 0.5)
    else:
        L = qd.log(I5)
        A = L + (I5 - 1.0) / I5
        result = 8.0 * kappa * (I5 * A * A + (I5 * I5 - 1.0) * L / I5 + (I5 - 1.0) * L * A * 0.5)
    return result


@qd.func
def barrier_grad_hess_pt_device(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """PT barrier grad (12) + Hessian (144 flat row-major)."""
    # Distance
    bx = (v2y - v1y) * (v3z - v1z) - (v2z - v1z) * (v3y - v1y)
    by = (v2z - v1z) * (v3x - v1x) - (v2x - v1x) * (v3z - v1z)
    bz = (v2x - v1x) * (v3y - v1y) - (v2y - v1y) * (v3x - v1x)
    aTb = (v0x - v1x) * bx + (v0y - v1y) * by + (v0z - v1z) * bz
    b2 = bx * bx + by * by + bz * bz
    dis_sq = aTb * aTb / b2
    dis = qd.sqrt(dis_sq)
    d_hat_sqrt = qd.sqrt(dHat_sq)

    # PFPx (12 values = column 8)
    pfpx = qd.Vector.zero(qd.f64, 12)
    pfpx_pt_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, d_hat_sqrt, pfpx)

    I5 = (dis / d_hat_sqrt) * (dis / d_hat_sqrt)
    lnI5 = qd.log(I5)
    f22 = dis / d_hat_sqrt

    A = lnI5 + (I5 - 1.0) / I5
    pk1_scalar = 4.0 * kappa * (I5 - 1.0) * lnI5 * A

    # grad = PFPx * f22 * pk1_scalar (rank-1: only col 8 nonzero, tmp[8] = f22)
    for i in range(12):
        grad_out[i] = pfpx[i] * f22 * pk1_scalar

    lambda0 = _barrier_lambda0_device(I5, dHat_sq, kappa, dis_sq, gass_t, gass_ln, gass_a)
    for i in range(12):
        for j in range(12):
            hess_out[i * 12 + j] = lambda0 * pfpx[i] * pfpx[j]


@qd.func
def barrier_grad_hess_ee_device(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """EE barrier grad (12) + Hessian (144 flat row-major)."""
    bx = (v1y - v0y) * (v3z - v2z) - (v1z - v0z) * (v3y - v2y)
    by = (v1z - v0z) * (v3x - v2x) - (v1x - v0x) * (v3z - v2z)
    bz = (v1x - v0x) * (v3y - v2y) - (v1y - v0y) * (v3x - v2x)
    aTb = (v2x - v0x) * bx + (v2y - v0y) * by + (v2z - v0z) * bz
    b2 = bx * bx + by * by + bz * bz
    dis_sq = aTb * aTb / b2
    dis = qd.sqrt(dis_sq)
    d_hat_sqrt = qd.sqrt(dHat_sq)

    pfpx = qd.Vector.zero(qd.f64, 12)
    pfpx_ee_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, d_hat_sqrt, pfpx)

    I5 = (dis / d_hat_sqrt) * (dis / d_hat_sqrt)
    lnI5 = qd.log(I5)
    f22 = dis / d_hat_sqrt
    A = lnI5 + (I5 - 1.0) / I5
    pk1_scalar = 4.0 * kappa * (I5 - 1.0) * lnI5 * A

    for i in range(12):
        grad_out[i] = pfpx[i] * f22 * pk1_scalar

    lambda0 = _barrier_lambda0_device(I5, dHat_sq, kappa, dis_sq, gass_t, gass_ln, gass_a)
    for i in range(12):
        for j in range(12):
            hess_out[i * 12 + j] = lambda0 * pfpx[i] * pfpx[j]


@qd.func
def barrier_grad_hess_pp_device(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """PP barrier grad (6) + Hessian (36 flat row-major)."""
    dx = v0x - v1x
    dy = v0y - v1y
    dz = v0z - v1z
    dis_sq = dx * dx + dy * dy + dz * dz
    dis = qd.sqrt(dis_sq)
    d_hat_sqrt = qd.sqrt(dHat_sq)

    pfpx = qd.Vector.zero(qd.f64, 6)
    pfpx_pp_device(v0x, v0y, v0z, v1x, v1y, v1z, d_hat_sqrt, pfpx)

    I5 = (dis / d_hat_sqrt) * (dis / d_hat_sqrt)
    lnI5 = qd.log(I5)
    fnn = dis / d_hat_sqrt
    A = lnI5 + (I5 - 1.0) / I5
    pk1_scalar = 4.0 * kappa * fnn * (I5 - 1.0) * lnI5 * A

    for i in range(6):
        grad_out[i] = pfpx[i] * pk1_scalar

    lambda0 = _barrier_lambda0_device(I5, dHat_sq, kappa, dis_sq, gass_t, gass_ln, gass_a)
    for i in range(6):
        for j in range(6):
            hess_out[i * 6 + j] = lambda0 * pfpx[i] * pfpx[j]


@qd.func
def barrier_grad_hess_pe_device(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """PE barrier grad (9) + Hessian (81 flat row-major)."""
    # PE distance: ||(e0-p) x (e1-p)||^2 / ||e1-e0||^2
    cx = (v1y - v0y) * (v2z - v0z) - (v1z - v0z) * (v2y - v0y)
    cy = (v1z - v0z) * (v2x - v0x) - (v1x - v0x) * (v2z - v0z)
    cz = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
    ex = v2x - v1x
    ey = v2y - v1y
    ez = v2z - v1z
    dis_sq = (cx * cx + cy * cy + cz * cz) / (ex * ex + ey * ey + ez * ez)
    dis = qd.sqrt(dis_sq)
    d_hat_sqrt = qd.sqrt(dHat_sq)

    pfpx = qd.Vector.zero(qd.f64, 9)
    pfpx_pe_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, d_hat_sqrt, pfpx)

    I5 = (dis / d_hat_sqrt) * (dis / d_hat_sqrt)
    lnI5 = qd.log(I5)
    f22 = dis / d_hat_sqrt

    A = lnI5 + (I5 - 1.0) / I5
    pk1_scalar = 4.0 * kappa * (I5 - 1.0) * lnI5 * A

    # PE PFPx has only column 3 nonzero, so tmp[3] = f22
    for i in range(9):
        grad_out[i] = pfpx[i] * f22 * pk1_scalar

    lambda0 = _barrier_lambda0_device(I5, dHat_sq, kappa, dis_sq, gass_t, gass_ln, gass_a)
    for i in range(9):
        for j in range(9):
            hess_out[i * 9 + j] = lambda0 * pfpx[i] * pfpx[j]
