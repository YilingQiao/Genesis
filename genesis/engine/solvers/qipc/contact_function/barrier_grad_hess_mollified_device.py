"""Mollified barrier gradient + Hessian (@qd.func device version).

Ports the mollified half of cgq ``gipc_barrier_gradient_hessian_rank2.h``, which
is what the ConsistentIPC constitution dispatches to for near-parallel edge
pairs (``consistent_ipc_contact.cu``).

Why a separate barrier at all: the interior edge-edge distance is a projection
onto the edge cross product, so as two edges approach parallel it becomes
ill-conditioned and the closest-feature classification jitters. The unmollified
barrier is then *discontinuous* in x, and a pair's whole energy can blink
between two nearby evaluations. Below ``eps_x = 1e-3 |ea|^2 |eb|^2`` (rest
lengths) the barrier is therefore multiplied by the C1 mollifier

    m(I1) = 2 I1 / eps_x - (I1 / eps_x)^2,    I1 = ||ea x eb||^2

which ramps to zero as the edges become parallel, exactly as upstream
GIPC/StiffGIPC.

cgq's mollified derivatives live in the ``gipc_rank2`` namespace and implement
the ``dH_tilde^2``-prefixed barrier family, so ConsistentIPC calls them with
``kappa / dH_tilde^2`` to recover the un-prefixed family (every coefficient is
linear in kappa, and PD projection commutes with positive scaling). The
coefficients below have that conversion folded in, so they take the plain
ConsistentIPC kappa and ``dHat_sq`` plays the role of ``dH_tilde``.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.contact_function.make_pd_device import make_pd_2x2_device
from genesis.engine.solvers.qipc.contact_function.pfpx_mollified_device import (
    pfpx_ee_mollified_device,
    pfpx_pe_mollified_device,
    pfpx_pp_mollified_device,
)


@qd.func
def compute_eps_x_device(
    eps_x_coeff: qd.f64,
    r0x: qd.f64,
    r0y: qd.f64,
    r0z: qd.f64,
    r1x: qd.f64,
    r1y: qd.f64,
    r1z: qd.f64,
    r2x: qd.f64,
    r2y: qd.f64,
    r2z: qd.f64,
    r3x: qd.f64,
    r3y: qd.f64,
    r3z: qd.f64,
) -> qd.f64:
    """Mollifier threshold ``coeff * |r0-r1|^2 * |r2-r3|^2`` from rest positions.

    Ports cgq ``gipc_compute_eps_x``, whose ``1e-3`` arrives here as a runtime
    scalar: it is a tunable, so it may not be baked into device code.
    """
    ax = r0x - r1x
    ay = r0y - r1y
    az = r0z - r1z
    bx = r2x - r3x
    by = r2y - r3y
    bz = r2z - r3z
    return eps_x_coeff * (ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz)


@qd.func
def mollifier_device(I1: qd.f64, eps_x: qd.f64) -> qd.f64:
    """C1 mollifier ``2 I1/eps_x - (I1/eps_x)^2``, zero at parallel, 1 at eps_x."""
    return -(I1 * I1) / (eps_x * eps_x) + 2.0 * I1 / eps_x


@qd.func
def _barrier_grad_hess_mollified_impl_device(
    c4: qd.template(),
    c8: qd.template(),
    dis_sq: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """Shared mollified grad (12) + Hessian (144 flat) from PFPx columns 4, 8.

    ``c4``/``c8`` are the only nonzero columns of the mollified 12x9 PFPx, and
    the projected 9x9 ``projH`` is nonzero only at ``(3,3)``, ``(7,7)`` and the
    ``(4,8)`` 2x2 block. The first two multiply zero columns, so
    ``PFPx projH PFPx^T`` reduces to the rank-2 form assembled below.
    """
    if I1 == 0.0:
        for i in range(12):
            grad_out[i] = qd.f64(0.0)
        for i in range(144):
            hess_out[i] = qd.f64(0.0)
    else:
        I2 = dis_sq / dHat_sq
        dis = qd.sqrt(dis_sq)
        d_hat_sqrt = qd.sqrt(dHat_sq)
        c = qd.sqrt(I1)
        f22 = dis / d_hat_sqrt
        lnI2 = qd.log(I2)
        lnI2_sq = lnI2 * lnI2
        eps_x_sq = eps_x * eps_x
        I2m1 = I2 - 1.0
        A2 = lnI2 + I2m1 / I2

        # dm/dI1 and m(I1) folded into the two PK1 entries.
        p1 = (kappa * 4.0) * (eps_x - I1) / eps_x_sq * I2m1 * I2m1 * lnI2_sq
        m_I1 = I1 * (2.0 * eps_x - I1) / eps_x_sq
        p2 = (kappa * 4.0) * m_I1 * I2m1 * lnI2 * A2

        for i in range(12):
            grad_out[i] = c4[i] * (p1 * c) + c8[i] * (p2 * f22)

        lambda10 = (kappa * 4.0) * (eps_x - 3.0 * I1) / eps_x_sq * I2m1 * I2m1 * lnI2_sq

        lambda20 = qd.f64(0.0)
        if dis_sq < gass_t * dHat_sq:
            g = gass_t
            lambda20 = (
                m_I1
                * 8.0
                * kappa
                * (g * gass_a * gass_a + (g * g - 1.0) * gass_ln / g + (g - 1.0) * gass_ln * gass_a * 0.5)
            )
        else:
            lambda20 = m_I1 * 8.0 * kappa * (I2 * A2 * A2 + (I2 * I2 - 1.0) * lnI2 / I2 + I2m1 * lnI2 * A2 * 0.5)

        lambdag1g = 16.0 * kappa * (eps_x - I1) * I2m1 * lnI2 * A2 * c * f22 / eps_x_sq

        pd = qd.Vector.zero(qd.f64, 3)
        make_pd_2x2_device(lambda10, lambdag1g, lambda20, pd)
        b00 = pd[0]
        b01 = pd[1]
        b11 = pd[2]

        for i in range(12):
            for j in range(12):
                hess_out[i * 12 + j] = b00 * c4[i] * c4[j] + b01 * (c4[i] * c8[j] + c8[i] * c4[j]) + b11 * c8[i] * c8[j]


@qd.func
def barrier_grad_hess_ee_mollified_device(
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
    dis_sq: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """EE mollified grad (12) + Hessian (144 flat).

    ``dis_sq`` is passed in rather than recomputed: the caller substitutes the
    flagged distance when the interior EE projection leaves the filtered band,
    and the energy kernel has to make the identical substitution.
    """
    c4 = qd.Vector.zero(qd.f64, 12)
    c8 = qd.Vector.zero(qd.f64, 12)
    pfpx_ee_mollified_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, qd.sqrt(dHat_sq), c4, c8)
    _barrier_grad_hess_mollified_impl_device(
        c4, c8, dis_sq, I1, eps_x, dHat_sq, kappa, gass_t, gass_ln, gass_a, grad_out, hess_out
    )


@qd.func
def barrier_grad_hess_pp_mollified_device(
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
    dis_sq: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """PP mollified grad (12) + Hessian (144 flat), vertex order (A, B, A', B')."""
    c4 = qd.Vector.zero(qd.f64, 12)
    c8 = qd.Vector.zero(qd.f64, 12)
    pfpx_pp_mollified_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, qd.sqrt(dHat_sq), c4, c8)
    _barrier_grad_hess_mollified_impl_device(
        c4, c8, dis_sq, I1, eps_x, dHat_sq, kappa, gass_t, gass_ln, gass_a, grad_out, hess_out
    )


@qd.func
def barrier_grad_hess_pe_mollified_device(
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
    dis_sq: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    gass_t: qd.f64,
    gass_ln: qd.f64,
    gass_a: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
):
    """PE mollified grad (12) + Hessian (144 flat), vertex order (P, E0, E1, P')."""
    c4 = qd.Vector.zero(qd.f64, 12)
    c8 = qd.Vector.zero(qd.f64, 12)
    pfpx_pe_mollified_device(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, qd.sqrt(dHat_sq), c4, c8)
    _barrier_grad_hess_mollified_impl_device(
        c4, c8, dis_sq, I1, eps_x, dHat_sq, kappa, gass_t, gass_ln, gass_a, grad_out, hess_out
    )
