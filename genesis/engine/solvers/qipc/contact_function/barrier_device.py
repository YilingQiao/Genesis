"""IPC barrier functions (@qd.func device version).

Provides both consistent-IPC and GIPC Rank-2 barrier formulas.
Rank-2 is used for alignment with GIPC ground truth data.

Consistent IPC (production): b = κ (r-1)² ln²(r),  r = d²/d̂²
GIPC Rank-2 (alignment):     b = κ (d²-d̂²)² ln²(r)

The difference is (r-1) = (d²-d̂²)/d̂², so Rank-2 = Consistent × d̂⁴.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def barrier_energy_device(d_sq: qd.f64, dHat_sq: qd.f64, kappa: qd.f64) -> qd.f64:
    """Consistent IPC barrier energy: κ(r-1)²ln²(r)."""
    r = d_sq / dHat_sq
    lnr = qd.log(r)
    rm1 = r - 1.0
    return kappa * rm1 * rm1 * lnr * lnr


@qd.func
def barrier_energy_rank2_device(d_sq: qd.f64, dHat_sq: qd.f64, kappa: qd.f64) -> qd.f64:
    """GIPC Rank-2 barrier energy: κ(d²-d̂²)²ln²(r).

    Ports cgq ``gipc_barrier_rank2.h::gipc_barrier_energy``.
    """
    I5 = d_sq / dHat_sq
    lnI5 = qd.log(I5)
    lenE = d_sq - dHat_sq
    return kappa * lenE * lenE * lnI5 * lnI5


@qd.func
def barrier_energy_mollified_device(
    d_sq: qd.f64,
    dHat_sq: qd.f64,
    kappa: qd.f64,
    I1: qd.f64,
    eps_x: qd.f64,
) -> qd.f64:
    """Mollified consistent-IPC barrier: ``m(I1, eps_x) * b(d², d̂², κ)``.

    Ports cgq ``gipc_barrier.h::gipc_barrier_energy_mollified``. The mollifier
    ramps from 0 at parallel edges (``I1 = 0``) to 1 at ``I1 = eps_x``; see
    ``barrier_grad_hess_mollified_device`` for why the EE barrier needs it.
    """
    mollifier = -(I1 * I1) / (eps_x * eps_x) + 2.0 * I1 / eps_x
    return mollifier * barrier_energy_device(d_sq, dHat_sq, kappa)


@qd.func
def barrier_first_derivative_device(d_sq: qd.f64, dHat_sq: qd.f64, kappa: qd.f64) -> qd.f64:
    """Consistent IPC db/d(d²).

    Ports cgq ``gipc_barrier.h::gipc_barrier_first_derivative``, whose
    ``d_tilde = D - xi^2`` and ``dH_tilde = d_hat^2 + 2 d_hat xi`` reduce to
    ``d_sq`` and ``dHat_sq`` at the zero thickness qipc currently carries.
    """
    r = d_sq / dHat_sq
    lnr = qd.log(r)
    A = lnr + (r - 1.0) / r
    return 2.0 * kappa * (r - 1.0) * lnr * A / dHat_sq


@qd.func
def barrier_second_derivative_device(d_sq: qd.f64, dHat_sq: qd.f64, kappa: qd.f64) -> qd.f64:
    """Consistent IPC d²b/d(d²)² = ``2κ(A² + (r²-1)ln(r)/r²) / d̂⁴``.

    Ports cgq ``gipc_barrier.h::gipc_barrier_second_derivative``. Only the
    half-plane channel needs it directly: PT and EE reach the second derivative
    through the ``pfpx`` chain instead, since their Hessian is not a scalar
    multiple of a fixed outer product.
    """
    r = d_sq / dHat_sq
    lnr = qd.log(r)
    A = lnr + (r - 1.0) / r
    return 2.0 * kappa * (A * A + (r * r - 1.0) * lnr / (r * r)) / (dHat_sq * dHat_sq)
