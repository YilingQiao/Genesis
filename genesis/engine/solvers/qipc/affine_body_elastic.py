"""ABD elastic energy -- analytic eigensystem for orthogonal potential.

Port of cgq ``abd_ortho_analytic.h`` + ``abd_bdf1.cu`` energy functions.
Uses Smith et al. 2019 analytic eigensystem for PSD-projected Hessian.

Energy: Psi = kappa * sum_i (sigma_i^2 - 1)^2
where sigma_i are singular values of the 3x3 affine block A.

Relation to cgq
---------------
cgq is the correctness standard, so the eigenvalues and the Hessian are its
expressions transcribed literally: the twist/flip difference quotients
``(g_i -+ g_j) / (s_i -+ s_j)`` behind the ``SIGMA_EPS`` guard, and the Hessian
as ``sum_k max(lam_k, 0) q_k q_k^T``. Reproducing the formulas reproduces cgq's
rounding, which is what agreement with its golden data actually requires: over
``test_linsys_golden.py``'s 64 frames these forms match cgq's assembled Hessian
to 4.5e-14 worst case, while an algebraically-exact factored rewrite of the same
eigenvalues -- more accurate in absolute terms -- drifted to ~1.5e-9.

Two deliberate departures, both measured rather than assumed.

**The gradient does not go through the SVD.** cgq writes it as
``U diag(dPsi/dsigma) V^T``; this port uses the equal closed form
``4 kappa A (A^T A - I)``. Every quantity here carries a ``4 kappa`` prefactor
(``kappa`` is 1e8 in the golden scene) so the assembled system amplifies
eigensystem error by ``4 kappa dt^2 ~ 4e4``, and the literal form feeds that
amplifier the *difference between our SVD basis and cgq's*. Going through ``A``
directly cannot: golden rhs agreement is 2.4e-9 for the closed form against
1.04e-8 for the literal one.

**cgq's SVD is not replicated.** cgq calls Eigen's
``SelfAdjointEigenSolver::computeDirect`` on ``A^T A``, which solves the
characteristic cubic in closed form. Eigen shifts and rescales first, so on a
generic ``A = I + eps R`` it matches LAPACK at the fp64 floor -- but it fails at
a *near-double* root, and ``sigma = (s, s, t)`` is ordinary uniaxial loading for
an affine body. Eigen forms ``q = a_over_3^3 - half_b^2``, the cubic's
discriminant up to a positive factor, and the discriminant is the product of
squared root differences, so ``q ~ gap^2`` (measured: ``q/gap^2`` is flat at 1.62
over nine decades). ``q`` therefore sinks below the ulp of its own O(1) operands
once ``gap < sqrt(eps) ~ 1.5e-8``, the classic sqrt(eps) barrier for a multiple
root, and past that it is noise: where it merely loses its digits the splitting
survives but is wrong, and where rounding drives it negative ``max(q, 0)``
flattens it, ``theta = atan2(0, half_b)/3`` is exactly zero, and the first two
roots coincide *by construction* for an error of exactly ``gap/2``. Measured on
a generic basis: up to 9e-10 on sigma, which ``4 kappa`` turns into 0.55
absolute against reference entries of 1e8. An iterative solver never forms the
polynomial and stays at 1e-16, because the eigenvalues of a symmetric matrix are
perfectly conditioned in the *matrix* and only ill-conditioned in the
characteristic polynomial's *coefficients*. Filed upstream with a standalone
reproducer as cuda-graph-qipc#344, together with the same defect in cgq's
``singular_values_3x3``, whose ``acos(r)/3`` diverges at a double root for the
sibling reason.

So the SVD here is ``svd3x3_polished``, a Jacobi cleanup sweep over ``qd.svd``.
That is not gold-plating either. Raw ``qd.svd`` returns fp64-exact singular
values but mis-rotated singular *vectors*: ``U`` and ``V`` are orthogonal to
1e-16 while ``U S V^T`` reconstructs ``A`` only to ~1e-10 on the near-degenerate
spectrum an at-rest body has, and raising its sweep budget does not move the
answer one bit (32 sweeps reproduce 8 exactly, so it is stagnation rather than
an iteration shortfall). Times ``4 kappa`` that 1e-10 is ~4e-2 -- and 3e-2 is
precisely what this file used to disagree with cgq by. The mis-rotation, not the
conditioning of cgq's expressions, was the original bug.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.svd3x3 import svd3x3_polished
from genesis.engine.solvers.qipc.affine_body_ortho_analytic import (
    abd_ortho_d2Psi_dsigma2,
    abd_ortho_dPsi_dsigma,
)

# -----------------------------------------------------------------------
# Analytic gradient + PSD Hessian (Step C)
# -----------------------------------------------------------------------


@qd.func
def _add_positive_mode(
    hess: qd.template(),
    lam: qd.f64,
    U: qd.template(),
    V: qd.template(),
    p: qd.template(),
    r: qd.template(),
    val_pr: qd.f64,
    val_rp: qd.f64,
):
    """Accumulate ``lam q q^T`` into ``hess`` for a twist/flip mode, if ``lam > 0``.

    Pattern ``M[p,r] = val_pr``, ``M[r,p] = val_rp``; ``q = vec(U M V^T)``
    row-major. Skipping the non-positive modes *is* the PSD projection.
    """
    if lam > 0.0:
        q = qd.Vector.zero(qd.f64, 9)
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                q[i * 3 + j] = U[i, p] * val_pr * V[j, r] + U[i, r] * val_rp * V[j, p]
        for i in range(9):
            for j in range(9):
                hess[i, j] = hess[i, j] + lam * q[i] * q[j]


@qd.func
def _add_positive_scale_mode(
    hess: qd.template(),
    lam: qd.f64,
    U: qd.template(),
    V: qd.template(),
    k: qd.template(),
):
    """Accumulate a scale mode (pattern ``M[k,k] = 1``) if its eigenvalue is positive."""
    if lam > 0.0:
        q = qd.Vector.zero(qd.f64, 9)
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                q[i * 3 + j] = U[i, k] * V[j, k]
        for i in range(9):
            for j in range(9):
                hess[i, j] = hess[i, j] + lam * q[i] * q[j]


@qd.func
def abd_ortho_analytic_assemble(
    grad_out: qd.template(),
    hess_out: qd.template(),
    A: qd.types.matrix(3, 3, qd.f64),
    kappa: qd.f64,
    sigma_eps: qd.f64,
):
    """Compute gradient (9-vec) and PSD-projected Hessian (9x9) of the
    orthogonality energy w.r.t. vec(A) in row-major order.

    Eigenvalues and Hessian follow cgq ``abd_ortho_analytic.h`` term for term,
    so that the result carries cgq's rounding and not merely cgq's mathematics.
    The gradient takes the equal SVD-free form instead, and the SVD underneath
    is polished; the module docstring has the measurements for both.
    """
    U = qd.Matrix.zero(qd.f64, 3, 3)
    s = qd.Vector.zero(qd.f64, 3)
    V = qd.Matrix.zero(qd.f64, 3, 3)
    svd3x3_polished(A, U, s, V)

    g = qd.Vector.zero(qd.f64, 3)
    abd_ortho_dPsi_dsigma(g, kappa, s)
    d2Psi = qd.Vector.zero(qd.f64, 3)
    abd_ortho_d2Psi_dsigma2(d2Psi, kappa, s)

    # dPsi/dA = U diag(4 kappa s (s^2 - 1)) V^T = 4 kappa A (A^T A - I). Taking
    # the right-hand form keeps the stiff ``4 kappa`` amplifier away from any
    # discrepancy between this SVD's basis and the one cgq's solver picks.
    AtA_m_I = A.transpose() @ A
    for i in qd.static(range(3)):
        AtA_m_I[i, i] = AtA_m_I[i, i] - 1.0
    grad_mat = A @ AtA_m_I
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            grad_out[i * 3 + j] = 4.0 * kappa * grad_mat[i, j]

    # Twist/flip eigenvalues as cgq's difference quotients, falling back to the
    # L'Hopital limit once the singular values are closer than ``sigma_eps``.
    lam_twist = qd.Vector.zero(qd.f64, 3)
    lam_flip = qd.Vector.zero(qd.f64, 3)
    for m in qd.static(range(3)):
        p = qd.static(0 if m < 2 else 1)
        r = qd.static(1 if m == 0 else 2)
        ssum = s[p] + s[r]
        if ssum > sigma_eps:
            lam_twist[m] = (g[p] + g[r]) / ssum
        else:
            lam_twist[m] = 0.5 * (d2Psi[p] + d2Psi[r])
        dif = s[p] - s[r]
        if qd.abs(dif) > sigma_eps:
            lam_flip[m] = (g[p] - g[r]) / dif
        else:
            lam_flip[m] = 0.5 * (d2Psi[p] + d2Psi[r])

    # PSD projection: accumulate only the modes with a positive eigenvalue.
    # Mode pattern matrices M (3x3), Q = U * M * V^T, then vec(Q) row-major.
    for i in range(9):
        for j in range(9):
            hess_out[i, j] = qd.f64(0.0)

    inv_sqrt2 = 1.0 / qd.sqrt(qd.f64(2.0))

    # Twist modes (antisymmetric): M[p,r]=+1/sqrt2, M[r,p]=-1/sqrt2
    _add_positive_mode(hess_out, lam_twist[0], U, V, 0, 1, inv_sqrt2, -inv_sqrt2)
    _add_positive_mode(hess_out, lam_twist[1], U, V, 0, 2, inv_sqrt2, -inv_sqrt2)
    _add_positive_mode(hess_out, lam_twist[2], U, V, 1, 2, inv_sqrt2, -inv_sqrt2)

    # Flip modes (symmetric): M[p,r]=M[r,p]=+1/sqrt2
    _add_positive_mode(hess_out, lam_flip[0], U, V, 0, 1, inv_sqrt2, inv_sqrt2)
    _add_positive_mode(hess_out, lam_flip[1], U, V, 0, 2, inv_sqrt2, inv_sqrt2)
    _add_positive_mode(hess_out, lam_flip[2], U, V, 1, 2, inv_sqrt2, inv_sqrt2)

    # Scale modes (diagonal): M[k,k]=1, eigenvalue d2Psi/dsigma_k^2
    _add_positive_scale_mode(hess_out, d2Psi[0], U, V, 0)
    _add_positive_scale_mode(hess_out, d2Psi[1], U, V, 1)
    _add_positive_scale_mode(hess_out, d2Psi[2], U, V, 2)


# -----------------------------------------------------------------------
# Dyadic mass helpers (Step D)
# -----------------------------------------------------------------------


@qd.func
def dyadic_mass_mul(
    m: qd.f64,
    m_xbar: qd.types.vector(3, qd.f64),
    m_xx: qd.types.matrix(3, 3, qd.f64),
    d: qd.types.vector(12, qd.f64),
) -> qd.types.vector(12, qd.f64):
    """Compute ``M @ d`` using the dyadic mass decomposition (m, m_xbar, m_xx).

    Mirrors cgq ``abd_compact_mass_mul``. ``q[3 + r*3 + c]`` is ``A[r, c]``
    (row-major), so a translation DOF couples only with the *matching row* of A,
    and within one row of A the *column* indices couple through ``m_xx``.
    """
    result = qd.Vector.zero(qd.f64, 12)
    dt_vec = qd.Vector.zero(qd.f64, 3)
    for a in qd.static(range(3)):
        dt_vec[a] = d[a]

    # Translation rows: m * dt[a] + sum_l m_xbar[l] * dA[a, l]
    for a in qd.static(range(3)):
        val = m * dt_vec[a]
        for lc in qd.static(range(3)):
            val = val + m_xbar[lc] * d[3 + a * 3 + lc]
        result[a] = val

    # Affine rows: m_xbar[l] * dt[r] + sum_k m_xx[l, k] * dA[r, k]
    for r in qd.static(range(3)):
        for lc in qd.static(range(3)):
            val = m_xbar[lc] * dt_vec[r]
            for k in qd.static(range(3)):
                val = val + m_xx[lc, k] * d[3 + r * 3 + k]
            result[3 + r * 3 + lc] = val

    return result


@qd.func
def dyadic_mass_energy(
    m: qd.f64,
    m_xbar: qd.types.vector(3, qd.f64),
    m_xx: qd.types.matrix(3, 3, qd.f64),
    d: qd.types.vector(12, qd.f64),
) -> qd.f64:
    """Compute 0.5 * d^T M d using dyadic form."""
    md = dyadic_mass_mul(m, m_xbar, m_xx, d)
    result = qd.f64(0.0)
    for i in range(12):
        result = result + d[i] * md[i]
    return 0.5 * result


@qd.func
def dyadic_mass_to_mat12(
    m: qd.f64,
    m_xbar: qd.types.vector(3, qd.f64),
    m_xx: qd.types.matrix(3, 3, qd.f64),
) -> qd.types.matrix(12, 12, qd.f64):
    """Reconstruct the full 12x12 mass matrix on device from dyadic components.

    Same layout as the host ``abd_math.dyadic_mass_to_mat12`` (cgq
    ``dyadic_mass_to_mat12``): ``q[3 + r*3 + c]`` is ``A[r, c]``, so ``t_a``
    couples with row ``a`` of A and each row of A self-couples through ``m_xx``.
    """
    M = qd.Matrix.zero(qd.f64, 12, 12)
    for a in qd.static(range(3)):
        M[a, a] = m
    # t-A coupling: t_a couples with row a of A.
    for a in qd.static(range(3)):
        for lc in qd.static(range(3)):
            M[a, 3 + a * 3 + lc] = m_xbar[lc]
            M[3 + a * 3 + lc, a] = m_xbar[lc]
    # A-A coupling: each row of A self-couples through m_xx.
    for r in qd.static(range(3)):
        for lc in qd.static(range(3)):
            for k in qd.static(range(3)):
                M[3 + r * 3 + lc, 3 + r * 3 + k] = m_xx[lc, k]
    return M


# -----------------------------------------------------------------------
# Ortho potential energy (scalar, for line search) (Step E)
# -----------------------------------------------------------------------


@qd.func
def ortho_potential_energy(
    kappa: qd.f64,
    A: qd.types.matrix(3, 3, qd.f64),
) -> qd.f64:
    """E_shape = kappa * ||A A^T - I||_F^2."""
    AAt = A @ A.transpose()
    result = qd.f64(0.0)
    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            d = AAt[i, j]
            if i == j:
                d = d - 1.0
            result = result + d * d
    return kappa * result


# -----------------------------------------------------------------------
# Per-body energy (Step E)
# -----------------------------------------------------------------------


@qd.func
def compute_body_energy(
    q: qd.types.vector(12, qd.f64),
    q_tilde: qd.types.vector(12, qd.f64),
    m: qd.f64,
    m_xbar: qd.types.vector(3, qd.f64),
    m_xx: qd.types.matrix(3, 3, qd.f64),
    kappa_vol: qd.f64,
    dt: qd.f64,
    is_fixed: qd.i32,
) -> qd.f64:
    """Total energy for one body: E_kin + dt^2 * E_shape."""
    result = qd.f64(0.0)

    if is_fixed == 0:
        dq = qd.Vector.zero(qd.f64, 12)
        for i in range(12):
            dq[i] = q[i] - q_tilde[i]

        E_kin = dyadic_mass_energy(m, m_xbar, m_xx, dq)

        A = qd.Matrix.zero(qd.f64, 3, 3)
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                A[i, j] = q[3 + i * 3 + j]

        dt2 = dt * dt
        E_shape = dt2 * ortho_potential_energy(kappa_vol, A)

        result = E_kin + E_shape

    return result
