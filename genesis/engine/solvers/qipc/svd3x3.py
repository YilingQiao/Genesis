"""Accurate 3x3 SVD -- ``qd.svd`` plus one warm-started Jacobi cleanup sweep.

``qd.svd`` is a Sifakis-family solver with a fixed Jacobi budget (8 sweeps for
f64). It returns *exact* singular values, and ``U``/``V`` orthogonal to 1e-16,
but on a near-degenerate spectrum those two frames come back slightly
mis-rotated, so the reconstruction ``U S V^T`` matches ``A`` only to ~1e-10:

    A = I + 1e-6 * randn      recon 9.9e-11   (numpy: 4.4e-16)
    diag(1, 1+1e-9, 1.3)      recon 1.0e-12   (numpy: 0)
    diag(1.3, 0.9, 0.5)       recon 4.4e-16   (numpy: 0)

Raising the budget does not help -- 32 sweeps reproduce the 8-sweep answer bit
for bit -- so this is a stagnation of the fixed-point rotation, not an iteration
shortfall.

That matters because a near-degenerate spectrum is the *common* case for affine
bodies: a body at rest has ``A = I``, which is triply degenerate. Any expression
of the form ``U f(S) V^T`` with ``f`` not the identity then inherits the 1e-10
instead of the singular values' 1e-16, and the ABD shape term multiplies it by
``4 kappa ~ 4e8``.

The fix is cheap because the returned frame is already almost right. ``V`` is
supposed to diagonalise ``A^T A``; forming ``B = V^T (A^T A) V`` leaves a nearly
diagonal matrix, and one Jacobi sweep over its three off-diagonal pairs -- from
that warm start, so quadratically convergent -- lands at the fp64 floor:

    A = I + 1e-6 * randn      recon 9.9e-11 -> 2.2e-16
    diag(1, 1+1e-9, 1.3)      recon 1.0e-12 -> 0

A second sweep changes nothing, which is the usual sign that one was enough.

Convention: singular values are returned *signed*, with ``det U = det V = +1``,
matching what ``qd.svd`` already does. cgq's ``eigen3/svd3x3.h`` instead forces
``sigma >= 0``. The two agree on everything that matters here -- flipping
``sigma_3`` only relabels the twist and flip modes of the Smith et al.
eigensystem (they differ solely in the sign of the ``s_i s_j`` cross term), so
the mode set and the PSD projection built from it are identical.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def _jacobi_rotate(B: qd.template(), V: qd.template(), p: qd.template(), q: qd.template()):
    """Zero ``B[p, q]`` with a Givens rotation, accumulating it into ``V``.

    Standard symmetric Jacobi. ``tau`` is formed as the ratio that stays bounded
    for the near-degenerate case this exists to fix: as ``B[p,p] -> B[q,q]`` it
    goes to zero and the rotation tends to the 45-degree one, which is exactly
    the rotation a degenerate pair needs. A vanishing off-diagonal leaves ``tau``
    huge, ``t`` underflows to zero, and the rotation degenerates to the identity
    -- so no epsilon guard is needed in either limit.
    """
    apq = B[p, q]
    if apq != 0.0:
        tau = (B[q, q] - B[p, p]) / (2.0 * apq)
        denom = qd.abs(tau) + qd.sqrt(1.0 + tau * tau)
        t = 1.0 / denom
        if tau < 0.0:
            t = -t
        c = 1.0 / qd.sqrt(1.0 + t * t)
        s = t * c

        for i in qd.static(range(3)):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq

        # Rebuild the rotated block rather than applying the usual incremental
        # update: at 3x3 the saving is irrelevant and the direct form cannot
        # drift out of symmetry over the sweep.
        for i in qd.static(range(3)):
            bip = B[i, p]
            biq = B[i, q]
            B[i, p] = c * bip - s * biq
            B[i, q] = s * bip + c * biq
        for j in qd.static(range(3)):
            bpj = B[p, j]
            bqj = B[q, j]
            B[p, j] = c * bpj - s * bqj
            B[q, j] = s * bpj + c * bqj


@qd.func
def svd3x3_polished(
    A: qd.types.matrix(3, 3, qd.f64),
    U_out: qd.template(),
    s_out: qd.template(),
    V_out: qd.template(),
):
    """SVD of ``A`` with the singular vectors polished to the fp64 floor.

    Writes ``U_out`` (3x3), ``s_out`` (3-vec, signed, descending as ``qd.svd``
    orders them) and ``V_out`` (3x3), satisfying ``A = U diag(s) V^T`` to ~1e-16
    even for a near-degenerate spectrum. Use this instead of ``qd.svd`` whenever
    the singular *vectors* feed an expression other than the reconstruction
    itself.
    """
    _, _, V0 = qd.svd(A, qd.f64)

    AtA = A.transpose() @ A
    B = V0.transpose() @ AtA @ V0
    V = V0

    _jacobi_rotate(B, V, 0, 1)
    _jacobi_rotate(B, V, 0, 2)
    _jacobi_rotate(B, V, 1, 2)

    # sigma from the column norms of A V, sign fixed so that both frames are
    # rotations. Column norms are used rather than sqrt of B's diagonal because
    # they are the quantity U is normalised by, so the two stay consistent.
    AV = A @ V
    for k in qd.static(range(3)):
        nrm = qd.sqrt(AV[0, k] * AV[0, k] + AV[1, k] * AV[1, k] + AV[2, k] * AV[2, k])
        s_out[k] = nrm
        if nrm > 0.0:
            for i in qd.static(range(3)):
                U_out[i, k] = AV[i, k] / nrm
        else:
            # Rank-deficient column: leave a placeholder, completed below.
            for i in qd.static(range(3)):
                U_out[i, k] = qd.f64(0.0)

    # Complete a degenerate third column so U stays a frame. Only the last
    # column can be short here: qd.svd orders sigma descending and the polish
    # preserves that order.
    if s_out[2] == 0.0:
        U_out[0, 2] = U_out[1, 0] * U_out[2, 1] - U_out[2, 0] * U_out[1, 1]
        U_out[1, 2] = U_out[2, 0] * U_out[0, 1] - U_out[0, 0] * U_out[2, 1]
        U_out[2, 2] = U_out[0, 0] * U_out[1, 1] - U_out[1, 0] * U_out[0, 1]

    det_u = (
        U_out[0, 0] * (U_out[1, 1] * U_out[2, 2] - U_out[1, 2] * U_out[2, 1])
        - U_out[0, 1] * (U_out[1, 0] * U_out[2, 2] - U_out[1, 2] * U_out[2, 0])
        + U_out[0, 2] * (U_out[1, 0] * U_out[2, 1] - U_out[1, 1] * U_out[2, 0])
    )
    det_v = (
        V[0, 0] * (V[1, 1] * V[2, 2] - V[1, 2] * V[2, 1])
        - V[0, 1] * (V[1, 0] * V[2, 2] - V[1, 2] * V[2, 0])
        + V[0, 2] * (V[1, 0] * V[2, 1] - V[1, 1] * V[2, 0])
    )

    # Push any reflection into the trailing singular value, so U and V are both
    # rotations. det U and det V are each +-1, so this is a sign test.
    if det_u < 0.0:
        for i in qd.static(range(3)):
            U_out[i, 2] = -U_out[i, 2]
        s_out[2] = -s_out[2]
    if det_v < 0.0:
        for i in qd.static(range(3)):
            V[i, 2] = -V[i, 2]
        s_out[2] = -s_out[2]

    for i in qd.static(range(3)):
        for j in qd.static(range(3)):
            V_out[i, j] = V[i, j]
