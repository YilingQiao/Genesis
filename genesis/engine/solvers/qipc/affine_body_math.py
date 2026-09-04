"""Affine Body Dynamics -- host-side mass computation and data utilities.

All functions are pure numpy, executed once at init time.
Reference: cgq ``qipc/solver/affine_body.py`` and libuipc ``abd_jacobi_matrix.h``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def compute_tet_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed volume of one tet: ``det(p1-p0, p2-p0, p3-p0) / 6``."""
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    return float(np.dot(d1, np.cross(d2, d3)) / 6.0)


def compute_mesh_volume(positions: np.ndarray, tetrahedra: np.ndarray) -> float:
    """Total mesh volume (sum of signed tet volumes), vectorized."""
    p = positions[tetrahedra]  # (N_tet, 4, 3)
    d1 = p[:, 1] - p[:, 0]
    d2 = p[:, 2] - p[:, 0]
    d3 = p[:, 3] - p[:, 0]
    det = np.einsum("ij,ij->i", d1, np.cross(d2, d3))
    return float(det.sum() / 6.0)


# ---------------------------------------------------------------------------
# Dyadic mass
# ---------------------------------------------------------------------------


def compute_dyadic_mass(
    positions: np.ndarray,
    tetrahedra: np.ndarray,
    density: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Compute dyadic mass decomposition from a tet mesh.

    Returns ``(m, m_xbar[3], m_xbar_xbar[3, 3])``.

    Uses the exact quadrature formulas from libuipc:

    - D = det(p1-p0, p2-p0, p3-p0)
    - m += rho * D / 6
    - m_xbar[a] += rho * D * (p0[a] + p1[a] + p2[a] + p3[a]) / 24
    - m_xbar_xbar[a, b] += rho * D * Q(a, b) where
      Q(a, b) = sum_k p_k[a]*p_k[b]/60 + sum_{k!=l} p_k[a]*p_l[b]/120
    """
    m = 0.0
    m_xbar = np.zeros(3)
    m_xbar_xbar = np.zeros((3, 3))

    for tet in tetrahedra:
        p = positions[tet]  # (4, 3)
        d1 = p[1] - p[0]
        d2 = p[2] - p[0]
        d3 = p[3] - p[0]
        D = np.dot(d1, np.cross(d2, d3))

        m += density * D / 6.0

        for a in range(3):
            m_xbar[a] += density * D * (p[0, a] + p[1, a] + p[2, a] + p[3, a]) / 24.0

        for a in range(3):
            for b in range(3):
                Q = 0.0
                for k in range(4):
                    Q += p[k, a] * p[k, b] / 60.0
                for k in range(4):
                    for j in range(4):
                        if k != j:
                            Q += p[k, a] * p[j, b] / 120.0
                m_xbar_xbar[a, b] += density * D * Q

    return m, m_xbar, m_xbar_xbar


# ---------------------------------------------------------------------------
# 12x12 mass matrix
# ---------------------------------------------------------------------------


def dyadic_mass_to_mat12(m: float, m_xbar: np.ndarray, m_xbar_xbar: np.ndarray) -> np.ndarray:
    r"""Reconstruct the full 12x12 mass matrix from dyadic components.

    Port of cgq ``qipc/solver/affine_body.py::dyadic_mass_to_mat12``.

    Layout for ``q = [t; vec_row(A)]`` with A row-major -- ``q[3 + r*3 + c]`` is
    ``A[r, c]``, matching ``gipc_abd_J_mul_q`` and ``abd_compact_mass_mul``.
    With ``x_world = t + A @ x_bar`` the Jacobian is
    ``J[a, 3 + r*3 + c] = delta(a, r) * x_bar[c]``, so ``M = int rho J^T J dV``
    couples a translation DOF only with the *matching row* of A, and couples
    the *column* indices within one row of A::

        M[a, b]                       = m * delta(a, b)
        M[a, 3 + a*3 + l]             = m_xbar[l]
        M[3 + r*3 + l, 3 + r*3 + k]   = m_xbar_xbar[l, k]
    """
    M = np.zeros((12, 12))
    for a in range(3):
        M[a, a] = m
    # t-A coupling: t_a couples with row a of A.
    for a in range(3):
        for col in range(3):
            M[a, 3 + a * 3 + col] = m_xbar[col]
            M[3 + a * 3 + col, a] = m_xbar[col]
    # A-A coupling: each row of A self-couples through m_xbar_xbar.
    for r in range(3):
        for col in range(3):
            for k in range(3):
                M[3 + r * 3 + col, 3 + r * 3 + k] = m_xbar_xbar[col, k]
    return M


def invert_mass_12x12(M: np.ndarray) -> np.ndarray:
    """Invert the 12x12 mass matrix (numpy, done once at init)."""
    return np.linalg.inv(M)


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------


def compute_abd_gravity(M12: np.ndarray, gravity_vec3: np.ndarray) -> np.ndarray:
    r"""Compute the 12D gravitational generalized force via the body-force integral.

    Port of cgq ``qipc/solver/affine_body.py::compute_abd_gravity``.

    For a uniform body force ``g``, the ABD generalized force is

    .. math:: f = \int_\Omega \rho J(\bar{x})^T g \, dV

    and since ``J(x_bar) @ [g; 0; ...; 0] == g`` (only J's translation block
    contributes), this collapses to ``f = M @ [g; 0; ...; 0]``.

    Taking ``M12`` rather than ``(m, m_xbar)`` deliberately avoids
    convention-specific index arithmetic on the affine block -- the layout is
    already encoded in ``M12`` (cgq makes the same choice for the same reason).
    """
    q_grav = np.zeros(12)
    q_grav[:3] = np.asarray(gravity_vec3, dtype=np.float64)
    return np.asarray(M12, dtype=np.float64).reshape(12, 12) @ q_grav


# ---------------------------------------------------------------------------
# Transform <-> q conversion
# ---------------------------------------------------------------------------


def transform_to_q(transform_4x4: np.ndarray) -> np.ndarray:
    """Extract ``q[12]`` from a 4x4 affine transform (row-major A, GIPC convention)."""
    T = np.asarray(transform_4x4, dtype=np.float64)
    q = np.zeros(12)
    q[0:3] = T[0:3, 3]
    for r in range(3):
        for c in range(3):
            q[3 + r * 3 + c] = T[r, c]
    return q


def q_to_transform(q: np.ndarray) -> np.ndarray:
    """Reconstruct a 4x4 affine transform from ``q[12]`` (row-major A, GIPC convention)."""
    T = np.eye(4)
    T[0:3, 3] = q[0:3]
    for r in range(3):
        for c in range(3):
            T[r, c] = q[3 + r * 3 + c]
    return T


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)


def extract_tet_surface(tetrahedra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary surface primitives of a tet mesh (local vertex indices).

    Returns ``(surf_tris (Nt,3), surf_edges (Ne,2), surf_verts (Nsv,))`` as int32.
    """
    tets = np.asarray(tetrahedra, dtype=np.int64)
    if tets.size == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]
    fs = faces[order]
    diff = np.any(ks[1:] != ks[:-1], axis=1)
    grp_start = np.concatenate(([0], np.where(diff)[0] + 1))
    grp_end = np.concatenate((grp_start[1:], [len(ks)]))
    counts = grp_end - grp_start
    boundary = grp_start[counts == 1]
    surf_tris = fs[boundary]
    if len(surf_tris) == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    e = np.vstack([surf_tris[:, [0, 1]], surf_tris[:, [1, 2]], surf_tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    surf_edges = np.unique(e, axis=0)
    surf_verts = np.unique(surf_tris)
    return surf_tris.astype(np.int32), surf_edges.astype(np.int32), surf_verts.astype(np.int32)
