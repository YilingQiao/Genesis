"""AffineBodyDynamics -- @qd.data_oriented SimSystem for ABD state and kernels.

Mirrors cgq ``ABDContext`` + ``AffineBodyDynamics``. QIPCSolver.build computes dyadic mass / gravity acceleration on
the host and uploads everything into these device buffers.

Buffers are ``qd.tensor(..., backend=NDARRAY)`` rather than ``qd.field``:
``qd.field`` members are rejected by quadrants' fastcache arg hasher and
would disqualify the whole step kernel (road-map Standing rule 6). Layout is
unchanged -- 2D scalar buffers ``(n, k)`` indexed with ``qd.static(range(k))``
loops, following the pattern validated in ``examples/sim_hierarchy_demo.py``.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd

from genesis.engine.solvers.qipc.affine_body_elastic import (
    abd_ortho_analytic_assemble,
    compute_body_energy,
)
from genesis.engine.solvers.qipc.affine_body_elastic import dyadic_mass_mul as _gpu_mass_mul
from genesis.engine.solvers.qipc.affine_body_elastic import dyadic_mass_to_mat12 as _gpu_mass_to_mat12
from genesis.engine.solvers.qipc.affine_body_jacobian import abd_JT_H_J, abd_JT_mul_g
from genesis.engine.solvers.qipc.affine_body_math import (
    compute_abd_gravity,
    compute_dyadic_mass,
    compute_mesh_volume,
    dyadic_mass_to_mat12,
    invert_mass_12x12,
    transform_to_q,
)
from genesis.engine.solvers.qipc.sim_system import SimSystem

# Host-only: buffer backend. NDARRAY (not qd.field) keeps this subsystem eligible
# for quadrants' fastcache -- see road-map Standing rule 6.
_BACKEND = qd.Backend.NDARRAY


@qd.data_oriented
class AffineBodyDynamics(SimSystem):
    """Owns all ABD per-body and per-vertex state on device.

    Construction is two-phase: the caller allocates with the counts, uploads every buffer with ``from_numpy``, then
    registers the instance on ``SimEngine`` via ``add_system()``.
    """

    def __init__(self, n_bodies: int, n_verts: int, sigma_eps: float = 1e-10) -> None:
        super().__init__()
        self.n_bodies = n_bodies
        self.n_verts = n_verts
        # cgq's ``SIGMA_EPS`` (abd_ortho_analytic.h): below this singular-value
        # separation the shape eigenvalues fall back to the L'Hopital limit
        # instead of the difference quotient. A tunable, hence a device scalar.
        self.sigma_eps = sigma_eps

        # Runtime scalar for loop bounds (avoids recompilation when count changes)
        self.n_bodies_rt = qd.ndarray(qd.i32, shape=(1,))
        self.sigma_eps_rt = qd.ndarray(qd.f64, shape=(1,))

        # --- Per-body state (topology-fixed) ---
        self.q = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.q_prev = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.q_v = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.mass_m = qd.tensor(qd.f64, (n_bodies,), backend=_BACKEND)
        self.mass_m_xbar = qd.tensor(qd.f64, (n_bodies, 3), backend=_BACKEND)
        self.mass_m_xx = qd.tensor(qd.f64, (n_bodies, 3, 3), backend=_BACKEND)
        self.mass_inv = qd.tensor(qd.f64, (n_bodies, 12, 12), backend=_BACKEND)
        self.gravity_acc = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.kappa_vol = qd.tensor(qd.f64, (n_bodies,), backend=_BACKEND)
        self.is_fixed = qd.tensor(qd.i32, (n_bodies,), backend=_BACKEND)

        # --- Per-vertex data ---
        self.x_bar = qd.tensor(qd.f64, (n_verts, 3), backend=_BACKEND)
        self.body_id = qd.tensor(qd.i32, (n_verts,), backend=_BACKEND)

        # --- Per-body workspace (topology-fixed) ---
        self.q_tilde = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.q_temp = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)
        self.dq = qd.tensor(qd.f64, (n_bodies, 12), backend=_BACKEND)

    def do_build(self) -> None:
        if self.n_bodies_rt.to_numpy()[0] == 0:
            self.n_bodies_rt.from_numpy(np.array([self.n_bodies], dtype=np.int32))
        self.sigma_eps_rt.from_numpy(np.array([self.sigma_eps], dtype=np.float64))

    # ------------------------------------------------------------------
    # Kernel methods (@qd.func, called from SimEngine's graph kernel)
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def predict(self, dt: qd.f64):
        """BDF1 predict: ``q_tilde = q + dt*v + dt^2*gravity_acc``.

        Fixed bodies: ``q_tilde = q``.  Also saves ``q_prev = q``.
        """
        for i in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.q_prev[i, j] = self.q[i, j]
            if self.is_fixed[i] != 0:
                for j in range(12):
                    self.q_tilde[i, j] = self.q[i, j]
            else:
                dt2 = dt * dt
                for j in range(12):
                    self.q_tilde[i, j] = self.q[i, j] + dt * self.q_v[i, j] + dt2 * self.gravity_acc[i, j]

    @qd.func(requires_top_level=True)
    def apply_q_tilde(self):
        """Trivial solve: ``q = q_tilde`` (freefall, no elastic/contact forces)."""
        for i in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.q[i, j] = self.q_tilde[i, j]

    @qd.func(requires_top_level=True)
    def velocity_update(self, inv_dt: qd.f64):
        """Post-solve: ``q_v = (q - q_prev) / dt``."""
        for b in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.q_v[b, j] = (self.q[b, j] - self.q_prev[b, j]) * inv_dt

    @qd.func(requires_top_level=True)
    def copy_q_prev(self):
        """End-of-frame: ``q_prev = q`` (cgq Checkpoint 4 ordering)."""
        for i in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.q_prev[i, j] = self.q[i, j]

    # ------------------------------------------------------------------
    # Newton solver methods (elastic)
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def record_q_temp(self):
        """Save q -> q_temp for line search rollback."""
        for b in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.q_temp[b, j] = self.q[b, j]

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha: qd.template()):
        """Line search trial: q = q_temp + alpha * dq.

        ``alpha`` is a ``qd.ndarray(qd.f64, shape=())`` -- read via ``alpha[()]``.
        """
        a = alpha[()]
        for b in range(self.n_bodies_rt[0]):
            if self.is_fixed[b] == 0:
                for j in range(12):
                    self.q[b, j] = self.q_temp[b, j] + a * self.dq[b, j]

    @qd.func(requires_top_level=True)
    def compute_total_energy(self, energy_out: qd.template(), dt: qd.f64):
        """Sum per-body energy into energy_out scalar."""
        for b in range(self.n_bodies_rt[0]):
            q_b = qd.Vector.zero(qd.f64, 12)
            qt_b = qd.Vector.zero(qd.f64, 12)
            for j in range(12):
                q_b[j] = self.q[b, j]
                qt_b[j] = self.q_tilde[b, j]

            m_xbar = qd.Vector.zero(qd.f64, 3)
            for j in qd.static(range(3)):
                m_xbar[j] = self.mass_m_xbar[b, j]
            m_xx = qd.Matrix.zero(qd.f64, 3, 3)
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    m_xx[i, j] = self.mass_m_xx[b, i, j]

            e = compute_body_energy(
                q_b,
                qt_b,
                self.mass_m[b],
                m_xbar,
                m_xx,
                self.kappa_vol[b],
                dt,
                self.is_fixed[b],
            )
            qd.atomic_add(energy_out[()], e)

    @qd.func(requires_top_level=True)
    def compute_max_displacement(self, max_disp_out: qd.template(), inv_dt: qd.f64):
        """Compute max |dq/dt| across all bodies for convergence check."""
        for b in range(self.n_bodies_rt[0]):
            if self.is_fixed[b] == 0:
                for j in range(12):
                    vel = qd.abs(self.dq[b, j] * inv_dt)
                    qd.atomic_max(max_disp_out[()], vel)

    # ------------------------------------------------------------------
    # BCOO assembly methods (for GlobalLinearSystem pipeline)
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def assemble_kinetic(self, lsys: qd.template(), dt: qd.f64):
        """Write kinetic gradient and Hessian (mass matrix) into lsys triplets.

        Triplet region: [0, n_bodies*10). Gradient: atomic_add into lsys.b_rhs.
        """
        for b in range(self.n_bodies_rt[0]):
            q_b = qd.Vector.zero(qd.f64, 12)
            qt_b = qd.Vector.zero(qd.f64, 12)
            for j in range(12):
                q_b[j] = self.q[b, j]
                qt_b[j] = self.q_tilde[b, j]

            m_xbar = qd.Vector.zero(qd.f64, 3)
            for j in qd.static(range(3)):
                m_xbar[j] = self.mass_m_xbar[b, j]
            m_xx = qd.Matrix.zero(qd.f64, 3, 3)
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    m_xx[i, j] = self.mass_m_xx[b, i, j]

            if self.is_fixed[b] != 0:
                # Fixed body: identity Hessian, zero gradient
                slot = b * 10
                idx = 0
                for br in range(4):
                    for bc in range(4):
                        if bc >= br:
                            lsys.tri_row[slot + idx] = 4 * b + br
                            lsys.tri_col[slot + idx] = 4 * b + bc
                            base9 = (slot + idx) * 9
                            for ii in qd.static(range(3)):
                                for jj in qd.static(range(3)):
                                    val = qd.f64(0.0)
                                    if br == bc:
                                        if ii == jj:
                                            val = qd.f64(1.0)
                                    lsys.tri_val[base9 + ii * 3 + jj] = val
                            idx = idx + 1
            else:
                dq_vec = qd.Vector.zero(qd.f64, 12)
                for j in range(12):
                    dq_vec[j] = q_b[j] - qt_b[j]
                g_kin = _gpu_mass_mul(self.mass_m[b], m_xbar, m_xx, dq_vec)
                H_kin = _gpu_mass_to_mat12(self.mass_m[b], m_xbar, m_xx)

                for a in range(12):
                    qd.atomic_add(lsys.b_rhs[b * 12 + a], g_kin[a])

                slot = b * 10
                idx = 0
                for br in range(4):
                    for bc in range(4):
                        if bc >= br:
                            lsys.tri_row[slot + idx] = 4 * b + br
                            lsys.tri_col[slot + idx] = 4 * b + bc
                            base9 = (slot + idx) * 9
                            for ii in qd.static(range(3)):
                                for jj in qd.static(range(3)):
                                    lsys.tri_val[base9 + ii * 3 + jj] = H_kin[br * 3 + ii, bc * 3 + jj]
                            idx = idx + 1

    @qd.func(requires_top_level=True)
    def assemble_shape(self, lsys: qd.template(), dt: qd.f64):
        """Write shape (elastic) gradient and PSD Hessian into lsys triplets.

        Triplet region: [n_bodies*10, 2*n_bodies*10).
        """
        offset = self.n_bodies_rt[0] * 10
        dt2 = dt * dt
        for b in range(self.n_bodies_rt[0]):
            if self.is_fixed[b] != 0:
                slot = offset + b * 10
                idx = 0
                for br in range(4):
                    for bc in range(4):
                        if bc >= br:
                            lsys.tri_row[slot + idx] = 4 * b + br
                            lsys.tri_col[slot + idx] = 4 * b + bc
                            base9 = (slot + idx) * 9
                            for ii in qd.static(range(3)):
                                for jj in qd.static(range(3)):
                                    lsys.tri_val[base9 + ii * 3 + jj] = qd.f64(0.0)
                            idx = idx + 1
            else:
                A = qd.Matrix.zero(qd.f64, 3, 3)
                for i in qd.static(range(3)):
                    for j in qd.static(range(3)):
                        A[i, j] = self.q[b, 3 + i * 3 + j]

                g_shape = qd.Vector.zero(qd.f64, 9)
                H_shape = qd.Matrix.zero(qd.f64, 9, 9)
                abd_ortho_analytic_assemble(g_shape, H_shape, A, self.kappa_vol[b], self.sigma_eps_rt[0])

                for a in range(9):
                    qd.atomic_add(lsys.b_rhs[b * 12 + 3 + a], dt2 * g_shape[a])

                # Embed 9x9 shape Hessian into 12x12, write upper-tri blocks
                H12 = qd.Matrix.zero(qd.f64, 12, 12)
                for i in range(9):
                    for j in range(9):
                        H12[3 + i, 3 + j] = dt2 * H_shape[i, j]

                slot = offset + b * 10
                idx = 0
                for br in range(4):
                    for bc in range(4):
                        if bc >= br:
                            lsys.tri_row[slot + idx] = 4 * b + br
                            lsys.tri_col[slot + idx] = 4 * b + bc
                            base9 = (slot + idx) * 9
                            for ii in qd.static(range(3)):
                                for jj in qd.static(range(3)):
                                    lsys.tri_val[base9 + ii * 3 + jj] = H12[br * 3 + ii, bc * 3 + jj]
                            idx = idx + 1

    # ------------------------------------------------------------------
    # Contact distribute: vertex space -> body DOF space
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def distribute_contact_gradient(self, csys: qd.template(), lsys: qd.template()):
        """Lift deduplicated per-vertex contact gradients onto body DOFs.

        Port of cgq ``distribute_abd_gradient_kernel``. A fixed body is a
        Dirichlet row, so its contribution is dropped entirely rather than
        assembled and later zeroed.
        """
        n_unique = csys.n_unique_doublets[()]
        for i in range(csys.max_contact_doublets[0]):
            if i < n_unique:
                vi = csys.unique_doublet_vert[i]
                body = self.body_id[vi]
                if self.is_fixed[body] == 0:
                    g3 = qd.Vector.zero(qd.f64, 3)
                    x_bar = qd.Vector.zero(qd.f64, 3)
                    for c in qd.static(range(3)):
                        g3[c] = csys.unique_doublet_grad[i * 3 + c]
                        x_bar[c] = self.x_bar[vi, c]

                    g12 = qd.Vector.zero(qd.f64, 12)
                    abd_JT_mul_g(x_bar, g3, g12)
                    for a in range(12):
                        qd.atomic_add(lsys.b_rhs[body * 12 + a], g12[a])

    @qd.func(requires_top_level=True)
    def distribute_contact_hessian(self, csys: qd.template(), lsys: qd.template()):
        """Lift deduplicated 3x3 contact Hessian blocks onto body-DOF blocks.

        Port of cgq ``distribute_abd_abd_kernel``. Each unique vertex-space
        triplet becomes a 4x4 grid of 3x3 blocks coupling the row body's 12 DOFs
        with the column body's, so every triplet owns exactly 16 slots in the
        contact region -- a fixed stride, which is what lets the region's extent
        be a pure function of the unique-triplet count.

        The incoming triplets are upper-triangular in *vertex* space, which says
        nothing about body order, so the pair is reordered to ``(L, R)`` with
        ``L <= R`` and the 3x3 block transposed when the order flips. That keeps
        the emitted blocks upper-triangular in *body* space, as the BCOO SpMV
        requires.

        Within a self-body triplet (``L == R``) only the 10 upper blocks are
        representable; the 6 lower ones are written as zero blocks on the
        diagonal, preserving the 16-slot stride at no cost to the reduce.
        """
        n_unique = csys.n_unique_triplets[()]
        offset = lsys.n_elastic_rt[0]
        for t in range(csys.max_contact_triplets[0]):
            if t < n_unique:
                vi = csys.unique_triplet_row[t]
                vj = csys.unique_triplet_col[t]
                body_i = self.body_id[vi]
                body_j = self.body_id[vj]
                base = offset + t * 16

                if self.is_fixed[body_i] != 0 or self.is_fixed[body_j] != 0:
                    for s in range(16):
                        lsys.tri_row[base + s] = 0
                        lsys.tri_col[base + s] = 0
                        for c in range(9):
                            lsys.tri_val[(base + s) * 9 + c] = qd.f64(0.0)
                else:
                    H3 = qd.Matrix.zero(qd.f64, 3, 3)
                    for a in qd.static(range(3)):
                        for b in qd.static(range(3)):
                            H3[a, b] = csys.unique_triplet_val[t * 9 + a * 3 + b]

                    xi = qd.Vector.zero(qd.f64, 3)
                    xj = qd.Vector.zero(qd.f64, 3)
                    for c in qd.static(range(3)):
                        xi[c] = self.x_bar[vi, c]
                        xj[c] = self.x_bar[vj, c]

                    left = body_i
                    right = body_j
                    if body_j < body_i:
                        left = body_j
                        right = body_i

                    H12 = qd.Matrix.zero(qd.f64, 12, 12)
                    if body_i < body_j:
                        abd_JT_H_J(xi, xj, H3, H12)
                    elif body_j < body_i:
                        H3t = qd.Matrix.zero(qd.f64, 3, 3)
                        for a in qd.static(range(3)):
                            for b in qd.static(range(3)):
                                H3t[a, b] = H3[b, a]
                        abd_JT_H_J(xj, xi, H3t, H12)
                    elif vi != vj:
                        # Same body, distinct vertices: the vertex-space triplet
                        # stands for both (vi, vj) and its mirror, and both land
                        # on this one diagonal body block.
                        H3t = qd.Matrix.zero(qd.f64, 3, 3)
                        for a in qd.static(range(3)):
                            for b in qd.static(range(3)):
                                H3t[a, b] = H3[b, a]
                        Ha = qd.Matrix.zero(qd.f64, 12, 12)
                        Hb = qd.Matrix.zero(qd.f64, 12, 12)
                        abd_JT_H_J(xi, xj, H3, Ha)
                        abd_JT_H_J(xj, xi, H3t, Hb)
                        for a in range(12):
                            for b in range(12):
                                H12[a, b] = Ha[a, b] + Hb[a, b]
                    else:
                        abd_JT_H_J(xi, xi, H3, H12)

                    idx = 0
                    for br in range(4):
                        for bc in range(4):
                            slot = base + idx
                            if left == right and br > bc:
                                lsys.tri_row[slot] = 4 * left + br
                                lsys.tri_col[slot] = 4 * left + br
                                for c in range(9):
                                    lsys.tri_val[slot * 9 + c] = qd.f64(0.0)
                            else:
                                lsys.tri_row[slot] = 4 * left + br
                                lsys.tri_col[slot] = 4 * right + bc
                                for ii in qd.static(range(3)):
                                    for jj in qd.static(range(3)):
                                        lsys.tri_val[slot * 9 + ii * 3 + jj] = H12[br * 3 + ii, bc * 3 + jj]
                            idx = idx + 1

    @qd.func(requires_top_level=True)
    def recover_dq(self, lsys: qd.template()):
        """Recover Newton direction: dq = -x_sol."""
        for b in range(self.n_bodies_rt[0]):
            for j in range(12):
                self.dq[b, j] = -lsys.x_sol[b * 12 + j]
