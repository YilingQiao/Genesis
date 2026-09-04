"""ABDPreconditioner -- block-Jacobi preconditioner for ABD bodies.

Mirrors cgq ``ABDDiagPreconditioner``. Gathers same-body diagonal 12x12
blocks from the BCOO matrix, inverts via Cholesky, and applies z = M^-1 r.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.sim_system import SimSystem


@qd.data_oriented
class ABDPreconditioner(SimSystem):
    """12x12 block-Jacobi preconditioner for ABD bodies."""

    def __init__(self, n_bodies: int) -> None:
        super().__init__()
        self.n_bodies = n_bodies
        self.pre_diag = qd.ndarray(qd.f64, shape=(n_bodies * 144,))
        self.inv_diag = qd.ndarray(qd.f64, shape=(n_bodies * 144,))

        # Runtime scalar for loop bounds (avoids recompilation)
        self.n_bodies_rt = qd.ndarray(qd.i32, shape=(1,))

    def do_build(self) -> None:
        import numpy as np

        self.n_bodies_rt.from_numpy(np.array([self.n_bodies], dtype=np.int32))

    def realloc_buffers(self, new_n_bodies: int) -> None:
        """Grow preconditioner buffers. Mirrors cgq pattern."""
        import numpy as np

        self.pre_diag = qd.ndarray(qd.f64, shape=(new_n_bodies * 144,))
        self.inv_diag = qd.ndarray(qd.f64, shape=(new_n_bodies * 144,))
        self.n_bodies_rt.from_numpy(np.array([new_n_bodies], dtype=np.int32))

    @qd.func(requires_top_level=True)
    def build(self, lsys: qd.template()):
        """Build preconditioner from unique BCOO in lsys.

        Phase 1: zero pre_diag.
        Phase 2: gather same-body diagonal blocks from BCOO.
        Phase 3: invert 12x12 per body via Cholesky.
        """
        # Phase 1: zero
        for i in range(self.n_bodies_rt[0] * 144):
            self.pre_diag[i] = qd.f64(0.0)

        # Phase 2: gather same-body blocks
        nnz = lsys.bcoo_nnz[()]
        for s in range(lsys.n_live_rt[0]):
            if s < nnz:
                R = lsys.bcoo_row[s]
                C = lsys.bcoo_col[s]
                body_r = R // 4
                body_c = C // 4
                if body_r == body_c:
                    lr = R - 4 * body_r
                    lc = C - 4 * body_r
                    base = body_r * 144
                    for i in qd.static(range(3)):
                        for j in qd.static(range(3)):
                            val = lsys.bcoo_val[s * 9 + i * 3 + j]
                            qd.atomic_add(
                                self.pre_diag[base + (lr * 3 + i) * 12 + (lc * 3 + j)],
                                val,
                            )
                            if R != C:
                                qd.atomic_add(
                                    self.pre_diag[base + (lc * 3 + j) * 12 + (lr * 3 + i)],
                                    val,
                                )

        # Phase 3: invert per-body 12x12 via Cholesky (use runtime loops to reduce IR size)
        for b in range(self.n_bodies_rt[0]):
            base = b * 144
            # Load H into local registers
            H = qd.Matrix.zero(qd.f64, 12, 12)
            for i in range(12):
                for j in range(12):
                    H[i, j] = self.pre_diag[base + i * 12 + j]

            # Cholesky factorize L
            L = qd.Matrix.zero(qd.f64, 12, 12)
            for col in range(12):
                val = H[col, col]
                for k in range(col):
                    val = val - L[col, k] * L[col, k]
                L[col, col] = qd.sqrt(val)
                inv_lcc = 1.0 / L[col, col]
                for row in range(12):
                    if row > col:
                        val2 = H[row, col]
                        for k in range(col):
                            val2 = val2 - L[row, k] * L[col, k]
                        L[row, col] = val2 * inv_lcc

            # Solve L L^T X = I for each column of inverse
            for c in range(12):
                # Forward: L y = e_c
                y = qd.Vector.zero(qd.f64, 12)
                for i in range(12):
                    val = qd.f64(0.0)
                    if i == c:
                        val = qd.f64(1.0)
                    for k in range(i):
                        val = val - L[i, k] * y[k]
                    y[i] = val / L[i, i]

                # Backward: L^T x = y
                for i_rev in range(12):
                    i = 11 - i_rev
                    val = y[i]
                    for k in range(12):
                        if k > i:
                            val = val - L[k, i] * self.inv_diag[base + k * 12 + c]
                    self.inv_diag[base + i * 12 + c] = val / L[i, i]

    @qd.func(requires_top_level=True)
    def apply(self, lsys: qd.template(), pcg: qd.template()):
        """Apply preconditioner: z = M^-1 * r (per-body 12x12 matvec)."""
        for b in range(self.n_bodies_rt[0]):
            base = b * 144
            dof_base = b * 12
            for i in range(12):
                acc = qd.f64(0.0)
                for j in range(12):
                    acc = acc + self.inv_diag[base + i * 12 + j] * pcg.r[dof_base + j]
                pcg.z[dof_base + i] = acc
