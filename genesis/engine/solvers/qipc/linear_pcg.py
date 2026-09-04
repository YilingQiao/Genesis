"""LinearPCG -- preconditioned conjugate gradient solver.

Mirrors cgq ``LinearPCG``. Owns all iterative workspace vectors and
PCG scalar slots. Reads BCOO from GlobalLinearSystem and preconditioner
from ABDPreconditioner via template() parameters.

Sign convention: r = +gradient (not negated). Solves H x = g.
After PCG, dq = -x.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.sim_system import SimSystem


@qd.data_oriented
class LinearPCG(SimSystem):
    """Preconditioned Conjugate Gradient solver.

    Operates on the BCOO matrix stored in ``GlobalLinearSystem`` with
    preconditioner from ``ABDPreconditioner``.
    """

    def __init__(self, ndof: int) -> None:
        super().__init__()
        self.ndof = ndof

        # Runtime scalar for loop bounds (avoids recompilation)
        self.ndof_rt = qd.ndarray(qd.i32, shape=(1,))

        self.r = qd.ndarray(qd.f64, shape=(ndof,))
        self.z = qd.ndarray(qd.f64, shape=(ndof,))
        self.p = qd.ndarray(qd.f64, shape=(ndof,))
        self.Ap = qd.ndarray(qd.f64, shape=(ndof,))

        # Solver scalars. Named device buffers rather than slots in a shared
        # array: a slot index is a module constant that device code would read
        # as a global, which never enters the fastcache key.
        self.rz = qd.ndarray(qd.f64, shape=(1,))
        self.rz0 = qd.ndarray(qd.f64, shape=(1,))
        self.rz_new = qd.ndarray(qd.f64, shape=(1,))
        self.pAp = qd.ndarray(qd.f64, shape=(1,))
        self.alpha = qd.ndarray(qd.f64, shape=(1,))
        self.beta = qd.ndarray(qd.f64, shape=(1,))
        self.bn2 = qd.ndarray(qd.f64, shape=(1,))
        self.active = qd.ndarray(qd.f64, shape=(1,))

        self.pcond = qd.ndarray(qd.i32, shape=())
        self.iter_buf = qd.ndarray(qd.i32, shape=(1,))

    def do_build(self) -> None:
        import numpy as np

        self.ndof_rt.from_numpy(np.array([self.ndof], dtype=np.int32))

    @qd.func(requires_top_level=True)
    def init(self, lsys: qd.template(), precond: qd.template()):
        """PCG init: x=0, r=b_rhs, z=M^-1 r, p=z, capture rz0."""
        ndof = self.ndof_rt[0]

        # x = 0, r = b_rhs
        for i in range(ndof):
            lsys.x_sol[i] = qd.f64(0.0)
            self.r[i] = lsys.b_rhs[i]

        # bn2 = r·r
        for _ in range(1):
            self.bn2[0] = qd.f64(0.0)
        for i in range(ndof):
            qd.atomic_add(self.bn2[0], self.r[i] * self.r[i])

        # z = M^-1 r (preconditioner apply)
        precond.apply(lsys, self)

        # p = z
        for i in range(ndof):
            self.p[i] = self.z[i]

        # rz = r·z
        for _ in range(1):
            self.rz[0] = qd.f64(0.0)
        for i in range(ndof):
            qd.atomic_add(self.rz[0], self.r[i] * self.z[i])

        for _ in range(1):
            self.rz0[0] = self.rz[0]
            if self.bn2[0] > 0.0:
                self.active[0] = qd.f64(1.0)
            else:
                self.active[0] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def iteration(
        self,
        lsys: qd.template(),
        precond: qd.template(),
        pcg_tol: qd.template(),
        pcg_max: qd.template(),
    ):
        """One PCG iteration: SpMV, dot, update, convergence check."""
        ndof = self.ndof_rt[0]

        # 1. Ap = 0
        for i in range(ndof):
            if self.active[0] > 0.5:
                self.Ap[i] = qd.f64(0.0)

        # 2. Symmetric SpMV with subgroup segmented reduce
        #
        # Row contribution (B * p[col] -> Ap[row]) uses a segmented scan:
        # lane 0 is ALWAYS a head, so a row segment never spans a subgroup.
        #
        # segmented_reduce_add is an *inclusive scan*, so the segment total
        # lands on the segment's LAST lane, not its head -- unlike cub's
        # HeadSegmentedReduce that cgq uses. The partial sum is therefore
        # written from the tail. A row split across subgroups emits one partial
        # per subgroup (the last lane is always a tail) and atomics combine them.
        #
        # Transpose contribution (B^T * p[row] -> Ap[col]) for off-diagonal
        # entries is written directly via atomic_add (no reduce needed).
        #
        # group_size() resolves to a Python int at compile time, so the padding
        # and the tail test follow the backend's subgroup width (32 / 64).
        n_lanes = qd.simt.subgroup.group_size()
        nnz = lsys.bcoo_nnz[()]
        nnz_pad = ((nnz + n_lanes - 1) // n_lanes) * n_lanes
        for s in range(nnz_pad):
            if self.active[0] > 0.5:
                R = qd.i32(-1)
                C = qd.i32(-1)
                y0 = qd.f64(0.0)
                y1 = qd.f64(0.0)
                y2 = qd.f64(0.0)

                if s < nnz:
                    R = lsys.bcoo_row[s]
                    C = lsys.bcoo_col[s]
                    xc0 = self.p[C * 3 + 0]
                    xc1 = self.p[C * 3 + 1]
                    xc2 = self.p[C * 3 + 2]
                    b0 = lsys.bcoo_val[s * 9 + 0]
                    b1 = lsys.bcoo_val[s * 9 + 1]
                    b2 = lsys.bcoo_val[s * 9 + 2]
                    b3 = lsys.bcoo_val[s * 9 + 3]
                    b4 = lsys.bcoo_val[s * 9 + 4]
                    b5 = lsys.bcoo_val[s * 9 + 5]
                    b6 = lsys.bcoo_val[s * 9 + 6]
                    b7 = lsys.bcoo_val[s * 9 + 7]
                    b8 = lsys.bcoo_val[s * 9 + 8]
                    y0 = b0 * xc0 + b1 * xc1 + b2 * xc2
                    y1 = b3 * xc0 + b4 * xc1 + b5 * xc2
                    y2 = b6 * xc0 + b7 * xc1 + b8 * xc2

                    # Off-diagonal: B^T * p[row] -> Ap[col]
                    if R != C:
                        xi0 = self.p[R * 3 + 0]
                        xi1 = self.p[R * 3 + 1]
                        xi2 = self.p[R * 3 + 2]
                        qd.atomic_add(self.Ap[C * 3 + 0], b0 * xi0 + b3 * xi1 + b6 * xi2)
                        qd.atomic_add(self.Ap[C * 3 + 1], b1 * xi0 + b4 * xi1 + b7 * xi2)
                        qd.atomic_add(self.Ap[C * 3 + 2], b2 * xi0 + b5 * xi1 + b8 * xi2)

                # Segment bounds: lane 0 is ALWAYS a head, so a segment never
                # spans a tile; otherwise a lane heads a row if the previous
                # lane's row differs.
                lane = qd.i32(qd.simt.subgroup.invocation_id())
                is_head = qd.i32(1)
                if lane > 0:
                    if s > 0:
                        if s - 1 < nnz:
                            prev_R = lsys.bcoo_row[s - 1]
                            if prev_R == R:
                                is_head = qd.i32(0)

                # A lane is a tail when it ends the subgroup or the next lane
                # starts a new row segment.
                is_tail = qd.i32(1)
                if lane < n_lanes - 1:
                    next_R = qd.i32(-1)
                    if s + 1 < nnz:
                        next_R = lsys.bcoo_row[s + 1]
                    if next_R == R:
                        is_tail = qd.i32(0)

                s0 = qd.simt.subgroup.segmented_reduce_add(y0, is_head)
                s1 = qd.simt.subgroup.segmented_reduce_add(y1, is_head)
                s2 = qd.simt.subgroup.segmented_reduce_add(y2, is_head)

                # Tail threads hold the segment total; write it via atomicAdd
                if is_tail == 1:
                    if s < nnz:
                        qd.atomic_add(self.Ap[R * 3 + 0], s0)
                        qd.atomic_add(self.Ap[R * 3 + 1], s1)
                        qd.atomic_add(self.Ap[R * 3 + 2], s2)

        # 3. pAp = p·Ap
        for _ in range(1):
            self.pAp[0] = qd.f64(0.0)
        for i in range(ndof):
            if self.active[0] > 0.5:
                qd.atomic_add(self.pAp[0], self.p[i] * self.Ap[i])

        # 4. alpha = rz / pAp
        for _ in range(1):
            if self.active[0] > 0.5:
                if self.pAp[0] > 0.0:
                    self.alpha[0] = self.rz[0] / self.pAp[0]
                else:
                    self.alpha[0] = qd.f64(0.0)
                    self.active[0] = qd.f64(0.0)

        # 5. x += alpha*p, r -= alpha*Ap
        for i in range(ndof):
            if self.active[0] > 0.5:
                lsys.x_sol[i] = lsys.x_sol[i] + self.alpha[0] * self.p[i]
                self.r[i] = self.r[i] - self.alpha[0] * self.Ap[i]

        # 6. z = M^-1 * r
        precond.apply(lsys, self)

        # 7. rz_new = r·z
        for _ in range(1):
            self.rz_new[0] = qd.f64(0.0)
        for i in range(ndof):
            if self.active[0] > 0.5:
                qd.atomic_add(self.rz_new[0], self.r[i] * self.z[i])

        # 8-10. Convergence, beta, p update
        for _ in range(1):
            if self.active[0] > 0.5:
                # Convergence: |rz_new| <= tol * |rz0|
                if qd.abs(self.rz_new[0]) <= pcg_tol * qd.abs(self.rz0[0]):
                    self.active[0] = qd.f64(0.0)
                else:
                    # beta = rz_new / rz
                    if qd.abs(self.rz[0]) > 0.0:
                        self.beta[0] = self.rz_new[0] / self.rz[0]
                    else:
                        self.beta[0] = qd.f64(0.0)
                    self.rz[0] = self.rz_new[0]

        # p = z + beta*p
        for i in range(ndof):
            if self.active[0] > 0.5:
                self.p[i] = self.z[i] + self.beta[0] * self.p[i]

        # iter++, max check
        for _ in range(1):
            it = self.iter_buf[0] + 1
            self.iter_buf[0] = it
            if it >= pcg_max:
                self.active[0] = qd.f64(0.0)
            if self.active[0] < 0.5:
                self.pcond[()] = 0
            else:
                self.pcond[()] = 1
