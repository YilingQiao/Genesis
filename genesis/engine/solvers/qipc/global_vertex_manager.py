"""GlobalVertexManager -- unified world-space vertex positions for all bodies.

Mirrors cgq ``GlobalVertexManager`` / ``GlobalVertexContext``.
Owns per-vertex position, safe_position, displacement, rest-pose, and body-id
buffers.  All position updates are ``@qd.func`` methods called from the graph
kernel.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd

from genesis.engine.solvers.qipc.sim_system import SimSystem

# Host-only: buffer backend. NDARRAY (not qd.field) keeps this subsystem eligible
# for quadrants' fastcache -- see road-map Standing rule 6.
_BACKEND = qd.Backend.NDARRAY


@qd.data_oriented
class GlobalVertexManager(SimSystem):
    """Unified world-space vertex data for all bodies (ABD + future FEM).

    Vertex index space: ``[0, n_verts)`` where ABD vertices occupy
    ``[0, V_abd)``.  This ordering is fixed at init.

    All buffers are topology-fixed, never reallocated.
    """

    def __init__(self, n_verts: int) -> None:
        super().__init__()
        self.n_verts = n_verts

        self.n_verts_rt = qd.ndarray(qd.i32, shape=(1,))

        self.positions = qd.tensor(qd.f64, (n_verts, 3), backend=_BACKEND)
        self.safe_positions = qd.tensor(qd.f64, (n_verts, 3), backend=_BACKEND)
        self.displacements = qd.tensor(qd.f64, (n_verts, 3), backend=_BACKEND)
        self.x_bar = qd.tensor(qd.f64, (n_verts, 3), backend=_BACKEND)
        self.body_id = qd.tensor(qd.i32, (n_verts,), backend=_BACKEND)

    def do_build(self) -> None:
        if self.n_verts_rt.to_numpy()[0] == 0:
            self.n_verts_rt.from_numpy(np.array([self.n_verts], dtype=np.int32))

    # ------------------------------------------------------------------
    # Kernel methods
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def compute_positions(self, abd: qd.template()):
        """Compute world positions from ABD affine state: ``x = t + A @ x_bar``.

        Matches cgq ``compute_vert_positions_kernel``.
        """
        for vi in range(self.n_verts_rt[0]):
            bi = self.body_id[vi]
            xb0 = self.x_bar[vi, 0]
            xb1 = self.x_bar[vi, 1]
            xb2 = self.x_bar[vi, 2]
            self.positions[vi, 0] = abd.q[bi, 0] + abd.q[bi, 3] * xb0 + abd.q[bi, 4] * xb1 + abd.q[bi, 5] * xb2
            self.positions[vi, 1] = abd.q[bi, 1] + abd.q[bi, 6] * xb0 + abd.q[bi, 7] * xb1 + abd.q[bi, 8] * xb2
            self.positions[vi, 2] = abd.q[bi, 2] + abd.q[bi, 9] * xb0 + abd.q[bi, 10] * xb1 + abd.q[bi, 11] * xb2

    @qd.func(requires_top_level=True)
    def compute_displacements(self, abd: qd.template()):
        """Compute per-vertex displacements from ABD search direction: ``disp = J @ dq``.

        Same linear map as ``compute_positions`` but using ``abd.dq`` instead of ``abd.q``.
        """
        for vi in range(self.n_verts_rt[0]):
            bi = self.body_id[vi]
            xb0 = self.x_bar[vi, 0]
            xb1 = self.x_bar[vi, 1]
            xb2 = self.x_bar[vi, 2]
            self.displacements[vi, 0] = abd.dq[bi, 0] + abd.dq[bi, 3] * xb0 + abd.dq[bi, 4] * xb1 + abd.dq[bi, 5] * xb2
            self.displacements[vi, 1] = abd.dq[bi, 1] + abd.dq[bi, 6] * xb0 + abd.dq[bi, 7] * xb1 + abd.dq[bi, 8] * xb2
            self.displacements[vi, 2] = (
                abd.dq[bi, 2] + abd.dq[bi, 9] * xb0 + abd.dq[bi, 10] * xb1 + abd.dq[bi, 11] * xb2
            )

    @qd.func(requires_top_level=True)
    def record_safe_positions(self):
        """Snapshot ``positions -> safe_positions`` before line search."""
        for vi in range(self.n_verts_rt[0]):
            for k in qd.static(range(3)):
                self.safe_positions[vi, k] = self.positions[vi, k]

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha: qd.template()):
        """Line-search trial: ``positions = safe_positions + alpha * displacements``."""
        a = alpha[()]
        for vi in range(self.n_verts_rt[0]):
            for k in qd.static(range(3)):
                self.positions[vi, k] = self.safe_positions[vi, k] + a * self.displacements[vi, k]

    @qd.func(requires_top_level=True)
    def zero_displacements(self):
        """Zero all displacement vectors."""
        for vi in range(self.n_verts_rt[0]):
            for k in qd.static(range(3)):
                self.displacements[vi, k] = qd.f64(0.0)
