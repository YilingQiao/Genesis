"""GlobalSurfaceManager -- collision-mesh surface topology.

Mirrors cgq ``GlobalSurfaceManager`` / ``GlobalSurfaceContext``.
Owns surface triangle, edge, and vertex index arrays (all global vertex IDs).
Topology is static after init.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd

from genesis.engine.solvers.qipc.sim_system import SimSystem
from genesis.engine.solvers.qipc.surface_area_weights import compute_area_weights

# Host-only: buffer backend. NDARRAY (not qd.field) keeps this subsystem eligible
# for quadrants' fastcache -- see road-map Standing rule 6.
_BACKEND = qd.Backend.NDARRAY


@qd.data_oriented
class GlobalSurfaceManager(SimSystem):
    """Scene-wide collision surface topology (static after init).

    Stores surface triangles, edges, and vertices as arrays of **global vertex
    IDs** matching the ``GlobalVertexManager`` index space. Topology never
    changes during simulation.
    """

    def __init__(
        self,
        n_surf_tri: int,
        n_surf_edges: int,
        n_surf_verts: int,
    ) -> None:
        super().__init__()
        self.n_surf_tri = n_surf_tri
        self.n_surf_edges = n_surf_edges
        self.n_surf_verts = n_surf_verts

        self.n_surf_tri_rt = qd.ndarray(qd.i32, shape=(1,))
        self.n_surf_edges_rt = qd.ndarray(qd.i32, shape=(1,))
        self.n_surf_verts_rt = qd.ndarray(qd.i32, shape=(1,))

        self.surf_triangles = qd.tensor(qd.i32, (max(n_surf_tri, 1), 3), backend=_BACKEND)
        self.surf_edges = qd.tensor(qd.i32, (max(n_surf_edges, 1), 2), backend=_BACKEND)
        self.surf_verts = qd.tensor(qd.i32, (max(n_surf_verts, 1),), backend=_BACKEND)

        # Consistent-IPC per-primitive contact area weights (rest pose, static).
        # Indexed by *surface* primitive index, not global vertex ID.
        self.vert_area_weight = qd.tensor(qd.f64, (max(n_surf_verts, 1),), backend=_BACKEND)
        self.edge_area_weight = qd.tensor(qd.f64, (max(n_surf_edges, 1),), backend=_BACKEND)
        self.face_area_weight = qd.tensor(qd.f64, (max(n_surf_tri, 1),), backend=_BACKEND)

    def do_build(self) -> None:
        self.n_surf_tri_rt.from_numpy(np.array([self.n_surf_tri], dtype=np.int32))
        self.n_surf_edges_rt.from_numpy(np.array([self.n_surf_edges], dtype=np.int32))
        self.n_surf_verts_rt.from_numpy(np.array([self.n_surf_verts], dtype=np.int32))

    def wire_area_weights(self, rest_positions: np.ndarray) -> None:
        """Compute and upload rest-pose contact area weights.

        Mirrors cgq ``GlobalSurfaceManager::wire_area_weights``. ConsistentIPC
        dereferences these for every pair, so they must be wired before any
        contact assembly runs; cgq asserts on a null pointer at setup for the
        same reason.

        Args:
            rest_positions: ``(n_total_verts, 3)`` rest positions in the global
                vertex layout (i.e. ``AffineBodyDynamics.x_bar``).
        """
        vaw, eaw, faw = compute_area_weights(
            self.surf_triangles.to_numpy()[: self.n_surf_tri],
            self.surf_edges.to_numpy()[: self.n_surf_edges],
            self.surf_verts.to_numpy()[: self.n_surf_verts],
            rest_positions,
        )
        if self.n_surf_verts > 0:
            self.vert_area_weight.from_numpy(vaw)
        if self.n_surf_edges > 0:
            self.edge_area_weight.from_numpy(eaw)
        if self.n_surf_tri > 0:
            self.face_area_weight.from_numpy(faw)
