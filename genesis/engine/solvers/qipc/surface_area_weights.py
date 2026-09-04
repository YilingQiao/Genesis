"""Per-primitive contact area weights for Consistent IPC.

Port of cgq ``Solver._compute_area_weights`` (``qipc/solver/solver.py``),
restricted to the codim2D case: closed triangle surfaces, which is what affine
bodies are. cgq additionally handles codim1D rods (``edge_length * thickness``)
and codim0D points (``thickness**2``); neither exists here yet, and both need a
per-vertex thickness that affine bodies do not carry, so they are left out
rather than guessed at.

ConsistentIPC scales every contact pair by an area so that the barrier is
independent of mesh resolution: refining a mesh subdivides the same contact
patch into more, smaller pairs whose weights sum to what the coarse pair
carried. Weights are computed once from the *rest* pose -- they are a property
of the mesh, not of the current configuration -- and multiplied by ``d_hat`` in
the kernels to form a volume (see ``cipc_pair_area_weight``).
"""

from __future__ import annotations

import numpy as np


def compute_area_weights(
    faces: np.ndarray,
    edges: np.ndarray,
    surf_verts: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest-pose area weights per surface vertex, edge and face.

    Each triangle carries its own rest area, and hands a third of that area to
    each of its three incident surface vertices and each of its three incident
    surface edges. Summing over incident triangles therefore conserves total
    area across all three primitive kinds.

    Args:
        faces: ``(nf, 3)`` int -- global vertex IDs per surface triangle.
        edges: ``(ne, 2)`` int -- global vertex IDs per surface edge.
        surf_verts: ``(nv,)`` int -- ``surf_verts[sv]`` is a global vertex ID.
        positions: ``(n_total, 3)`` float -- rest positions, global indexing.

    Returns:
        ``(vaw, eaw, faw)`` float64 arrays of length ``nv``, ``ne``, ``nf``.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    surf_verts = np.asarray(surf_verts, dtype=np.int64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)

    nv, ne, nf = len(surf_verts), len(edges), len(faces)
    n_total = positions.shape[0]

    vaw = np.zeros(nv, dtype=np.float64)
    eaw = np.zeros(ne, dtype=np.float64)
    faw = np.zeros(nf, dtype=np.float64)
    if nf == 0:
        return vaw, eaw, faw

    # Global vertex ID -> surface vertex index.
    g2sv = np.full(n_total, -1, dtype=np.int64)
    g2sv[surf_verts] = np.arange(nv, dtype=np.int64)

    p0, p1, p2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    faw = areas.copy()

    # bincount rather than np.add.at: same accumulation order, so the sums are
    # bit-identical, but far faster at scale (cgq makes the same swap).
    third = areas / 3.0
    sv_faces = g2sv[faces]
    for c in range(3):
        vaw += np.bincount(sv_faces[:, c], weights=third, minlength=nv)

    if ne > 0:
        # Match each triangle side against the surface edge list by undirected
        # key. Edge keys are unique, so the lookup is exact.
        def key(a, b):
            return np.minimum(a, b) * (n_total + 1) + np.maximum(a, b)

        edge_keys = key(edges[:, 0], edges[:, 1])
        order = np.argsort(edge_keys)
        sorted_keys = edge_keys[order]

        for a_col, b_col in ((0, 1), (1, 2), (0, 2)):
            tri_keys = key(faces[:, a_col], faces[:, b_col])
            pos = np.searchsorted(sorted_keys, tri_keys)
            valid = (pos < ne) & (sorted_keys[np.clip(pos, None, ne - 1)] == tri_keys)
            eaw += np.bincount(order[pos[valid]], weights=third[valid], minlength=ne)

    return vaw, eaw, faw
