"""Fused narrowphase filter + barrier assembly (@qd.func device version).

Ports cgq ``contact_kernels.cu`` fused_filter_assemble_pt/ee_contact and
filter_energy_pt/ee_contact.

All per-pair physics -- vertex resolution, distance classification, the d_hat
filter, the barrier grad/Hessian dispatch, the EE mollifier dispatch, the
Consistent-IPC per-pair scale and the barrier energy -- lives in
``pair_eval_device``, shared with the per-pair export in
``export_assemble_device``. What remains here is the cooperative sinks:

- the batch-atomic doublet / upper-triangle-triplet scatter
- the active-pair counter
- the atomic energy slot

That is the same division cgq drew in ``contact_device_functions.h``
(cuda-graph-qipc#354), and it is what keeps the export from drifting away from
production: there is one implementation of the physics, not a twin of it.

Energy is a separate pass from grad/hess (matching cgq's design), so the
assembly kernel never pays for an energy it does not use.

Scaling is Consistent IPC, cgq's production barrier (``consistent_ipc_contact.cu``):
``barrier_grad_hess_*_device`` already returns the consistent-IPC derivative, and
each pair is then weighted by ``dt^2 * cipc_pair_area_weight(wa, wb, d_hat)``.
The area weight makes the barrier independent of mesh resolution, and the
``dt^2`` puts the contact Hessian on the same footing as the dt^2-scaled elastic
terms in the Newton system. (cgq's alternative GIPC Rank-2 barrier instead scales
by ``d_hat_sq^2``, which is resolution dependent; it is not ported here.)
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.contact_function.ccd_device import (
    directional_ee_ccd_device,
    directional_pt_ccd_device,
    halfplane_ccd_device,
)
from genesis.engine.solvers.qipc.contact_function.distance_flag_device import (
    _get_offset,
    _popcount4,
    ee_flagged_distance2_device,
    pt_distance_flag_device,
    pt_flagged_distance2_device,
)
from genesis.engine.solvers.qipc.contact_function.halfplane_device import (
    halfplane_signed_distance_device,
)
from genesis.engine.solvers.qipc.contact_function.pair_eval_device import (
    _ee_flag_device,
    ee_mollifier_state_device,
    eval_pair_ee_device,
    eval_pair_energy_ee_device,
    eval_pair_energy_ph_device,
    eval_pair_energy_pt_device,
    eval_pair_ph_device,
    eval_pair_pt_device,
)

# ---------------------------------------------------------------------------
# Scatter helpers (batch atomic reserve, matching cgq sparse_view.h)
# ---------------------------------------------------------------------------


@qd.func
def _scatter_doublets_device(
    csys: qd.template(),
    flag: qd.i32,
    g0: qd.i32,
    g1: qd.i32,
    g2: qd.i32,
    g3: qd.i32,
    grad_12: qd.template(),
):
    """Scatter gradient 3-vectors into doublet buffers with batch atomic."""
    pc = _popcount4(flag)
    base = qd.atomic_add(csys.n_doublets[()], pc)

    for ii in range(pc):
        oi = _get_offset(flag, ii)
        slot = base + ii

        vid = qd.i32(0)
        if oi == 0:
            vid = g0
        if oi == 1:
            vid = g1
        if oi == 2:
            vid = g2
        if oi == 3:
            vid = g3

        csys.doublet_vert[slot] = vid
        for k in qd.static(range(3)):
            csys.doublet_grad[slot * 3 + k] = grad_12[oi * 3 + k]


@qd.func
def _scatter_triplets_upper_device(
    csys: qd.template(),
    flag: qd.i32,
    g0: qd.i32,
    g1: qd.i32,
    g2: qd.i32,
    g3: qd.i32,
    hess_144: qd.template(),
):
    """Scatter upper-triangle 3x3 Hessian blocks into triplet buffers."""
    pc = _popcount4(flag)
    n_tri = (pc * (pc + 1)) // 2
    base = qd.atomic_add(csys.n_triplets[()], n_tri)

    slot = qd.i32(0)
    for ii in range(pc):
        for jj in range(ii, pc):
            oi = _get_offset(flag, ii)
            oj = _get_offset(flag, jj)

            row_id = qd.i32(0)
            if oi == 0:
                row_id = g0
            if oi == 1:
                row_id = g1
            if oi == 2:
                row_id = g2
            if oi == 3:
                row_id = g3

            col_id = qd.i32(0)
            if oj == 0:
                col_id = g0
            if oj == 1:
                col_id = g1
            if oj == 2:
                col_id = g2
            if oj == 3:
                col_id = g3

            ri = oi
            ci = oj
            if row_id > col_id:
                tmp = row_id
                row_id = col_id
                col_id = tmp
                ri = oj
                ci = oi

            s = base + slot
            csys.triplet_row[s] = row_id
            csys.triplet_col[s] = col_id
            for ki in qd.static(range(3)):
                for kj in qd.static(range(3)):
                    csys.triplet_val[s * 9 + ki * 3 + kj] = hess_144[ri * 3 * 12 + ki * 12 + ci * 3 + kj]

            slot = slot + 1


# ---------------------------------------------------------------------------
# Fused filter + assemble: PT (grad/hess scatter only)
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def fused_filter_assemble_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_pt: qd.template(),
    n_pairs_pt: qd.template(),
):
    """Fused PT narrowphase: ``eval_pair_pt_device`` + scatter.

    Ports cgq ``fused_filter_assemble_pt_contact`` (contact_kernels.cu:172-297).
    Does NOT compute barrier energy (separate kernel).

    pairs_pt: (max_pairs, 2) i32 — broadphase candidate pairs
    n_pairs_pt: scalar i32 — number of valid pairs
    """
    for idx in range(n_pairs_pt[()]):
        gids = qd.Vector.zero(qd.i32, 4)
        meta = qd.Vector.zero(qd.i32, 2)
        scal = qd.Vector.zero(qd.f64, 1)
        grad_12 = qd.Vector.zero(qd.f64, 12)
        hess_144 = qd.Vector.zero(qd.f64, 144)

        eval_pair_pt_device(
            csys,
            surf_mgr,
            vtx_mgr,
            pairs_pt[idx, 0],
            pairs_pt[idx, 1],
            gids,
            meta,
            scal,
            grad_12,
            hess_144,
        )

        if meta[0] == 1:
            qd.atomic_add(csys.n_active_pairs[()], 1)
            _scatter_doublets_device(csys, meta[1], gids[0], gids[1], gids[2], gids[3], grad_12)
            _scatter_triplets_upper_device(csys, meta[1], gids[0], gids[1], gids[2], gids[3], hess_144)


# ---------------------------------------------------------------------------
# Fused filter + assemble: EE (grad/hess scatter only)
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def fused_filter_assemble_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_ee: qd.template(),
    n_pairs_ee: qd.template(),
):
    """Fused EE narrowphase: ``eval_pair_ee_device`` + scatter.

    Ports cgq ``cipc_filter_assemble_ee_kernel`` (``consistent_ipc_contact.cu``).
    The eval hands back a scatter flag (0xF whenever the emitted 12-vector is
    dense, which covers pc=4 and every mollified sub-case) and the column
    permutation, so the scatter carries the permuted global ids that travel with
    a mollified evaluation.

    pairs_ee: (max_pairs, 2) i32 — broadphase candidate pairs
    n_pairs_ee: scalar i32 — number of valid pairs
    """
    for idx in range(n_pairs_ee[()]):
        gids = qd.Vector.zero(qd.i32, 4)
        meta = qd.Vector.zero(qd.i32, 3)
        scal = qd.Vector.zero(qd.f64, 1)
        sperm = qd.Vector.zero(qd.i32, 4)
        grad_12 = qd.Vector.zero(qd.f64, 12)
        hess_144 = qd.Vector.zero(qd.f64, 144)

        eval_pair_ee_device(
            csys,
            surf_mgr,
            vtx_mgr,
            pairs_ee[idx, 0],
            pairs_ee[idx, 1],
            gids,
            meta,
            scal,
            sperm,
            grad_12,
            hess_144,
        )

        if meta[0] == 1:
            qd.atomic_add(csys.n_active_pairs[()], 1)
            s0 = gids[sperm[0]]
            s1 = gids[sperm[1]]
            s2 = gids[sperm[2]]
            s3 = gids[sperm[3]]
            _scatter_doublets_device(csys, meta[2], s0, s1, s2, s3, grad_12)
            _scatter_triplets_upper_device(csys, meta[2], s0, s1, s2, s3, hess_144)


# ---------------------------------------------------------------------------
# Separate energy kernels (matching cgq filter_energy_pt/ee_contact)
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def filter_energy_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_pt: qd.template(),
    n_pairs_pt: qd.template(),
):
    """Compute per-pair PT barrier energy. Separate from grad/hess assembly.

    Ports cgq ``filter_energy_pt_contact`` (contact_kernels.cu:494-556).
    Writes to ``csys.pair_energy[slot]`` with slot from atomic n_active counter.
    """
    for idx in range(n_pairs_pt[()]):
        meta = qd.Vector.zero(qd.i32, 2)
        scal = qd.Vector.zero(qd.f64, 1)

        eval_pair_energy_pt_device(csys, surf_mgr, vtx_mgr, pairs_pt[idx, 0], pairs_pt[idx, 1], meta, scal)

        if meta[1] == 1:
            slot = qd.atomic_add(csys.n_active_pairs[()], 1)
            csys.pair_energy[slot] = scal[0]


@qd.func(requires_top_level=True)
def filter_energy_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_ee: qd.template(),
    n_pairs_ee: qd.template(),
):
    """Compute per-pair EE barrier energy. Separate from grad/hess assembly.

    Ports cgq ``cipc_filter_energy_ee_kernel``. The mollifier dispatch mirrors
    the assembly pass exactly -- same trigger, same distance substitution -- so
    the line-search energy stays consistent with the system that was assembled
    and continuous through near-parallel configurations.

    A pair can be active for the gradient yet contribute no energy (the
    unmollified interior-interior case with a non-positive interior projection),
    which is why the slot is reserved on ``emit`` rather than on ``active``.
    """
    for idx in range(n_pairs_ee[()]):
        meta = qd.Vector.zero(qd.i32, 2)
        scal = qd.Vector.zero(qd.f64, 1)

        eval_pair_energy_ee_device(csys, surf_mgr, vtx_mgr, pairs_ee[idx, 0], pairs_ee[idx, 1], meta, scal)

        if meta[1] == 1:
            slot = qd.atomic_add(csys.n_active_pairs[()], 1)
            csys.pair_energy[slot] = scal[0]


# ---------------------------------------------------------------------------
# Half-plane: query, assemble, energy
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def halfplane_query_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
):
    """Enumerate surface-vertex / plane candidates into ``csys.pairs_ph``.

    Ports cgq ``halfplane_query_kernel``. There is no BVH: a plane has no
    bounding volume worth building against, so cgq sweeps the full
    ``surf_verts x planes`` product, and at one or two planes that is cheaper
    than any tree would be.

    Swept over the pending search direction, like the PT and EE queries whose
    BVH leaves cover ``positions + displacements``: the plane distance is linear
    in the displacement, so the swept lower bound is one extra dot product. A
    static test would skip a vertex whose step crosses the whole ``(0, d_hat)``
    window at once, and a candidate CCD never sees is a candidate it cannot
    clamp: the vertex would tunnel straight through the plane. The assembly
    filter downstream still evaluates at ``positions`` alone, so the swept form
    only widens which inactive candidates get enumerated.

    Pairs are stored as ``(surf_vert_idx, plane_id)`` -- surface-local, so the
    assembly can index the point area weight directly, as cgq notes.
    """
    n_sv = surf_mgr.n_surf_verts_rt[0]
    n_hp = csys.n_halfplanes
    d_hat = csys.d_hat[0]
    cap = csys.max_pairs_ph[0]

    for sv_idx in range(n_sv):
        v0 = surf_mgr.surf_verts[sv_idx]
        for plane_id in range(n_hp):
            d = halfplane_signed_distance_device(
                vtx_mgr.positions[v0, 0],
                vtx_mgr.positions[v0, 1],
                vtx_mgr.positions[v0, 2],
                csys.hp_position[plane_id, 0],
                csys.hp_position[plane_id, 1],
                csys.hp_position[plane_id, 2],
                csys.hp_normal[plane_id, 0],
                csys.hp_normal[plane_id, 1],
                csys.hp_normal[plane_id, 2],
            )
            d_step = (
                csys.hp_normal[plane_id, 0] * vtx_mgr.displacements[v0, 0]
                + csys.hp_normal[plane_id, 1] * vtx_mgr.displacements[v0, 1]
                + csys.hp_normal[plane_id, 2] * vtx_mgr.displacements[v0, 2]
            )
            d_swept = qd.min(d, d + d_step)
            if d > 0.0 and d_swept < d_hat:
                slot = qd.atomic_add(csys.n_pairs_ph[()], 1)
                if slot < cap:
                    csys.pairs_ph[slot, 0] = sv_idx
                    csys.pairs_ph[slot, 1] = plane_id
                else:
                    csys.overflow_flag[()] = 1


@qd.func(requires_top_level=True)
def fused_filter_assemble_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
):
    """Half-plane narrowphase: ``eval_pair_ph_device`` + scatter.

    Ports cgq ``halfplane_filter_assemble_kernel``. The scatter is direct rather
    than flag-driven: the pair couples exactly one real vertex to an analytic
    plane, so it emits one doublet and one diagonal triplet, with no offsets to
    resolve and no upper-triangle swap to make.
    """
    for idx in range(csys.n_pairs_ph[()]):
        gids = qd.Vector.zero(qd.i32, 2)
        meta = qd.Vector.zero(qd.i32, 1)
        scal = qd.Vector.zero(qd.f64, 1)
        grad_3 = qd.Vector.zero(qd.f64, 3)
        hess_9 = qd.Vector.zero(qd.f64, 9)

        eval_pair_ph_device(
            csys,
            surf_mgr,
            vtx_mgr,
            csys.pairs_ph[idx, 0],
            csys.pairs_ph[idx, 1],
            gids,
            meta,
            scal,
            grad_3,
            hess_9,
        )

        if meta[0] == 1:
            qd.atomic_add(csys.n_active_pairs[()], 1)
            v0 = gids[0]

            d_slot = qd.atomic_add(csys.n_doublets[()], 1)
            csys.doublet_vert[d_slot] = v0
            for k in qd.static(range(3)):
                csys.doublet_grad[d_slot * 3 + k] = grad_3[k]

            t_slot = qd.atomic_add(csys.n_triplets[()], 1)
            csys.triplet_row[t_slot] = v0
            csys.triplet_col[t_slot] = v0
            for k in range(9):
                csys.triplet_val[t_slot * 9 + k] = hess_9[k]


@qd.func(requires_top_level=True)
def filter_energy_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
):
    """Per-pair half-plane barrier energy. Ports cgq ``halfplane_filter_energy_kernel``."""
    for idx in range(csys.n_pairs_ph[()]):
        meta = qd.Vector.zero(qd.i32, 2)
        scal = qd.Vector.zero(qd.f64, 1)

        eval_pair_energy_ph_device(csys, surf_mgr, vtx_mgr, csys.pairs_ph[idx, 0], csys.pairs_ph[idx, 1], meta, scal)

        if meta[1] == 1:
            slot = qd.atomic_add(csys.n_active_pairs[()], 1)
            csys.pair_energy[slot] = scal[0]


# ---------------------------------------------------------------------------
# Count-only narrowphase (cgq cipc_count_active_*_kernel)
#
# Same distance filter and flag classification as the assembly pass, but no
# grad/hess/scatter — only doublet/triplet demand accumulated into
# n_counted_doublets / n_counted_triplets via atomic add.
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def count_active_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_pt: qd.template(),
    n_pairs_pt: qd.template(),
):
    """Count doublet/triplet demand for PT pairs. Ports cgq ``cipc_count_active_pt_kernel``."""
    d_hat_sq = csys.d_hat_sq[0]
    for idx in range(n_pairs_pt[()]):
        sv_idx = pairs_pt[idx, 0]
        face_idx = pairs_pt[idx, 1]
        v0 = surf_mgr.surf_verts[sv_idx]
        v1 = surf_mgr.surf_triangles[face_idx, 0]
        v2 = surf_mgr.surf_triangles[face_idx, 1]
        v3 = surf_mgr.surf_triangles[face_idx, 2]

        flag = pt_distance_flag_device(
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            vtx_mgr.positions[v1, 0],
            vtx_mgr.positions[v1, 1],
            vtx_mgr.positions[v1, 2],
            vtx_mgr.positions[v2, 0],
            vtx_mgr.positions[v2, 1],
            vtx_mgr.positions[v2, 2],
            vtx_mgr.positions[v3, 0],
            vtx_mgr.positions[v3, 1],
            vtx_mgr.positions[v3, 2],
        )
        verts = qd.Matrix.zero(qd.f64, 4, 3)
        for c in qd.static(range(3)):
            verts[0, c] = vtx_mgr.positions[v0, c]
            verts[1, c] = vtx_mgr.positions[v1, c]
            verts[2, c] = vtx_mgr.positions[v2, c]
            verts[3, c] = vtx_mgr.positions[v3, c]
        d2 = pt_flagged_distance2_device(flag, verts)

        if d2 > 0.0 and d2 < d_hat_sq:
            pc = _popcount4(flag)
            n_trip = (pc * (pc + 1)) // 2
            qd.atomic_add(csys.n_counted_triplets[()], n_trip)
            qd.atomic_add(csys.n_counted_doublets[()], pc)


@qd.func(requires_top_level=True)
def count_active_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_ee: qd.template(),
    n_pairs_ee: qd.template(),
):
    """Count doublet/triplet demand for EE pairs. Ports cgq ``cipc_count_active_ee_kernel``."""
    d_hat_sq = csys.d_hat_sq[0]
    for idx in range(n_pairs_ee[()]):
        ea = pairs_ee[idx, 0]
        eb = pairs_ee[idx, 1]
        v0 = surf_mgr.surf_edges[ea, 0]
        v1 = surf_mgr.surf_edges[ea, 1]
        v2 = surf_mgr.surf_edges[eb, 0]
        v3 = surf_mgr.surf_edges[eb, 1]

        flag = _ee_flag_device(
            csys,
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            vtx_mgr.positions[v1, 0],
            vtx_mgr.positions[v1, 1],
            vtx_mgr.positions[v1, 2],
            vtx_mgr.positions[v2, 0],
            vtx_mgr.positions[v2, 1],
            vtx_mgr.positions[v2, 2],
            vtx_mgr.positions[v3, 0],
            vtx_mgr.positions[v3, 1],
            vtx_mgr.positions[v3, 2],
        )
        verts = qd.Matrix.zero(qd.f64, 4, 3)
        for c in qd.static(range(3)):
            verts[0, c] = vtx_mgr.positions[v0, c]
            verts[1, c] = vtx_mgr.positions[v1, c]
            verts[2, c] = vtx_mgr.positions[v2, c]
            verts[3, c] = vtx_mgr.positions[v3, c]
        d2 = ee_flagged_distance2_device(flag, verts)

        if d2 > 0.0 and d2 < d_hat_sq:
            pc = _popcount4(flag)
            moll = qd.Vector.zero(qd.f64, 2)
            ee_mollifier_state_device(csys, vtx_mgr, verts, v0, v1, v2, v3, moll)
            I1 = moll[0]
            eps_x = moll[1]
            n_dbl = pc
            n_trip = (pc * (pc + 1)) // 2
            if I1 < eps_x:
                n_dbl = qd.i32(4)
                n_trip = qd.i32(10)
            qd.atomic_add(csys.n_counted_triplets[()], n_trip)
            qd.atomic_add(csys.n_counted_doublets[()], n_dbl)


@qd.func(requires_top_level=True)
def count_active_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
):
    """Count doublet/triplet demand for PH pairs. Ports cgq ``cipc_count_active_ph_kernel``."""
    d_hat = csys.d_hat[0]
    for idx in range(csys.n_pairs_ph[()]):
        sv_idx = csys.pairs_ph[idx, 0]
        plane_id = csys.pairs_ph[idx, 1]
        v0 = surf_mgr.surf_verts[sv_idx]
        d = halfplane_signed_distance_device(
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            csys.hp_position[plane_id, 0],
            csys.hp_position[plane_id, 1],
            csys.hp_position[plane_id, 2],
            csys.hp_normal[plane_id, 0],
            csys.hp_normal[plane_id, 1],
            csys.hp_normal[plane_id, 2],
        )
        if d > 0.0 and d < d_hat:
            qd.atomic_add(csys.n_counted_triplets[()], 1)
            qd.atomic_add(csys.n_counted_doublets[()], 1)


# ---------------------------------------------------------------------------
# CCD per-pair sweep (directional conservative advancement)
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def ccd_alpha_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_pt: qd.template(),
    n_pairs_pt: qd.template(),
):
    """Per-pair directional PT CCD. Atomic-min result into ``csys.ccd_alpha``."""
    result = qd.Vector.zero(qd.f64, 1)
    eta = csys.ccd_eta[0]
    for idx in range(n_pairs_pt[()]):
        sv_idx = pairs_pt[idx, 0]
        face_idx = pairs_pt[idx, 1]
        v0 = surf_mgr.surf_verts[sv_idx]
        v1 = surf_mgr.surf_triangles[face_idx, 0]
        v2 = surf_mgr.surf_triangles[face_idx, 1]
        v3 = surf_mgr.surf_triangles[face_idx, 2]

        directional_pt_ccd_device(
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            vtx_mgr.positions[v1, 0],
            vtx_mgr.positions[v1, 1],
            vtx_mgr.positions[v1, 2],
            vtx_mgr.positions[v2, 0],
            vtx_mgr.positions[v2, 1],
            vtx_mgr.positions[v2, 2],
            vtx_mgr.positions[v3, 0],
            vtx_mgr.positions[v3, 1],
            vtx_mgr.positions[v3, 2],
            vtx_mgr.displacements[v0, 0],
            vtx_mgr.displacements[v0, 1],
            vtx_mgr.displacements[v0, 2],
            vtx_mgr.displacements[v1, 0],
            vtx_mgr.displacements[v1, 1],
            vtx_mgr.displacements[v1, 2],
            vtx_mgr.displacements[v2, 0],
            vtx_mgr.displacements[v2, 1],
            vtx_mgr.displacements[v2, 2],
            vtx_mgr.displacements[v3, 0],
            vtx_mgr.displacements[v3, 1],
            vtx_mgr.displacements[v3, 2],
            eta,
            qd.f64(0.0),
            csys.ccd_max_iters,
            result,
        )
        qd.atomic_min(csys.ccd_alpha[()], result[0])


@qd.func(requires_top_level=True)
def ccd_alpha_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    pairs_ee: qd.template(),
    n_pairs_ee: qd.template(),
):
    """Per-pair directional EE CCD. Atomic-min result into ``csys.ccd_alpha``."""
    result = qd.Vector.zero(qd.f64, 1)
    eta = csys.ccd_eta[0]
    for idx in range(n_pairs_ee[()]):
        ea = pairs_ee[idx, 0]
        eb = pairs_ee[idx, 1]
        v0 = surf_mgr.surf_edges[ea, 0]
        v1 = surf_mgr.surf_edges[ea, 1]
        v2 = surf_mgr.surf_edges[eb, 0]
        v3 = surf_mgr.surf_edges[eb, 1]

        directional_ee_ccd_device(
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            vtx_mgr.positions[v1, 0],
            vtx_mgr.positions[v1, 1],
            vtx_mgr.positions[v1, 2],
            vtx_mgr.positions[v2, 0],
            vtx_mgr.positions[v2, 1],
            vtx_mgr.positions[v2, 2],
            vtx_mgr.positions[v3, 0],
            vtx_mgr.positions[v3, 1],
            vtx_mgr.positions[v3, 2],
            vtx_mgr.displacements[v0, 0],
            vtx_mgr.displacements[v0, 1],
            vtx_mgr.displacements[v0, 2],
            vtx_mgr.displacements[v1, 0],
            vtx_mgr.displacements[v1, 1],
            vtx_mgr.displacements[v1, 2],
            vtx_mgr.displacements[v2, 0],
            vtx_mgr.displacements[v2, 1],
            vtx_mgr.displacements[v2, 2],
            vtx_mgr.displacements[v3, 0],
            vtx_mgr.displacements[v3, 1],
            vtx_mgr.displacements[v3, 2],
            eta,
            qd.f64(0.0),
            csys.ccd_max_iters,
            result,
        )
        qd.atomic_min(csys.ccd_alpha[()], result[0])


@qd.func(requires_top_level=True)
def ccd_alpha_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
):
    """Per-pair halfplane CCD. Atomic-min result into ``csys.ccd_alpha``."""
    result = qd.Vector.zero(qd.f64, 1)
    eta = csys.ccd_eta[0]
    for idx in range(csys.n_pairs_ph[()]):
        sv_idx = csys.pairs_ph[idx, 0]
        plane_id = csys.pairs_ph[idx, 1]
        v0 = surf_mgr.surf_verts[sv_idx]

        halfplane_ccd_device(
            vtx_mgr.positions[v0, 0],
            vtx_mgr.positions[v0, 1],
            vtx_mgr.positions[v0, 2],
            vtx_mgr.displacements[v0, 0],
            vtx_mgr.displacements[v0, 1],
            vtx_mgr.displacements[v0, 2],
            csys.hp_position[plane_id, 0],
            csys.hp_position[plane_id, 1],
            csys.hp_position[plane_id, 2],
            csys.hp_normal[plane_id, 0],
            csys.hp_normal[plane_id, 1],
            csys.hp_normal[plane_id, 2],
            eta,
            qd.f64(0.0),
            result,
        )
        qd.atomic_min(csys.ccd_alpha[()], result[0])


# ---------------------------------------------------------------------------
# Contact energy reduction (for line search)
# ---------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def reduce_contact_energy_device(
    csys: qd.template(),
    energy_out: qd.template(),
):
    """Sum all per-pair energies into ``energy_out``. Called in LS body after energy kernels."""
    for idx in range(csys.n_active_pairs[()]):
        qd.atomic_add(energy_out[()], csys.pair_energy[idx])
