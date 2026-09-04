"""Shared per-pair contact evaluation (``@qd.func`` device layer).

Everything one broadphase candidate produces -- distance classification, the
d_hat filter, the barrier grad/Hessian dispatch, the EE mollifier dispatch,
the Consistent-IPC per-pair scale and the per-pair energy -- lives here, with
no notion of where the result goes.

Two consumers share this layer, which is what keeps them from drifting apart:

- ``filter_assemble_device`` -- production. Keeps only the cooperative sinks:
  the batch-atomic doublet/triplet scatter and the active-pair counter.
- ``export_assemble_device`` -- the per-pair export driving cgq alignment.
  Writes the same values into candidate-indexed tables.

Mirrors the split cgq made in ``contact_device_functions.h``
(cuda-graph-qipc#354), where "production kernels keep only cooperative sinks
(atomic scatter, statistics folds, block reductions)".

``grad_12`` / ``hess_144`` are caller-allocated and must arrive **zeroed**:
the degenerate branches (pc < 4) fill only the flagged slots and rely on the
rest already being zero. Allocating them with ``qd.Vector.zero`` inside the
candidate loop, as both consumers do, satisfies this.
"""

from __future__ import annotations

import quadrants as qd

from genesis.engine.solvers.qipc.contact_function.barrier_device import (
    barrier_energy_device,
    barrier_energy_mollified_device,
    barrier_first_derivative_device,
    barrier_second_derivative_device,
)
from genesis.engine.solvers.qipc.contact_function.barrier_grad_hess_device import (
    barrier_grad_hess_ee_device,
    barrier_grad_hess_pe_device,
    barrier_grad_hess_pp_device,
    barrier_grad_hess_pt_device,
)
from genesis.engine.solvers.qipc.contact_function.barrier_grad_hess_mollified_device import (
    barrier_grad_hess_ee_mollified_device,
    barrier_grad_hess_pe_mollified_device,
    barrier_grad_hess_pp_mollified_device,
    compute_eps_x_device,
)
from genesis.engine.solvers.qipc.contact_function.distance_flag_device import (
    _get_offset,
    _popcount4,
    ee_distance_flag_cipc_device,
    ee_distance_flag_device,
    ee_flagged_distance2_device,
    ee_interior_distance2_device,
    pt_distance_flag_device,
    pt_flagged_distance2_device,
)
from genesis.engine.solvers.qipc.contact_function.halfplane_device import (
    halfplane_barrier_gradient_device,
    halfplane_barrier_hessian_device,
    halfplane_signed_distance_device,
)


@qd.func
def _ee_flag_device(
    csys: qd.template(),
    p0x: qd.f64,
    p0y: qd.f64,
    p0z: qd.f64,
    p1x: qd.f64,
    p1y: qd.f64,
    p1z: qd.f64,
    p2x: qd.f64,
    p2y: qd.f64,
    p2z: qd.f64,
    p3x: qd.f64,
    p3y: qd.f64,
    p3z: qd.f64,
) -> qd.i32:
    """Classify an EE pair with the constitution's own classifier.

    cgq dispatches this by which kernel does the calling; qipc shares one eval
    across both constitutions, so the choice is a compile-time flag instead.
    Same outcome: ConsistentIPC gets the near-parallel-guarded classifier,
    GIPC the legacy one, and the unselected branch is pruned.
    """
    ret = qd.i32(0)
    if qd.static(csys.use_cipc_distance_flag):
        ret = ee_distance_flag_cipc_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
    else:
        ret = ee_distance_flag_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
    return ret


# ---------------------------------------------------------------------------
# Consistent-IPC per-pair volume weight
# ---------------------------------------------------------------------------


@qd.func
def cipc_pair_area_weight_device(wa: qd.f64, wb: qd.f64, d_hat: qd.f64) -> qd.f64:
    """Per-pair volume weight, port of cgq ``cipc_pair_area_weight``.

    Takes the *minimum* of the two primitives' area weights rather than their
    average, so the weight reflects the actual contact patch: a small primitive
    touching a large one is weighted by its own small area.
    """
    w = wa
    if wb < wa:
        w = wb
    return w * d_hat / 4.0


@qd.func
def pt_pair_scale_device(csys: qd.template(), surf_mgr: qd.template(), vert_idx: qd.i32, face_idx: qd.i32) -> qd.f64:
    """Per-pair barrier scale for a point-triangle pair.

    ``use_area_weight`` is a compile-time flag, so the unselected branch is
    pruned and the GIPC path never touches the area-weight buffers -- callers
    running that barrier need not supply them.
    """
    if qd.static(csys.use_area_weight):
        return csys.dt_sq[0] * cipc_pair_area_weight_device(
            surf_mgr.vert_area_weight[vert_idx],
            surf_mgr.face_area_weight[face_idx],
            csys.d_hat[0],
        )
    else:
        return csys.d_hat_sq[0] * csys.d_hat_sq[0]


@qd.func
def ee_pair_scale_device(csys: qd.template(), surf_mgr: qd.template(), edge_a: qd.i32, edge_b: qd.i32) -> qd.f64:
    """Per-pair barrier scale for an edge-edge pair."""
    if qd.static(csys.use_area_weight):
        return csys.dt_sq[0] * cipc_pair_area_weight_device(
            surf_mgr.edge_area_weight[edge_a],
            surf_mgr.edge_area_weight[edge_b],
            csys.d_hat[0],
        )
    else:
        return csys.d_hat_sq[0] * csys.d_hat_sq[0]


@qd.func
def ph_pair_scale_device(csys: qd.template(), surf_mgr: qd.template(), vert_idx: qd.i32) -> qd.f64:
    """Per-pair barrier scale for a point-halfplane pair.

    **Not** the PT/EE formula. cgq's ``halfplane_filter_assemble_kernel`` says
    it outright: "a point-vs-rigid-plane pair carries the point's own area
    weight * d_hat (no /4, no averaging -- the analytical plane contributes no
    area), scaled by dt^2".

    So there is no ``min(wa, wb)`` -- the plane has no area to be the smaller of
    -- and no division by 4, which in the two-primitive case accounts for the
    pair sharing one contact patch between them.
    """
    if qd.static(csys.use_area_weight):
        return csys.dt_sq[0] * surf_mgr.vert_area_weight[vert_idx] * csys.d_hat[0]
    else:
        return csys.d_hat_sq[0] * csys.d_hat_sq[0]


# ---------------------------------------------------------------------------
# EE mollifier trigger
# ---------------------------------------------------------------------------


@qd.func
def ee_mollifier_state_device(
    csys: qd.template(),
    vtx_mgr: qd.template(),
    verts: qd.template(),
    v0: qd.i32,
    v1: qd.i32,
    v2: qd.i32,
    v3: qd.i32,
    out: qd.template(),
):
    """Write ``out[0] = I1 = ||ea x eb||^2`` and ``out[1] = eps_x``.

    cgq recomputes both inside each mollified variant, from that variant's
    permuted vertex order; one evaluation here serves the trigger and all three
    branches instead. ``I1`` is unaffected: every permutation crosses the same
    two edge vectors, and negating an operand only negates the result, so the
    squared norm is bit-identical. ``eps_x`` can differ by an ulp when a
    permutation swaps which edge multiplies first, which perturbs a mollifier
    weight by ~1e-16 -- see ``test_trigger_is_permutation_invariant``.
    """
    ax = verts[1, 0] - verts[0, 0]
    ay = verts[1, 1] - verts[0, 1]
    az = verts[1, 2] - verts[0, 2]
    bx = verts[3, 0] - verts[2, 0]
    by = verts[3, 1] - verts[2, 1]
    bz = verts[3, 2] - verts[2, 2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    out[0] = cx * cx + cy * cy + cz * cz
    out[1] = compute_eps_x_device(
        csys.eps_x_coeff[0],
        vtx_mgr.x_bar[v0, 0],
        vtx_mgr.x_bar[v0, 1],
        vtx_mgr.x_bar[v0, 2],
        vtx_mgr.x_bar[v1, 0],
        vtx_mgr.x_bar[v1, 1],
        vtx_mgr.x_bar[v1, 2],
        vtx_mgr.x_bar[v2, 0],
        vtx_mgr.x_bar[v2, 1],
        vtx_mgr.x_bar[v2, 2],
        vtx_mgr.x_bar[v3, 0],
        vtx_mgr.x_bar[v3, 1],
        vtx_mgr.x_bar[v3, 2],
    )


# ---------------------------------------------------------------------------
# EE-specific PE vertex reorder (ports cgq flag_active_offsets_for_ee)
# ---------------------------------------------------------------------------


@qd.func
def ee_pe_offsets_device(
    flag: qd.i32,
    off0: qd.template(),
    off1: qd.template(),
    off2: qd.template(),
):
    """EE PE vertex reorder: returns offsets via output templates.

    When EE degenerates to PE (pc=3), the inactive bit determines
    which edge contributes the point and which is the edge.
    Ports cgq ``flag_active_offsets_for_ee``.
    """
    inactive = qd.i32(0)
    for i in qd.static(range(4)):
        if not (flag & (1 << i)):
            inactive = i

    if inactive < 2:
        pt_idx = 1 - inactive
        off0[0] = pt_idx
        off1[0] = qd.i32(2)
        off2[0] = qd.i32(3)
    else:
        pt_idx = 5 - inactive
        off0[0] = pt_idx
        off1[0] = qd.i32(0)
        off2[0] = qd.i32(1)


# ---------------------------------------------------------------------------
# Mollified EE dispatch
# ---------------------------------------------------------------------------


@qd.func
def _ee_mollified_dispatch_device(
    csys: qd.template(),
    vtx_mgr: qd.template(),
    verts: qd.template(),
    gids: qd.template(),
    flag: qd.i32,
    pc: qd.i32,
    d2: qd.f64,
    grad_out: qd.template(),
    hess_out: qd.template(),
    sperm: qd.template(),
) -> qd.i32:
    """Assemble a near-parallel EE pair through the mollified barrier.

    Returns 1 when the pair was mollified (and ``grad_out``/``hess_out``/``sperm``
    were written), 0 when it falls outside the mollified regime and the caller
    should use the ordinary barrier. Ports the ``I1 < eps_x`` arm of cgq
    ``cipc_filter_assemble_ee_kernel``.

    Each sub-case emits a dense 12-vector over its own vertex permutation, so
    the candidate *slot* each emitted column belongs to is handed back through
    ``sperm``. Slots rather than global vertex ids: the scatter needs the ids
    (``gids[sperm[k]]``) while the per-pair export needs the slots, to put the
    columns back in candidate-stencil order. cgq's export twin keeps the same
    quantity, under the name ``slots``.
    """
    moll = qd.Vector.zero(qd.f64, 2)
    ee_mollifier_state_device(csys, vtx_mgr, verts, gids[0], gids[1], gids[2], gids[3], moll)
    I1 = moll[0]
    eps_x = moll[1]

    ret = qd.i32(0)
    if I1 < eps_x:
        ret = 1

        d_hat_sq = csys.d_hat_sq[0]
        kappa = csys.kappa[0]
        gass_t = csys.gass_t[0]
        gass_ln = csys.gass_ln[0]
        gass_a = csys.gass_a[0]

        if pc == 4:
            # Interior-interior: keep the interior EE distance while it is well
            # behaved, and fall back to the flagged distance when the degenerate
            # projection leaves the filtered band (the two agree for genuinely
            # interior closest points). The energy path makes the identical
            # substitution.
            d2_ee = ee_interior_distance2_device(
                verts[0, 0],
                verts[0, 1],
                verts[0, 2],
                verts[1, 0],
                verts[1, 1],
                verts[1, 2],
                verts[2, 0],
                verts[2, 1],
                verts[2, 2],
                verts[3, 0],
                verts[3, 1],
                verts[3, 2],
            )
            if d2_ee <= 0.0:
                d2_ee = d2
            if d2_ee >= d_hat_sq:
                d2_ee = d2
            barrier_grad_hess_ee_mollified_device(
                verts[0, 0],
                verts[0, 1],
                verts[0, 2],
                verts[1, 0],
                verts[1, 1],
                verts[1, 2],
                verts[2, 0],
                verts[2, 1],
                verts[2, 2],
                verts[3, 0],
                verts[3, 1],
                verts[3, 2],
                d2_ee,
                I1,
                eps_x,
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                grad_out,
                hess_out,
            )
        elif pc == 3:
            # Point-edge subcase: the mollified variant expects
            # (P, E0, E1, P's edge partner).
            off0_buf = qd.Vector.zero(qd.i32, 1)
            off1_buf = qd.Vector.zero(qd.i32, 1)
            off2_buf = qd.Vector.zero(qd.i32, 1)
            ee_pe_offsets_device(flag, off0_buf, off1_buf, off2_buf)
            o0 = off0_buf[0]
            o1 = off1_buf[0]
            o2 = off2_buf[0]
            miss = 6 - o0 - o1 - o2
            barrier_grad_hess_pe_mollified_device(
                verts[o0, 0],
                verts[o0, 1],
                verts[o0, 2],
                verts[o1, 0],
                verts[o1, 1],
                verts[o1, 2],
                verts[o2, 0],
                verts[o2, 1],
                verts[o2, 2],
                verts[miss, 0],
                verts[miss, 1],
                verts[miss, 2],
                d2,
                I1,
                eps_x,
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                grad_out,
                hess_out,
            )
            sperm[0] = o0
            sperm[1] = o1
            sperm[2] = o2
            sperm[3] = miss
        else:
            # Point-point subcase: the mollified variant expects
            # (A-closest, B-closest, A-other, B-other).
            o0 = _get_offset(flag, 0)
            o1 = _get_offset(flag, 1)
            a_other = o0 ^ 1
            b_other = 5 - o1
            barrier_grad_hess_pp_mollified_device(
                verts[o0, 0],
                verts[o0, 1],
                verts[o0, 2],
                verts[o1, 0],
                verts[o1, 1],
                verts[o1, 2],
                verts[a_other, 0],
                verts[a_other, 1],
                verts[a_other, 2],
                verts[b_other, 0],
                verts[b_other, 1],
                verts[b_other, 2],
                d2,
                I1,
                eps_x,
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                grad_out,
                hess_out,
            )
            sperm[0] = o0
            sperm[1] = o1
            sperm[2] = a_other
            sperm[3] = b_other
    return ret


# ---------------------------------------------------------------------------
# Per-pair grad/hess evaluation
# ---------------------------------------------------------------------------


@qd.func
def eval_pair_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    vert_idx: qd.i32,
    face_idx: qd.i32,
    gids: qd.template(),
    meta: qd.template(),
    scal: qd.template(),
    grad_12: qd.template(),
    hess_144: qd.template(),
):
    """Evaluate one PT candidate: classify, filter, assemble the scaled barrier.

    Ports the body of cgq ``fused_filter_assemble_pt_contact``
    (``contact_kernels.cu``) minus the scatter.

    Outputs, all caller-allocated:
      ``gids``     (4,) i32 -- candidate stencil in global vertex ids
      ``meta``     (2,) i32 -- ``[0]`` active (passed the d_hat filter), ``[1]`` flag
      ``scal``     (1,) f64 -- ``[0]`` d2
      ``grad_12``  (12,) f64 -- scaled gradient, candidate stencil order
      ``hess_144`` (144,) f64 -- scaled Hessian, row-major, candidate stencil order

    ``grad_12`` / ``hess_144`` are written only when active, and only in the
    flagged slots, so they must arrive zeroed. ``gids``, ``meta`` and ``scal``
    are written unconditionally: classification is defined pre-filter, which is
    what lets the export tabulate inactive candidates too.
    """
    v0 = surf_mgr.surf_verts[vert_idx]
    v1 = surf_mgr.surf_triangles[face_idx, 0]
    v2 = surf_mgr.surf_triangles[face_idx, 1]
    v3 = surf_mgr.surf_triangles[face_idx, 2]

    gids[0] = v0
    gids[1] = v1
    gids[2] = v2
    gids[3] = v3

    p0x = vtx_mgr.positions[v0, 0]
    p0y = vtx_mgr.positions[v0, 1]
    p0z = vtx_mgr.positions[v0, 2]
    p1x = vtx_mgr.positions[v1, 0]
    p1y = vtx_mgr.positions[v1, 1]
    p1z = vtx_mgr.positions[v1, 2]
    p2x = vtx_mgr.positions[v2, 0]
    p2y = vtx_mgr.positions[v2, 1]
    p2z = vtx_mgr.positions[v2, 2]
    p3x = vtx_mgr.positions[v3, 0]
    p3y = vtx_mgr.positions[v3, 1]
    p3z = vtx_mgr.positions[v3, 2]

    flag = pt_distance_flag_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)

    verts = qd.Matrix.zero(qd.f64, 4, 3)
    verts[0, 0] = p0x
    verts[0, 1] = p0y
    verts[0, 2] = p0z
    verts[1, 0] = p1x
    verts[1, 1] = p1y
    verts[1, 2] = p1z
    verts[2, 0] = p2x
    verts[2, 1] = p2y
    verts[2, 2] = p2z
    verts[3, 0] = p3x
    verts[3, 1] = p3y
    verts[3, 2] = p3z

    d2 = pt_flagged_distance2_device(flag, verts)

    meta[0] = qd.i32(0)
    meta[1] = flag
    scal[0] = d2

    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]
    gass_t = csys.gass_t[0]
    gass_ln = csys.gass_ln[0]
    gass_a = csys.gass_a[0]

    if d2 > 0.0 and d2 < d_hat_sq:
        meta[0] = qd.i32(1)
        pc = _popcount4(flag)

        if pc == 4:
            barrier_grad_hess_pt_device(
                p0x,
                p0y,
                p0z,
                p1x,
                p1y,
                p1z,
                p2x,
                p2y,
                p2z,
                p3x,
                p3y,
                p3z,
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                grad_12,
                hess_144,
            )
        elif pc == 3:
            o0 = _get_offset(flag, 0)
            o1 = _get_offset(flag, 1)
            o2 = _get_offset(flag, 2)

            g9 = qd.Vector.zero(qd.f64, 9)
            h81 = qd.Vector.zero(qd.f64, 81)
            barrier_grad_hess_pe_device(
                verts[o0, 0],
                verts[o0, 1],
                verts[o0, 2],
                verts[o1, 0],
                verts[o1, 1],
                verts[o1, 2],
                verts[o2, 0],
                verts[o2, 1],
                verts[o2, 2],
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                g9,
                h81,
            )

            offs = qd.Vector.zero(qd.i32, 3)
            offs[0] = o0
            offs[1] = o1
            offs[2] = o2
            for i in range(3):
                for k in qd.static(range(3)):
                    grad_12[offs[i] * 3 + k] = g9[i * 3 + k]
                for j in range(3):
                    for ki in qd.static(range(3)):
                        for kj in qd.static(range(3)):
                            hess_144[offs[i] * 3 * 12 + ki * 12 + offs[j] * 3 + kj] = h81[
                                i * 3 * 9 + ki * 9 + j * 3 + kj
                            ]
        else:
            o0 = _get_offset(flag, 0)
            o1 = _get_offset(flag, 1)

            g6 = qd.Vector.zero(qd.f64, 6)
            h36 = qd.Vector.zero(qd.f64, 36)
            barrier_grad_hess_pp_device(
                verts[o0, 0],
                verts[o0, 1],
                verts[o0, 2],
                verts[o1, 0],
                verts[o1, 1],
                verts[o1, 2],
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                g6,
                h36,
            )

            offs = qd.Vector.zero(qd.i32, 2)
            offs[0] = o0
            offs[1] = o1
            for i in range(2):
                for k in qd.static(range(3)):
                    grad_12[offs[i] * 3 + k] = g6[i * 3 + k]
                for j in range(2):
                    for ki in qd.static(range(3)):
                        for kj in qd.static(range(3)):
                            hess_144[offs[i] * 3 * 12 + ki * 12 + offs[j] * 3 + kj] = h36[
                                i * 3 * 6 + ki * 6 + j * 3 + kj
                            ]

        scale = pt_pair_scale_device(csys, surf_mgr, vert_idx, face_idx)
        for gi in range(12):
            grad_12[gi] = grad_12[gi] * scale
        for hi in range(144):
            hess_144[hi] = hess_144[hi] * scale


@qd.func
def eval_pair_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    edge_a: qd.i32,
    edge_b: qd.i32,
    gids: qd.template(),
    meta: qd.template(),
    scal: qd.template(),
    sperm: qd.template(),
    grad_12: qd.template(),
    hess_144: qd.template(),
):
    """Evaluate one EE candidate: classify, filter, assemble the scaled barrier.

    Ports the body of cgq ``cipc_filter_assemble_ee_kernel``
    (``consistent_ipc_contact.cu``) minus the scatter.

    Outputs, all caller-allocated:
      ``gids``     (4,) i32 -- candidate stencil in global vertex ids
      ``meta``     (3,) i32 -- ``[0]`` active, ``[1]`` flag, ``[2]`` scatter flag
      ``scal``     (1,) f64 -- ``[0]`` d2
      ``sperm``    (4,) i32 -- emitted column -> candidate slot
      ``grad_12``  (12,) f64 -- scaled gradient, in ``sperm`` column order
      ``hess_144`` (144,) f64 -- scaled Hessian, in ``sperm`` column order

    The scatter flag differs from the classification flag: pc=4 and every
    mollified sub-case emit a dense 12-vector, so they scatter as 0xF (cgq
    convention) while ``flag`` keeps the geometric classification. Only the
    degenerate unmollified branch scatters a subset.

    ``sperm`` is the identity except in the mollified pc=3 / pc=2 branches,
    which evaluate over a permuted vertex order. A consumer that wants columns
    in candidate-stencil order must un-permute through it; the production
    scatter instead carries the permuted ids as ``gids[sperm[k]]``.
    """
    v0 = surf_mgr.surf_edges[edge_a, 0]
    v1 = surf_mgr.surf_edges[edge_a, 1]
    v2 = surf_mgr.surf_edges[edge_b, 0]
    v3 = surf_mgr.surf_edges[edge_b, 1]

    gids[0] = v0
    gids[1] = v1
    gids[2] = v2
    gids[3] = v3
    for i in qd.static(range(4)):
        sperm[i] = i

    p0x = vtx_mgr.positions[v0, 0]
    p0y = vtx_mgr.positions[v0, 1]
    p0z = vtx_mgr.positions[v0, 2]
    p1x = vtx_mgr.positions[v1, 0]
    p1y = vtx_mgr.positions[v1, 1]
    p1z = vtx_mgr.positions[v1, 2]
    p2x = vtx_mgr.positions[v2, 0]
    p2y = vtx_mgr.positions[v2, 1]
    p2z = vtx_mgr.positions[v2, 2]
    p3x = vtx_mgr.positions[v3, 0]
    p3y = vtx_mgr.positions[v3, 1]
    p3z = vtx_mgr.positions[v3, 2]

    flag = _ee_flag_device(csys, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)

    verts = qd.Matrix.zero(qd.f64, 4, 3)
    verts[0, 0] = p0x
    verts[0, 1] = p0y
    verts[0, 2] = p0z
    verts[1, 0] = p1x
    verts[1, 1] = p1y
    verts[1, 2] = p1z
    verts[2, 0] = p2x
    verts[2, 1] = p2y
    verts[2, 2] = p2z
    verts[3, 0] = p3x
    verts[3, 1] = p3y
    verts[3, 2] = p3z

    d2 = ee_flagged_distance2_device(flag, verts)

    meta[0] = qd.i32(0)
    meta[1] = flag
    meta[2] = qd.i32(0xF)
    scal[0] = d2

    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]
    gass_t = csys.gass_t[0]
    gass_ln = csys.gass_ln[0]
    gass_a = csys.gass_a[0]

    if d2 > 0.0 and d2 < d_hat_sq:
        meta[0] = qd.i32(1)
        pc = _popcount4(flag)

        # `use_mollifier` is compile-time, so for the GIPC constitution the call
        # below -- and with it the only read of the rest positions -- is pruned,
        # and callers running that barrier need not supply them.
        mollified = qd.i32(0)
        if qd.static(csys.use_mollifier):
            mollified = _ee_mollified_dispatch_device(
                csys, vtx_mgr, verts, gids, flag, pc, d2, grad_12, hess_144, sperm
            )

        if mollified == 0 and pc == 4:
            barrier_grad_hess_ee_device(
                p0x,
                p0y,
                p0z,
                p1x,
                p1y,
                p1z,
                p2x,
                p2y,
                p2z,
                p3x,
                p3y,
                p3z,
                d_hat_sq,
                kappa,
                gass_t,
                gass_ln,
                gass_a,
                grad_12,
                hess_144,
            )
        elif mollified == 0:
            meta[2] = flag
            if pc == 3:
                off0_buf = qd.Vector.zero(qd.i32, 1)
                off1_buf = qd.Vector.zero(qd.i32, 1)
                off2_buf = qd.Vector.zero(qd.i32, 1)
                ee_pe_offsets_device(flag, off0_buf, off1_buf, off2_buf)
                o0 = off0_buf[0]
                o1 = off1_buf[0]
                o2 = off2_buf[0]

                g9 = qd.Vector.zero(qd.f64, 9)
                h81 = qd.Vector.zero(qd.f64, 81)
                barrier_grad_hess_pe_device(
                    verts[o0, 0],
                    verts[o0, 1],
                    verts[o0, 2],
                    verts[o1, 0],
                    verts[o1, 1],
                    verts[o1, 2],
                    verts[o2, 0],
                    verts[o2, 1],
                    verts[o2, 2],
                    d_hat_sq,
                    kappa,
                    gass_t,
                    gass_ln,
                    gass_a,
                    g9,
                    h81,
                )

                offs = qd.Vector.zero(qd.i32, 3)
                offs[0] = o0
                offs[1] = o1
                offs[2] = o2
                for i in range(3):
                    for k in qd.static(range(3)):
                        grad_12[offs[i] * 3 + k] = g9[i * 3 + k]
                    for j in range(3):
                        for ki in qd.static(range(3)):
                            for kj in qd.static(range(3)):
                                hess_144[offs[i] * 3 * 12 + ki * 12 + offs[j] * 3 + kj] = h81[
                                    i * 3 * 9 + ki * 9 + j * 3 + kj
                                ]
            else:
                o0 = _get_offset(flag, 0)
                o1 = _get_offset(flag, 1)

                g6 = qd.Vector.zero(qd.f64, 6)
                h36 = qd.Vector.zero(qd.f64, 36)
                barrier_grad_hess_pp_device(
                    verts[o0, 0],
                    verts[o0, 1],
                    verts[o0, 2],
                    verts[o1, 0],
                    verts[o1, 1],
                    verts[o1, 2],
                    d_hat_sq,
                    kappa,
                    gass_t,
                    gass_ln,
                    gass_a,
                    g6,
                    h36,
                )

                offs = qd.Vector.zero(qd.i32, 2)
                offs[0] = o0
                offs[1] = o1
                for i in range(2):
                    for k in qd.static(range(3)):
                        grad_12[offs[i] * 3 + k] = g6[i * 3 + k]
                    for j in range(2):
                        for ki in qd.static(range(3)):
                            for kj in qd.static(range(3)):
                                hess_144[offs[i] * 3 * 12 + ki * 12 + offs[j] * 3 + kj] = h36[
                                    i * 3 * 6 + ki * 6 + j * 3 + kj
                                ]

        scale = ee_pair_scale_device(csys, surf_mgr, edge_a, edge_b)
        for gi in range(12):
            grad_12[gi] = grad_12[gi] * scale
        for hi in range(144):
            hess_144[hi] = hess_144[hi] * scale


# ---------------------------------------------------------------------------
# Per-pair energy evaluation
# ---------------------------------------------------------------------------


@qd.func
def eval_pair_energy_pt_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    vert_idx: qd.i32,
    face_idx: qd.i32,
    meta: qd.template(),
    scal: qd.template(),
):
    """Evaluate one PT candidate's barrier energy.

    Ports the body of cgq ``filter_energy_pt_contact`` minus the atomic slot
    reservation. Separate from ``eval_pair_pt_device`` because the production
    assembly kernel does not need the energy and must not pay for it -- cgq
    splits the same way.

    Outputs: ``meta`` (2,) i32 ``[active, emit]``, ``scal`` (1,) f64 ``[energy]``.
    ``emit`` is what the caller counts; PT has no case where an active pair is
    dropped, so it always equals ``active``. It exists for signature symmetry
    with the EE twin, which does drop pairs.
    """
    v0 = surf_mgr.surf_verts[vert_idx]
    v1 = surf_mgr.surf_triangles[face_idx, 0]
    v2 = surf_mgr.surf_triangles[face_idx, 1]
    v3 = surf_mgr.surf_triangles[face_idx, 2]

    p0x = vtx_mgr.positions[v0, 0]
    p0y = vtx_mgr.positions[v0, 1]
    p0z = vtx_mgr.positions[v0, 2]
    p1x = vtx_mgr.positions[v1, 0]
    p1y = vtx_mgr.positions[v1, 1]
    p1z = vtx_mgr.positions[v1, 2]
    p2x = vtx_mgr.positions[v2, 0]
    p2y = vtx_mgr.positions[v2, 1]
    p2z = vtx_mgr.positions[v2, 2]
    p3x = vtx_mgr.positions[v3, 0]
    p3y = vtx_mgr.positions[v3, 1]
    p3z = vtx_mgr.positions[v3, 2]

    flag = pt_distance_flag_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)

    verts = qd.Matrix.zero(qd.f64, 4, 3)
    verts[0, 0] = p0x
    verts[0, 1] = p0y
    verts[0, 2] = p0z
    verts[1, 0] = p1x
    verts[1, 1] = p1y
    verts[1, 2] = p1z
    verts[2, 0] = p2x
    verts[2, 1] = p2y
    verts[2, 2] = p2z
    verts[3, 0] = p3x
    verts[3, 1] = p3y
    verts[3, 2] = p3z

    d2 = pt_flagged_distance2_device(flag, verts)
    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]

    meta[0] = qd.i32(0)
    meta[1] = qd.i32(0)
    scal[0] = 0.0

    if d2 > 0.0 and d2 < d_hat_sq:
        meta[0] = qd.i32(1)
        meta[1] = qd.i32(1)
        # Same per-pair scale as the grad/hess pass: the line search compares
        # this energy against the gradient it was differentiated from, so a
        # mismatched weight would break descent.
        scale = pt_pair_scale_device(csys, surf_mgr, vert_idx, face_idx)
        scal[0] = scale * barrier_energy_device(d2, d_hat_sq, kappa)


@qd.func
def eval_pair_energy_ee_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    edge_a: qd.i32,
    edge_b: qd.i32,
    meta: qd.template(),
    scal: qd.template(),
):
    """Evaluate one EE candidate's barrier energy.

    Ports the body of cgq ``cipc_filter_energy_ee_kernel`` minus the atomic slot
    reservation. The mollifier dispatch mirrors the assembly pass exactly --
    same trigger, same distance substitution -- so the line-search energy stays
    consistent with the system that was assembled and continuous through
    near-parallel configurations.

    Outputs: ``meta`` (2,) i32 ``[active, emit]``, ``scal`` (1,) f64 ``[energy]``.
    ``active`` and ``emit`` differ for the unmollified interior-interior pair
    whose interior projection is non-positive: cgq assembles its gradient but
    contributes no energy, so it is active and not emitted.
    """
    v0 = surf_mgr.surf_edges[edge_a, 0]
    v1 = surf_mgr.surf_edges[edge_a, 1]
    v2 = surf_mgr.surf_edges[edge_b, 0]
    v3 = surf_mgr.surf_edges[edge_b, 1]

    p0x = vtx_mgr.positions[v0, 0]
    p0y = vtx_mgr.positions[v0, 1]
    p0z = vtx_mgr.positions[v0, 2]
    p1x = vtx_mgr.positions[v1, 0]
    p1y = vtx_mgr.positions[v1, 1]
    p1z = vtx_mgr.positions[v1, 2]
    p2x = vtx_mgr.positions[v2, 0]
    p2y = vtx_mgr.positions[v2, 1]
    p2z = vtx_mgr.positions[v2, 2]
    p3x = vtx_mgr.positions[v3, 0]
    p3y = vtx_mgr.positions[v3, 1]
    p3z = vtx_mgr.positions[v3, 2]

    flag = _ee_flag_device(csys, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)

    verts = qd.Matrix.zero(qd.f64, 4, 3)
    verts[0, 0] = p0x
    verts[0, 1] = p0y
    verts[0, 2] = p0z
    verts[1, 0] = p1x
    verts[1, 1] = p1y
    verts[1, 2] = p1z
    verts[2, 0] = p2x
    verts[2, 1] = p2y
    verts[2, 2] = p2z
    verts[3, 0] = p3x
    verts[3, 1] = p3y
    verts[3, 2] = p3z

    d2 = ee_flagged_distance2_device(flag, verts)
    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]

    meta[0] = qd.i32(0)
    meta[1] = qd.i32(0)
    scal[0] = 0.0

    if d2 > 0.0 and d2 < d_hat_sq:
        meta[0] = qd.i32(1)
        pc = _popcount4(flag)

        # With the mollifier compiled out (GIPC constitution) this stays zero, so
        # the trigger below is false and every pair takes the ordinary barrier --
        # the same effect as pruning the branch, without a second copy of it.
        moll = qd.Vector.zero(qd.f64, 2)
        if qd.static(csys.use_mollifier):
            ee_mollifier_state_device(csys, vtx_mgr, verts, v0, v1, v2, v3, moll)
        I1 = moll[0]
        eps_x = moll[1]

        emit = qd.i32(1)
        use_moll = qd.i32(0)
        d_used = d2
        if I1 < eps_x:
            use_moll = 1
            if pc == 4:
                d2_ee = ee_interior_distance2_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
                if d2_ee > 0.0 and d2_ee < d_hat_sq:
                    d_used = d2_ee
        elif pc == 4:
            # Unmollified interior-interior takes the interior distance as given;
            # a non-positive projection means the pair is degenerate and cgq
            # drops it rather than substituting.
            d2_ee = ee_interior_distance2_device(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
            if d2_ee <= 0.0:
                emit = 0
            else:
                d_used = d2_ee

        meta[1] = emit
        if emit == 1:
            scale = ee_pair_scale_device(csys, surf_mgr, edge_a, edge_b)
            if use_moll == 1:
                scal[0] = scale * barrier_energy_mollified_device(d_used, d_hat_sq, kappa, I1, eps_x)
            else:
                scal[0] = scale * barrier_energy_device(d_used, d_hat_sq, kappa)


# ---------------------------------------------------------------------------
# Point vs half-plane
# ---------------------------------------------------------------------------


@qd.func
def eval_pair_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    sv_idx: qd.i32,
    plane_id: qd.i32,
    gids: qd.template(),
    meta: qd.template(),
    scal: qd.template(),
    grad_3: qd.template(),
    hess_9: qd.template(),
):
    """Evaluate one PH candidate: signed distance, filter, scaled barrier.

    Ports the body of cgq ``halfplane_filter_assemble_kernel``
    (``halfplane_contact_kernels.cu``) minus the scatter.

    Structurally simpler than PT and EE: a plane is analytic, so there is no
    second primitive to classify against and hence no distance flag, no
    degenerate sub-cases and no permutation. The whole contribution lands on the
    one real vertex -- a 3-vector and a 3x3 block -- which is why this takes
    ``grad_3`` / ``hess_9`` rather than the 12/144 the other channels use.

    Outputs, all caller-allocated:
      ``gids``   (2,) i32 -- ``[global vid, hp_off + plane_id]``, matching cgq's
                 stencil, whose tail slot is the plane's virtual vertex id
      ``meta``   (1,) i32 -- ``[0]`` active (passed the d_hat filter)
      ``scal``   (1,) f64 -- ``[0]`` d2 (the *squared* signed distance)
      ``grad_3`` (3,) f64, ``hess_9`` (9,) f64 -- scaled, written only when active

    ``gids`` and ``scal`` are written unconditionally, as in the other channels,
    so an inactive candidate still carries a meaningful stencil and distance.
    """
    v0 = surf_mgr.surf_verts[sv_idx]
    gids[0] = v0
    gids[1] = csys.hp_off + plane_id

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
    d_sq = d * d

    meta[0] = qd.i32(0)
    scal[0] = d_sq

    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]

    # The filter is on d^2, so a vertex that has passed through the plane looks
    # exactly like one in front of it. cgq traps that case in its query with an
    # invariant on d > xi rather than filtering it here; qipc's query does the
    # same, so by this point d is positive.
    if d_sq > 0.0 and d_sq < d_hat_sq:
        meta[0] = qd.i32(1)

        g_b = barrier_first_derivative_device(d_sq, d_hat_sq, kappa)
        H_b = barrier_second_derivative_device(d_sq, d_hat_sq, kappa)

        nx = csys.hp_normal[plane_id, 0]
        ny = csys.hp_normal[plane_id, 1]
        nz = csys.hp_normal[plane_id, 2]
        halfplane_barrier_gradient_device(g_b, d, nx, ny, nz, grad_3)
        halfplane_barrier_hessian_device(H_b, g_b, d_sq, nx, ny, nz, hess_9)

        scale = ph_pair_scale_device(csys, surf_mgr, sv_idx)
        for gi in qd.static(range(3)):
            grad_3[gi] = grad_3[gi] * scale
        for hi in range(9):
            hess_9[hi] = hess_9[hi] * scale


@qd.func
def eval_pair_energy_ph_device(
    csys: qd.template(),
    surf_mgr: qd.template(),
    vtx_mgr: qd.template(),
    sv_idx: qd.i32,
    plane_id: qd.i32,
    meta: qd.template(),
    scal: qd.template(),
):
    """Evaluate one PH candidate's barrier energy.

    Ports the body of cgq ``halfplane_filter_energy_kernel`` minus the atomic
    slot reservation. Outputs ``meta`` (2,) i32 ``[active, emit]`` and ``scal``
    (1,) f64 ``[energy]``; unlike EE there is no case where an active PH pair
    contributes no energy, so the two flags always agree.
    """
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
    d_sq = d * d

    d_hat_sq = csys.d_hat_sq[0]
    kappa = csys.kappa[0]

    meta[0] = qd.i32(0)
    meta[1] = qd.i32(0)
    scal[0] = 0.0

    if d_sq > 0.0 and d_sq < d_hat_sq:
        meta[0] = qd.i32(1)
        meta[1] = qd.i32(1)
        scale = ph_pair_scale_device(csys, surf_mgr, sv_idx)
        scal[0] = scale * barrier_energy_device(d_sq, d_hat_sq, kappa)
