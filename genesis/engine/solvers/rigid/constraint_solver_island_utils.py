"""
Core utility functions for ConstraintSolverIsland.

These functions are extracted to support GsTaichi ndarray compatibility.
When gs.use_ndarray=True, @ti.kernel/@ti.func methods in @ti.data_oriented
classes cannot access instance ndarray fields. These standalone functions
take all required arrays as arguments instead.

This module contains:
- kernel_clear, kernel_reset: Kernels for clearing and resetting solver state
- Basic @ti.func functions for constraint updates
- Cholesky factorization and solve functions
- Line search initialization and evaluation functions
"""

import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu


# =============================================================================
# Kernel clear and reset
# =============================================================================


@ti.kernel
def kernel_clear(
    _B: ti.i32,
    para_level: ti.i32,
    n_constraints: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
):
    ti.loop_config(serialize=para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        n_constraints[i_b] = 0


@ti.kernel
def kernel_reset(
    _B: ti.i32,
    n_dofs_: ti.i32,
    len_constraints_: ti.i32,
    para_level: ti.i32,
    sparse_solve: ti.template(),
    qacc_ws: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
):
    ti.loop_config(serialize=para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_d in range(n_dofs_):
            qacc_ws[i_d, i_b] = 0
            for i_c in range(len_constraints_):
                jac[i_c, i_d, i_b] = 0
        if ti.static(sparse_solve):
            for i_c in range(len_constraints_):
                jac_n_relevant_dofs[i_c, i_b] = 0


# =============================================================================
# Standalone @ti.func functions for constraint solver
# =============================================================================


@ti.func
def func_update_qacc(
    i_island: ti.i32,
    i_b: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    qacc_ws: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
):
    for i_island_entity in range(island_entity_n[i_island, i_b]):
        i_e_ = island_entity_start[i_island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            dofs_state_acc[i_d, i_b] = qacc[i_d, i_b]
            qacc_ws[i_d, i_b] = qacc[i_d, i_b]


@ti.func
def func_update_contact_force(
    i_island: ti.i32,
    i_b: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    island_col_n: ti.types.ndarray(),
    island_col_start: ti.types.ndarray(),
    constraint_id: ti.types.ndarray(),
    entities_info_link_start: ti.types.ndarray(),
    entities_info_link_end: ti.types.ndarray(),
    links_state_contact_force: ti.types.ndarray(),
    contact_data_normal: ti.types.ndarray(),
    contact_data_friction: ti.types.ndarray(),
    contact_data_force: ti.types.ndarray(),
    contact_data_link_a: ti.types.ndarray(),
    contact_data_link_b: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
):
    for i_island_entity in range(island_entity_n[i_island, i_b]):
        i_e_ = island_entity_start[i_island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_l in range(entities_info_link_start[i_e], entities_info_link_end[i_e]):
            links_state_contact_force[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)

    for i_island_col in range(island_col_n[i_island, i_b]):
        i_col_ = island_col_start[i_island, i_b] + i_island_col
        i_col = constraint_id[i_col_, i_b]

        contact_normal = contact_data_normal[i_col, i_b]
        contact_friction = contact_data_friction[i_col, i_b]

        force = ti.Vector.zero(gs.ti_float, 3)
        d1, d2 = gu.ti_orthogonals(contact_normal)
        for i in range(4):
            d = (2 * (i % 2) - 1) * (d1 if i < 2 else d2)
            n = d * contact_friction - contact_normal
            force += n * efc_force[i_island_col * 4 + i, i_b]
        contact_data_force[i_col, i_b] = force

        link_a = contact_data_link_a[i_col, i_b]
        link_b = contact_data_link_b[i_col, i_b]

        links_state_contact_force[link_a, i_b] = links_state_contact_force[link_a, i_b] - force
        links_state_contact_force[link_b, i_b] = links_state_contact_force[link_b, i_b] + force


@ti.func
def func_initialize_Jaref(
    i_b: ti.i32,
    n_constraints: ti.types.ndarray(),
    sparse_solve: ti.template(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    aref: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    n_dofs: ti.i32,
):
    for i_c in range(n_constraints[i_b]):
        Jaref_val = -aref[i_c, i_b]
        if ti.static(sparse_solve):
            for i_d_ in range(jac_n_relevant_dofs[i_c, i_b]):
                i_d = jac_relevant_dofs[i_c, i_d_, i_b]
                Jaref_val += jac[i_c, i_d, i_b] * qacc[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                Jaref_val += jac[i_c, i_d, i_b] * qacc[i_d, i_b]
        Jaref[i_c, i_b] = Jaref_val


@ti.func
def func_initialize_Ma(
    island: ti.i32,
    i_b: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
):
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d1 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            Ma_ = gs.ti_float(0.0)
            for i_d2 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                Ma_ += mass_mat[i_d1, i_d2, i_b] * qacc[i_d2, i_b]
            Ma[i_d1, i_b] = Ma_


@ti.func
def func_update_constraint(
    island: ti.i32,
    i_b: ti.i32,
    solver_type: ti.template(),
    sparse_solve: ti.template(),
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    prev_cost: ti.types.ndarray(),
    cost: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    active: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
    n_dofs: ti.i32,
):
    prev_cost[i_b] = cost[i_b]
    cost[i_b] = gs.ti_float(0.0)
    gauss[i_b] = gs.ti_float(0.0)

    for i_c in range(n_constraints[i_b]):
        if ti.static(solver_type == gs.constraint_solver.Newton):
            prev_active[i_c, i_b] = active[i_c, i_b]
        active[i_c, i_b] = Jaref[i_c, i_b] < 0
        efc_force[i_c, i_b] = -efc_D[i_c, i_b] * Jaref[i_c, i_b] * active[i_c, i_b]

    if ti.static(sparse_solve):
        for i_island_entity in range(island_entity_n[island, i_b]):
            i_e_ = island_entity_start[island, i_b] + i_island_entity
            i_e = entity_id[i_e_, i_b]
            for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                qfrc_constraint[i_d, i_b] = gs.ti_float(0.0)
        for i_c in range(n_constraints[i_b]):
            for i_d_ in range(jac_n_relevant_dofs[i_c, i_b]):
                i_d = jac_relevant_dofs[i_c, i_d_, i_b]
                qfrc_constraint[i_d, i_b] = qfrc_constraint[i_d, i_b] + jac[i_c, i_d, i_b] * efc_force[i_c, i_b]
    else:
        for i_d in range(n_dofs):
            qfrc_constraint_val = gs.ti_float(0.0)
            for i_c in range(n_constraints[i_b]):
                qfrc_constraint_val += jac[i_c, i_d, i_b] * efc_force[i_c, i_b]
            qfrc_constraint[i_d, i_b] = qfrc_constraint_val

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            v = 0.5 * (Ma[i_d, i_b] - dofs_state_force[i_d, i_b]) * (qacc[i_d, i_b] - dofs_state_acc[i_d, i_b])
            gauss[i_b] = gauss[i_b] + v
            cost[i_b] = cost[i_b] + v

    # D * (Jx - aref) ** 2
    for i_c in range(n_constraints[i_b]):
        cost[i_b] = cost[i_b] + 0.5 * (efc_D[i_c, i_b] * Jaref[i_c, i_b] * Jaref[i_c, i_b] * active[i_c, i_b])


@ti.func
def func_update_gradient(
    island: ti.i32,
    i_b: ti.i32,
    solver_type: ti.template(),
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_entities: ti.i32,
    grad: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    mass_mat_mask: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
):
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            grad[i_d, i_b] = Ma[i_d, i_b] - dofs_state_force[i_d, i_b] - qfrc_constraint[i_d, i_b]

    # Note: CG solver path requires calling func_solve_mass_batch externally
    # Newton solver path uses cholesky solve
    if ti.static(solver_type == gs.constraint_solver.Newton):
        func_nt_chol_solve(
            island,
            i_b,
            island_entity_n,
            island_entity_start,
            entity_id,
            entities_info_dof_start,
            entities_info_dof_end,
            grad,
            Mgrad,
            nt_H,
        )


@ti.func
def func_nt_chol_solve(
    island: ti.i32,
    i_b: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    grad: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
):
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            Mgrad[i_d, i_b] = grad[i_d, i_b]

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            for j_island_entity in range(i_island_entity + 1):
                j_e_ = island_entity_start[island, i_b] + j_island_entity
                j_e = entity_id[j_e_, i_b]
                for j_d in range(entities_info_dof_start[j_e], ti.min(entities_info_dof_end[j_e], i_d)):
                    Mgrad[i_d, i_b] = Mgrad[i_d, i_b] - (nt_H[i_b, i_d, j_d] * Mgrad[j_d, i_b])
            Mgrad[i_d, i_b] = Mgrad[i_d, i_b] / nt_H[i_b, i_d, i_d]

    for i_island_entity_ in range(island_entity_n[island, i_b]):
        i_island_entity = island_entity_n[island, i_b] - 1 - i_island_entity_
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d_ in range(entities_info_dof_end[i_e] - entities_info_dof_start[i_e]):
            i_d = entities_info_dof_end[i_e] - 1 - i_d_

            for j_island_entity in range(i_island_entity, island_entity_n[island, i_b]):
                j_e_ = island_entity_start[island, i_b] + j_island_entity
                j_e = entity_id[j_e_, i_b]
                for j_d in range(ti.max(i_d + 1, entities_info_dof_start[j_e]), entities_info_dof_end[j_e]):
                    Mgrad[i_d, i_b] = Mgrad[i_d, i_b] - nt_H[i_b, j_d, i_d] * Mgrad[j_d, i_b]

            Mgrad[i_d, i_b] = Mgrad[i_d, i_b] / nt_H[i_b, i_d, i_d]


@ti.func
def func_nt_chol_factor(
    island: ti.i32,
    i_b: ti.i32,
    n_dofs: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
):
    rank = n_dofs

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            tmp = nt_H[i_b, i_d, i_d]

            for j_island_entity in range(i_island_entity + 1):
                j_e_ = island_entity_start[island, i_b] + j_island_entity
                j_e = entity_id[j_e_, i_b]
                for j_d in range(entities_info_dof_start[j_e], ti.min(entities_info_dof_end[j_e], i_d)):
                    tmp = tmp - (nt_H[i_b, i_d, j_d] * nt_H[i_b, i_d, j_d])

            mindiag = 1e-8
            if tmp < mindiag:
                tmp = mindiag
                rank = rank - 1
            nt_H[i_b, i_d, i_d] = ti.sqrt(tmp)

            tmp = 1 / nt_H[i_b, i_d, i_d]

            for j_island_entity in range(i_island_entity, island_entity_n[island, i_b]):
                j_e_ = island_entity_start[island, i_b] + j_island_entity
                j_e = entity_id[j_e_, i_b]
                for j_d in range(ti.max(i_d + 1, entities_info_dof_start[j_e]), entities_info_dof_end[j_e]):
                    dot = gs.ti_float(0.0)

                    for k_island_entity in range(i_island_entity + 1):
                        k_e_ = island_entity_start[island, i_b] + k_island_entity
                        k_e = entity_id[k_e_, i_b]
                        for k_d in range(entities_info_dof_start[k_e], ti.min(entities_info_dof_end[k_e], i_d)):
                            dot = dot + nt_H[i_b, j_d, k_d] * nt_H[i_b, i_d, k_d]

                    nt_H[i_b, j_d, i_d] = (nt_H[i_b, j_d, i_d] - dot) * tmp


@ti.func
def func_nt_hessian_direct(
    island: ti.i32,
    i_b: ti.i32,
    n_dofs: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    active: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
):
    # H = M + J'*D*J
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d1 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            for i_island_entity2 in range(island_entity_n[island, i_b]):
                i_e2_ = island_entity_start[island, i_b] + i_island_entity2
                i_e2 = entity_id[i_e2_, i_b]
                for i_d2 in range(entities_info_dof_start[i_e2], entities_info_dof_end[i_e2]):
                    nt_H[i_b, i_d1, i_d2] = gs.ti_float(0.0)

    for i_c in range(n_constraints[i_b]):
        jac_n_relevant_dofs_val = jac_n_relevant_dofs[i_c, i_b]
        for i_d1_ in range(jac_n_relevant_dofs_val):
            i_d1 = jac_relevant_dofs[i_c, jac_n_relevant_dofs_val - 1 - i_d1_, i_b]
            if ti.abs(jac[i_c, i_d1, i_b]) > gs.EPS:
                for i_d2_ in range(i_d1_ + 1):
                    i_d2 = jac_relevant_dofs[i_c, jac_n_relevant_dofs_val - 1 - i_d2_, i_b]

                    d1 = ti.max(i_d1, i_d2)
                    d2 = ti.min(i_d1, i_d2)

                    nt_H[i_b, d1, d2] = (
                        nt_H[i_b, d1, d2] + jac[i_c, d2, i_b] * jac[i_c, d1, i_b] * efc_D[i_c, i_b] * active[i_c, i_b]
                    )

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d1 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            for i_island_entity2 in range(island_entity_n[island, i_b]):
                i_e2_ = island_entity_start[island, i_b] + i_island_entity2
                i_e2 = entity_id[i_e2_, i_b]
                for i_d2 in range(entities_info_dof_start[i_e2], entities_info_dof_end[i_e2]):
                    if i_d1 < i_d2:
                        nt_H[i_b, i_d1, i_d2] = nt_H[i_b, i_d2, i_d1]

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d1 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            for i_d2 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                nt_H[i_b, i_d1, i_d2] = nt_H[i_b, i_d1, i_d2] + mass_mat[i_d1, i_d2, i_b]

    func_nt_chol_factor(
        island,
        i_b,
        n_dofs,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        nt_H,
    )


@ti.func
def func_ls_init(
    island: ti.i32,
    i_b: ti.i32,
    sparse_solve: ti.template(),
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    search: ti.types.ndarray(),
    mv: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    quad: ti.types.ndarray(),
):
    # mv and jv
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d1 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            mv_val = gs.ti_float(0.0)
            for i_d2 in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                mv_val += mass_mat[i_d1, i_d2, i_b] * search[i_d2, i_b]
            mv[i_d1, i_b] = mv_val

    for i_c in range(n_constraints[i_b]):
        jv_val = gs.ti_float(0.0)
        if ti.static(sparse_solve):
            for i_d_ in range(jac_n_relevant_dofs[i_c, i_b]):
                i_d = jac_relevant_dofs[i_c, i_d_, i_b]
                jv_val += jac[i_c, i_d, i_b] * search[i_d, i_b]
        else:
            for i_island_entity in range(island_entity_n[island, i_b]):
                i_e_ = island_entity_start[island, i_b] + i_island_entity
                i_e = entity_id[i_e_, i_b]
                for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                    jv_val += jac[i_c, i_d, i_b] * search[i_d, i_b]
        jv[i_c, i_b] = jv_val

    # quad and quad_gauss
    quad_gauss_1 = gs.ti_float(0.0)
    quad_gauss_2 = gs.ti_float(0.0)

    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            quad_gauss_1 += search[i_d, i_b] * Ma[i_d, i_b] - search[i_d, i_b] * dofs_state_force[i_d, i_b]
            quad_gauss_2 += 0.5 * search[i_d, i_b] * mv[i_d, i_b]

    quad_gauss[0, i_b] = gauss[i_b]
    quad_gauss[1, i_b] = quad_gauss_1
    quad_gauss[2, i_b] = quad_gauss_2

    for i_c in range(n_constraints[i_b]):
        quad[i_c, 0, i_b] = efc_D[i_c, i_b] * (0.5 * Jaref[i_c, i_b] * Jaref[i_c, i_b])
        quad[i_c, 1, i_b] = efc_D[i_c, i_b] * (jv[i_c, i_b] * Jaref[i_c, i_b])
        quad[i_c, 2, i_b] = efc_D[i_c, i_b] * (0.5 * jv[i_c, i_b] * jv[i_c, i_b])


@ti.func
def func_ls_point_fn(
    i_b: ti.i32,
    alpha: gs.ti_float,
    n_constraints: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    quad: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
):
    tmp_quad_total0 = quad_gauss[0, i_b]
    tmp_quad_total1 = quad_gauss[1, i_b]
    tmp_quad_total2 = quad_gauss[2, i_b]

    for i_c in range(n_constraints[i_b]):
        is_active = Jaref[i_c, i_b] + alpha * jv[i_c, i_b] < 0
        tmp_quad_total0 += quad[i_c, 0, i_b] * is_active
        tmp_quad_total1 += quad[i_c, 1, i_b] * is_active
        tmp_quad_total2 += quad[i_c, 2, i_b] * is_active

    cost = alpha * alpha * tmp_quad_total2 + alpha * tmp_quad_total1 + tmp_quad_total0
    deriv_0 = 2 * alpha * tmp_quad_total2 + tmp_quad_total1
    deriv_1 = 2 * tmp_quad_total2 + gs.EPS * (ti.abs(tmp_quad_total2) < gs.EPS)

    ls_it[i_b] = ls_it[i_b] + 1

    return alpha, cost, deriv_0, deriv_1


@ti.func
def func_update_bracket(
    p_alpha: gs.ti_float,
    p_cost: gs.ti_float,
    p_deriv_0: gs.ti_float,
    p_deriv_1: gs.ti_float,
    i_b: ti.i32,
    n_constraints: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    quad: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
    candidates: ti.types.ndarray(),
):
    flag = 0
    out_alpha = p_alpha
    out_cost = p_cost
    out_deriv_0 = p_deriv_0
    out_deriv_1 = p_deriv_1

    for i in range(3):
        if p_deriv_0 < 0 and candidates[4 * i + 2, i_b] < 0 and p_deriv_0 < candidates[4 * i + 2, i_b]:
            out_alpha = candidates[4 * i + 0, i_b]
            out_cost = candidates[4 * i + 1, i_b]
            out_deriv_0 = candidates[4 * i + 2, i_b]
            out_deriv_1 = candidates[4 * i + 3, i_b]
            flag = 1
        elif p_deriv_0 > 0 and candidates[4 * i + 2, i_b] > 0 and p_deriv_0 > candidates[4 * i + 2, i_b]:
            out_alpha = candidates[4 * i + 0, i_b]
            out_cost = candidates[4 * i + 1, i_b]
            out_deriv_0 = candidates[4 * i + 2, i_b]
            out_deriv_1 = candidates[4 * i + 3, i_b]
            flag = 2

    p_next_alpha = out_alpha
    p_next_cost = out_cost
    p_next_deriv_0 = out_deriv_0
    p_next_deriv_1 = out_deriv_1

    if flag > 0:
        p_next_alpha, p_next_cost, p_next_deriv_0, p_next_deriv_1 = func_ls_point_fn(
            i_b, out_alpha - out_deriv_0 / out_deriv_1, n_constraints, quad_gauss, Jaref, jv, quad, ls_it
        )

    return (
        flag,
        out_alpha,
        out_cost,
        out_deriv_0,
        out_deriv_1,
        p_next_alpha,
        p_next_cost,
        p_next_deriv_0,
        p_next_deriv_1,
    )
