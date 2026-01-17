"""
Solver-specific functions for ConstraintSolverIsland.

This module contains:
- Constraint addition functions (collision, joint limits)
- Main solve loop and body functions
- Linesearch implementation
- Newton Hessian updates
- Solver initialization
"""

import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

from .rigid_solver_decomp_util import func_wakeup_entity_and_its_temp_island
from .constraint_solver_island_utils import (
    func_initialize_Jaref,
    func_initialize_Ma,
    func_update_constraint,
    func_update_gradient,
    func_nt_chol_factor,
    func_nt_hessian_direct,
    func_ls_init,
    func_ls_point_fn,
    func_update_bracket,
)


# =============================================================================
# Additional standalone functions for constraint resolution
# =============================================================================


@ti.func
def func_add_collision_constraints_and_wakeup_entities(
    i_island: ti.i32,
    i_b: ti.i32,
    use_hibernation: ti.template(),
    sparse_solve: ti.template(),
    batch_links_info: ti.template(),
    n_dofs: ti.i32,
    # Contact island arrays (flattened)
    island_col_n: ti.types.ndarray(),
    island_col_start: ti.types.ndarray(),
    constraint_id: ti.types.ndarray(),
    # Collider state arrays
    contact_data_link_a: ti.types.ndarray(),
    contact_data_link_b: ti.types.ndarray(),
    contact_data_normal: ti.types.ndarray(),
    contact_data_friction: ti.types.ndarray(),
    contact_data_pos: ti.types.ndarray(),
    contact_data_sol_params: ti.types.ndarray(),
    contact_data_penetration: ti.types.ndarray(),
    # Links info arrays
    links_info_invweight: ti.types.ndarray(),
    links_info_n_dofs: ti.types.ndarray(),
    links_info_dof_end: ti.types.ndarray(),
    links_info_parent_idx: ti.types.ndarray(),
    links_info_entity_idx: ti.types.ndarray(),
    # Links state arrays
    links_state_root_COM: ti.types.ndarray(),
    # Dofs state arrays
    dofs_state_cdof_ang: ti.types.ndarray(),
    dofs_state_cdof_vel: ti.types.ndarray(),
    dofs_state_vel: ti.types.ndarray(),
    # Entities state
    entities_state_hibernated: ti.types.ndarray(),
    # Constraint solver arrays
    n_constraints: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    diag: ti.types.ndarray(),
    aref: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    # For wakeup - use proper struct type annotations
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    contact_island_state: array_class.ContactIslandState,
):
    """Add collision constraints for an island and wake up hibernated entities if needed."""
    n_constraints[i_b] = 0

    for i_island_col in range(island_col_n[i_island, i_b]):
        i_col_ = island_col_start[i_island, i_b] + i_island_col
        i_col = constraint_id[i_col_, i_b]

        # Get links indices of the contact_data
        link_a = contact_data_link_a[i_col, i_b]
        link_b = contact_data_link_b[i_col, i_b]
        link_a_maybe_batch = [link_a, i_b] if ti.static(batch_links_info) else link_a
        link_b_maybe_batch = [link_b, i_b] if ti.static(batch_links_info) else link_b

        contact_normal = contact_data_normal[i_col, i_b]
        d1, d2 = gu.ti_orthogonals(contact_normal)

        invweight = links_info_invweight[link_a_maybe_batch][0] + links_info_invweight[link_b_maybe_batch][0] * (
            link_b > -1
        )

        for i in range(4):
            d = (2 * (i % 2) - 1) * (d1 if i < 2 else d2)
            contact_friction = contact_data_friction[i_col, i_b]
            n = d * contact_friction - contact_normal

            n_con = ti.atomic_add(n_constraints[i_b], 1)
            if ti.static(sparse_solve):
                for i_d_ in range(jac_n_relevant_dofs[n_con, i_b]):
                    i_d = jac_relevant_dofs[n_con, i_d_, i_b]
                    jac[n_con, i_d, i_b] = gs.ti_float(0.0)
            else:
                for i_d in range(n_dofs):
                    jac[n_con, i_d, i_b] = gs.ti_float(0.0)

            con_n_relevant_dofs = 0
            jac_qvel = gs.ti_float(0.0)
            for i_ab in range(2):
                sign = gs.ti_float(-1.0)
                link = link_a
                if i_ab == 1:
                    sign = gs.ti_float(1.0)
                    link = link_b

                while link > -1:
                    link_maybe_batch = [link, i_b] if ti.static(batch_links_info) else link

                    # Reverse order to make dofs in each row strictly descending
                    for i_d_ in range(links_info_n_dofs[link]):
                        i_d = links_info_dof_end[link_maybe_batch] - 1 - i_d_

                        cdof_ang = dofs_state_cdof_ang[i_d, i_b]
                        cdot_vel = dofs_state_cdof_vel[i_d, i_b]

                        t_quat = gu.ti_identity_quat()
                        contact_pos = contact_data_pos[i_col, i_b]
                        t_pos = contact_pos - links_state_root_COM[link, i_b]
                        _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdot_vel, t_pos, t_quat)

                        diff = sign * vel
                        jac_val = diff @ n
                        jac_qvel = jac_qvel + jac_val * dofs_state_vel[i_d, i_b]
                        jac[n_con, i_d, i_b] = jac[n_con, i_d, i_b] + jac_val
                        if ti.static(sparse_solve):
                            jac_relevant_dofs[n_con, con_n_relevant_dofs, i_b] = i_d
                            con_n_relevant_dofs += 1

                    link = links_info_parent_idx[link_maybe_batch]

            if ti.static(sparse_solve):
                jac_n_relevant_dofs[n_con, i_b] = con_n_relevant_dofs

            contact_sol_params = contact_data_sol_params[i_col, i_b]
            contact_penetration = contact_data_penetration[i_col, i_b]
            imp, aref_val = gu.imp_aref(contact_sol_params, -contact_penetration, jac_qvel, -contact_penetration)

            diag_val = invweight + contact_friction * contact_friction * invweight
            diag_val *= 2 * contact_friction * contact_friction * (1 - imp) / ti.max(imp, gs.EPS)

            diag[n_con, i_b] = diag_val
            aref[n_con, i_b] = aref_val
            efc_D[n_con, i_b] = 1 / ti.max(diag_val, gs.EPS)

        if ti.static(use_hibernation):
            entity_idx_a = links_info_entity_idx[link_a_maybe_batch]
            entity_idx_b = links_info_entity_idx[link_b_maybe_batch]

            is_entity_a_hibernated = entities_state_hibernated[entity_idx_a, i_b]
            is_entity_b_hibernated = entities_state_hibernated[entity_idx_b, i_b]
            if is_entity_a_hibernated or is_entity_b_hibernated:
                any_hibernated_entity_idx = entity_idx_a if is_entity_a_hibernated else entity_idx_b
                func_wakeup_entity_and_its_temp_island(
                    any_hibernated_entity_idx,
                    i_b,
                    entities_state,
                    entities_info,
                    dofs_state,
                    links_state,
                    geoms_state,
                    rigid_global_info,
                    contact_island_state,
                )


@ti.func
def func_add_joint_limit_constraints(
    i_island: ti.i32,
    i_b: ti.i32,
    sparse_solve: ti.template(),
    batch_links_info: ti.template(),
    batch_joints_info: ti.template(),
    batch_dofs_info: ti.template(),
    n_dofs: ti.i32,
    # Contact island arrays (flattened)
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    # Entities info
    entities_info_link_start: ti.types.ndarray(),
    entities_info_link_end: ti.types.ndarray(),
    # Links info
    links_info_joint_start: ti.types.ndarray(),
    links_info_joint_end: ti.types.ndarray(),
    # Joints info
    joints_info_type: ti.types.ndarray(),
    joints_info_q_start: ti.types.ndarray(),
    joints_info_dof_start: ti.types.ndarray(),
    joints_info_sol_params: ti.types.ndarray(),
    # Dofs info
    dofs_info_limit: ti.types.ndarray(),
    dofs_info_invweight: ti.types.ndarray(),
    # Solver state
    qpos: ti.types.ndarray(),
    dofs_state_vel: ti.types.ndarray(),
    # Constraint solver arrays
    n_constraints: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    diag: ti.types.ndarray(),
    aref: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
):
    """Add joint limit constraints for an island."""
    for i_island_entity in range(island_entity_n[i_island, i_b]):
        i_e_ = island_entity_start[i_island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]

        for i_l in range(entities_info_link_start[i_e], entities_info_link_end[i_e]):
            I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l
            l_info_start = links_info_joint_start[I_l]
            l_info_end = links_info_joint_end[I_l]

            for i_j in range(l_info_start, l_info_end):
                I_j = [i_j, i_b] if ti.static(batch_joints_info) else i_j

                if joints_info_type[I_j] == gs.JOINT_TYPE.REVOLUTE or joints_info_type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_q = joints_info_q_start[I_j]
                    i_d = joints_info_dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(batch_dofs_info) else i_d
                    pos_delta_min = qpos[i_q, i_b] - dofs_info_limit[I_d][0]
                    pos_delta_max = dofs_info_limit[I_d][1] - qpos[i_q, i_b]
                    pos_delta = min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        jac_val = (pos_delta_min < pos_delta_max) * 2 - 1
                        jac_qvel = jac_val * dofs_state_vel[i_d, i_b]
                        imp, aref_val = gu.imp_aref(joints_info_sol_params[I_j], pos_delta, jac_qvel, pos_delta)
                        diag_val = ti.max(dofs_info_invweight[I_d] * (1 - imp) / imp, gs.EPS)

                        n_con = n_constraints[i_b]
                        n_constraints[i_b] = n_con + 1
                        diag[n_con, i_b] = diag_val
                        aref[n_con, i_b] = aref_val
                        efc_D[n_con, i_b] = 1 / diag_val

                        if ti.static(sparse_solve):
                            for i_d2_ in range(jac_n_relevant_dofs[n_con, i_b]):
                                i_d2 = jac_relevant_dofs[n_con, i_d2_, i_b]
                                jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        else:
                            for i_d2 in range(n_dofs):
                                jac[n_con, i_d2, i_b] = gs.ti_float(0.0)
                        jac[n_con, i_d, i_b] = jac_val

                        if ti.static(sparse_solve):
                            jac_n_relevant_dofs[n_con, i_b] = 1
                            jac_relevant_dofs[n_con, 0, i_b] = i_d


@ti.func
def func_solve(
    i_island: ti.i32,
    i_b: ti.i32,
    iterations: ti.i32,
    tolerance: gs.ti_float,
    n_dofs: ti.i32,
    n_dofs_: ti.i32,
    solver_type: ti.template(),
    sparse_solve: ti.template(),
    # Contact island arrays
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    # Entities info
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    # Solver state
    meaninertia: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    improved: ti.types.ndarray(),
    grad: ti.types.ndarray(),
    prev_cost: ti.types.ndarray(),
    cost: ti.types.ndarray(),
    # Additional arrays for solve_body (passed through)
    ls_iterations: ti.i32,
    ls_tolerance: gs.ti_float,
    search: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    mv: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    quad: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    gtol: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
    ls_result: ti.types.ndarray(),
    candidates: ti.types.ndarray(),
    active: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
    # CG solver specific
    cg_prev_grad: ti.types.ndarray(),
    cg_prev_Mgrad: ti.types.ndarray(),
    cg_beta: ti.types.ndarray(),
    cg_pg_dot_pMg: ti.types.ndarray(),
    # Newton solver specific
    nt_H: ti.types.ndarray(),
    nt_vec: ti.types.ndarray(),
    n_entities: ti.i32,
    mass_mat_mask: ti.types.ndarray(),
):
    """Main constraint solving loop for an island."""
    if n_constraints[i_b] > 0:
        tol_scaled = (meaninertia[i_b] * ti.max(1, n_dofs)) * tolerance
        for it in range(iterations):
            func_solve_body(
                i_island,
                i_b,
                solver_type,
                sparse_solve,
                n_dofs,
                n_dofs_,
                ls_iterations,
                ls_tolerance,
                tolerance,
                island_entity_n,
                island_entity_start,
                entity_id,
                entities_info_dof_start,
                entities_info_dof_end,
                meaninertia,
                n_constraints,
                improved,
                search,
                Mgrad,
                qacc,
                Ma,
                mv,
                jv,
                Jaref,
                gauss,
                quad_gauss,
                quad,
                efc_D,
                jac,
                jac_n_relevant_dofs,
                jac_relevant_dofs,
                mass_mat,
                dofs_state_force,
                gtol,
                ls_it,
                ls_result,
                candidates,
                prev_cost,
                cost,
                active,
                prev_active,
                efc_force,
                qfrc_constraint,
                dofs_state_acc,
                grad,
                cg_prev_grad,
                cg_prev_Mgrad,
                cg_beta,
                cg_pg_dot_pMg,
                nt_H,
                nt_vec,
                n_entities,
                mass_mat_mask,
            )

            if not improved[i_b]:
                break

            gradient = gs.ti_float(0.0)
            for i_island_entity in range(island_entity_n[i_island, i_b]):
                i_e_ = island_entity_start[i_island, i_b] + i_island_entity
                i_e = entity_id[i_e_, i_b]
                for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                    gradient += grad[i_d, i_b] * grad[i_d, i_b]

            gradient = ti.sqrt(gradient)
            improvement = prev_cost[i_b] - cost[i_b]
            if gradient < tol_scaled or improvement < tol_scaled:
                break


@ti.func
def func_solve_body(
    island: ti.i32,
    i_b: ti.i32,
    solver_type: ti.template(),
    sparse_solve: ti.template(),
    n_dofs: ti.i32,
    n_dofs_: ti.i32,
    ls_iterations: ti.i32,
    ls_tolerance: gs.ti_float,
    tolerance: gs.ti_float,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    meaninertia: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    improved: ti.types.ndarray(),
    search: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    mv: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    quad: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    gtol: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
    ls_result: ti.types.ndarray(),
    candidates: ti.types.ndarray(),
    prev_cost: ti.types.ndarray(),
    cost: ti.types.ndarray(),
    active: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
    grad: ti.types.ndarray(),
    cg_prev_grad: ti.types.ndarray(),
    cg_prev_Mgrad: ti.types.ndarray(),
    cg_beta: ti.types.ndarray(),
    cg_pg_dot_pMg: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
    nt_vec: ti.types.ndarray(),
    n_entities: ti.i32,
    mass_mat_mask: ti.types.ndarray(),
):
    """Single iteration of the solver body - performs linesearch and updates."""
    alpha = func_linesearch(
        island,
        i_b,
        sparse_solve,
        n_dofs,
        n_dofs_,
        ls_iterations,
        ls_tolerance,
        tolerance,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        meaninertia,
        n_constraints,
        search,
        mv,
        jv,
        Ma,
        dofs_state_force,
        gauss,
        quad_gauss,
        efc_D,
        Jaref,
        quad,
        jac,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        mass_mat,
        gtol,
        ls_it,
        ls_result,
        candidates,
    )

    if ti.abs(alpha) < gs.EPS:
        improved[i_b] = False
    else:
        improved[i_b] = True
        for i_island_entity in range(island_entity_n[island, i_b]):
            i_e_ = island_entity_start[island, i_b] + i_island_entity
            i_e = entity_id[i_e_, i_b]
            for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                qacc[i_d, i_b] = qacc[i_d, i_b] + search[i_d, i_b] * alpha
                Ma[i_d, i_b] = Ma[i_d, i_b] + mv[i_d, i_b] * alpha

        for i_c in range(n_constraints[i_b]):
            Jaref[i_c, i_b] = Jaref[i_c, i_b] + jv[i_c, i_b] * alpha

        if ti.static(solver_type == gs.constraint_solver.CG):
            for i_island_entity in range(island_entity_n[island, i_b]):
                i_e_ = island_entity_start[island, i_b] + i_island_entity
                i_e = entity_id[i_e_, i_b]
                for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                    cg_prev_grad[i_d, i_b] = grad[i_d, i_b]
                    cg_prev_Mgrad[i_d, i_b] = Mgrad[i_d, i_b]

        func_update_constraint(
            island,
            i_b,
            solver_type,
            sparse_solve,
            island_entity_n,
            island_entity_start,
            entity_id,
            entities_info_dof_start,
            entities_info_dof_end,
            n_constraints,
            prev_cost,
            cost,
            gauss,
            prev_active,
            active,
            Jaref,
            efc_force,
            efc_D,
            jac_n_relevant_dofs,
            jac_relevant_dofs,
            jac,
            qfrc_constraint,
            Ma,
            dofs_state_force,
            qacc,
            dofs_state_acc,
            n_dofs,
        )

        if ti.static(solver_type == gs.constraint_solver.CG):
            func_update_gradient(
                island,
                i_b,
                solver_type,
                island_entity_n,
                island_entity_start,
                entity_id,
                entities_info_dof_start,
                entities_info_dof_end,
                n_entities,
                grad,
                Ma,
                dofs_state_force,
                qfrc_constraint,
                mass_mat_mask,
                Mgrad,
                nt_H,
            )

            cg_beta[i_b] = gs.ti_float(0.0)
            cg_pg_dot_pMg[i_b] = gs.ti_float(0.0)

            for i_island_entity in range(island_entity_n[island, i_b]):
                i_e_ = island_entity_start[island, i_b] + i_island_entity
                i_e = entity_id[i_e_, i_b]
                for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                    cg_beta[i_b] += grad[i_d, i_b] * (Mgrad[i_d, i_b] - cg_prev_Mgrad[i_d, i_b])
                    cg_pg_dot_pMg[i_b] += cg_prev_Mgrad[i_d, i_b] * cg_prev_grad[i_d, i_b]

            cg_beta[i_b] = cg_beta[i_b] / ti.max(gs.EPS, cg_pg_dot_pMg[i_b])
            cg_beta[i_b] = ti.max(0.0, cg_beta[i_b])

            for i_island_entity in range(island_entity_n[island, i_b]):
                i_e_ = island_entity_start[island, i_b] + i_island_entity
                i_e = entity_id[i_e_, i_b]
                for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                    search[i_d, i_b] = -Mgrad[i_d, i_b] + cg_beta[i_b] * search[i_d, i_b]

        if ti.static(solver_type == gs.constraint_solver.Newton):
            improvement = prev_cost[i_b] - cost[i_b]
            if improvement > 0:
                # Hessian update
                func_nt_hessian_incremental(
                    island,
                    i_b,
                    n_dofs,
                    sparse_solve,
                    island_entity_n,
                    island_entity_start,
                    entity_id,
                    entities_info_dof_start,
                    entities_info_dof_end,
                    n_constraints,
                    prev_active,
                    active,
                    jac_n_relevant_dofs,
                    jac_relevant_dofs,
                    jac,
                    efc_D,
                    nt_vec,
                    nt_H,
                    mass_mat,
                )

                func_update_gradient(
                    island,
                    i_b,
                    solver_type,
                    island_entity_n,
                    island_entity_start,
                    entity_id,
                    entities_info_dof_start,
                    entities_info_dof_end,
                    n_entities,
                    grad,
                    Ma,
                    dofs_state_force,
                    qfrc_constraint,
                    mass_mat_mask,
                    Mgrad,
                    nt_H,
                )

                for i_island_entity in range(island_entity_n[island, i_b]):
                    i_e_ = island_entity_start[island, i_b] + i_island_entity
                    i_e = entity_id[i_e_, i_b]
                    for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
                        search[i_d, i_b] = -Mgrad[i_d, i_b]


@ti.func
def func_linesearch(
    island: ti.i32,
    i_b: ti.i32,
    sparse_solve: ti.template(),
    n_dofs: ti.i32,
    n_dofs_: ti.i32,
    ls_iterations: ti.i32,
    ls_tolerance: gs.ti_float,
    tolerance: gs.ti_float,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    meaninertia: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
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
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    gtol_arr: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
    ls_result: ti.types.ndarray(),
    candidates: ti.types.ndarray(),
) -> gs.ti_float:
    """Perform linesearch for constraint solver."""
    # Adaptive linesearch tolerance
    snorm = gs.ti_float(0.0)
    for i_island_entity in range(island_entity_n[island, i_b]):
        i_e_ = island_entity_start[island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            snorm += search[i_d, i_b] ** 2
    snorm = ti.sqrt(snorm / n_dofs_) * meaninertia[i_b] * n_dofs
    gtol_arr[i_b] = tolerance * ls_tolerance * snorm
    gtol = tolerance * ls_tolerance * snorm

    ls_it[i_b] = 0
    ls_result[i_b] = 0

    res_alpha = gs.ti_float(0.0)
    done = False

    if snorm < 1e-8:
        ls_result[i_b] = 1
        res_alpha = 0.0
    else:
        scale = 1 / (meaninertia[i_b] * ti.max(1, n_dofs))
        gtol = tolerance * ls_tolerance * snorm
        slopescl = scale / snorm

        func_ls_init(
            island,
            i_b,
            sparse_solve,
            island_entity_n,
            island_entity_start,
            entity_id,
            entities_info_dof_start,
            entities_info_dof_end,
            n_constraints,
            jac_n_relevant_dofs,
            jac_relevant_dofs,
            jac,
            mass_mat,
            search,
            mv,
            jv,
            Ma,
            dofs_state_force,
            gauss,
            quad_gauss,
            efc_D,
            Jaref,
            quad,
        )

        p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1 = func_ls_point_fn(
            i_b, gs.ti_float(0.0), n_constraints, quad_gauss, Jaref, jv, quad, ls_it
        )
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn(
            i_b, p0_alpha - p0_deriv_0 / p0_deriv_1, n_constraints, quad_gauss, Jaref, jv, quad, ls_it
        )
        if p0_cost < p1_cost:
            p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p0_alpha, p0_cost, p0_deriv_0, p0_deriv_1

        if ti.abs(p1_deriv_0) < gtol:
            if ti.abs(p1_alpha) < gs.EPS:
                ls_result[i_b] = 2
            else:
                ls_result[i_b] = 0
            res_alpha = p1_alpha
        else:
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1

            while p1_deriv_0 * direction <= -gtol and ls_it[i_b] < ls_iterations:
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
                p2update = 1

                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = func_ls_point_fn(
                    i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, n_constraints, quad_gauss, Jaref, jv, quad, ls_it
                )
                if ti.abs(p1_deriv_0) < gtol:
                    res_alpha = p1_alpha
                    done = True
                    break

            if not done:
                if ls_it[i_b] >= ls_iterations:
                    ls_result[i_b] = 3
                    res_alpha = p1_alpha
                    done = True

                if not p2update and not done:
                    ls_result[i_b] = 6
                    res_alpha = p1_alpha
                    done = True

                if not done:
                    # Bracketing phase
                    p2_next_alpha = p1_alpha
                    p1_next_alpha, p1_next_cost, p1_next_deriv_0, p1_next_deriv_1 = func_ls_point_fn(
                        i_b, p1_alpha - p1_deriv_0 / p1_deriv_1, n_constraints, quad_gauss, Jaref, jv, quad, ls_it
                    )

                    while ls_it[i_b] < ls_iterations:
                        pmid_alpha, pmid_cost, pmid_deriv_0, pmid_deriv_1 = func_ls_point_fn(
                            i_b, (p1_alpha + p2_alpha) * 0.5, n_constraints, quad_gauss, Jaref, jv, quad, ls_it
                        )

                        # Store candidates
                        candidates[0, i_b] = p1_next_alpha
                        candidates[1, i_b] = p1_next_cost if p1_next_cost == p1_next_cost else 0.0
                        candidates[2, i_b] = p1_next_deriv_0
                        candidates[3, i_b] = p1_next_deriv_1
                        candidates[4, i_b] = p2_next_alpha
                        candidates[5, i_b] = p2_cost
                        candidates[6, i_b] = p2_deriv_0
                        candidates[7, i_b] = p2_deriv_1
                        candidates[8, i_b] = pmid_alpha
                        candidates[9, i_b] = pmid_cost
                        candidates[10, i_b] = pmid_deriv_0
                        candidates[11, i_b] = pmid_deriv_1

                        best_i = -1
                        best_cost = gs.ti_float(0.0)
                        for ii in range(3):
                            if ti.abs(candidates[4 * ii + 2, i_b]) < gtol and (
                                best_i < 0 or candidates[4 * ii + 1, i_b] < best_cost
                            ):
                                best_cost = candidates[4 * ii + 1, i_b]
                                best_i = ii

                        if best_i >= 0:
                            res_alpha = candidates[4 * best_i + 0, i_b]
                            done = True
                            break
                        else:
                            # Update brackets
                            (
                                b1,
                                p1_alpha,
                                p1_cost,
                                p1_deriv_0,
                                p1_deriv_1,
                                p1_next_alpha,
                                p1_next_cost,
                                p1_next_deriv_0,
                                p1_next_deriv_1,
                            ) = func_update_bracket(
                                p1_alpha,
                                p1_cost,
                                p1_deriv_0,
                                p1_deriv_1,
                                i_b,
                                n_constraints,
                                quad_gauss,
                                Jaref,
                                jv,
                                quad,
                                ls_it,
                                candidates,
                            )
                            (
                                b2,
                                p2_alpha,
                                p2_cost,
                                p2_deriv_0,
                                p2_deriv_1,
                                p2_next_alpha,
                                _,
                                _,
                                _,
                            ) = func_update_bracket(
                                p2_alpha,
                                p2_cost,
                                p2_deriv_0,
                                p2_deriv_1,
                                i_b,
                                n_constraints,
                                quad_gauss,
                                Jaref,
                                jv,
                                quad,
                                ls_it,
                                candidates,
                            )

                            if b1 == 0 and b2 == 0:
                                if pmid_cost < p0_cost:
                                    ls_result[i_b] = 0
                                else:
                                    ls_result[i_b] = 7
                                res_alpha = pmid_alpha
                                done = True
                                break

                    if not done:
                        if p1_cost <= p2_cost and p1_cost < p0_cost:
                            ls_result[i_b] = 4
                            res_alpha = p1_alpha
                        elif p2_cost <= p1_cost and p2_cost < p1_cost:
                            ls_result[i_b] = 4
                            res_alpha = p2_alpha
                        else:
                            ls_result[i_b] = 5
                            res_alpha = 0.0

    return res_alpha


@ti.func
def func_nt_hessian_incremental(
    island: ti.i32,
    i_b: ti.i32,
    n_dofs: ti.i32,
    sparse_solve: ti.template(),
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_constraints: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    active: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    nt_vec: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
):
    """Incremental Hessian update for Newton solver."""
    rank = n_dofs
    updated = False

    for i_c in range(n_constraints[i_b]):
        if not updated:
            flag_update = -1
            if prev_active[i_c, i_b] == 0 and active[i_c, i_b] == 1:
                flag_update = 1
            if prev_active[i_c, i_b] == 1 and active[i_c, i_b] == 0:
                flag_update = 0

            if ti.static(sparse_solve):
                if flag_update != -1:
                    for i_d_ in range(jac_n_relevant_dofs[i_c, i_b]):
                        i_d = jac_relevant_dofs[i_c, i_d_, i_b]
                        nt_vec[i_d, i_b] = jac[i_c, i_d, i_b] * ti.sqrt(efc_D[i_c, i_b])

                    rank = n_dofs
                    for k_ in range(jac_n_relevant_dofs[i_c, i_b]):
                        k = jac_relevant_dofs[i_c, k_, i_b]
                        Lkk = nt_H[i_b, k, k]
                        tmp = Lkk * Lkk + nt_vec[k, i_b] * nt_vec[k, i_b] * (flag_update * 2 - 1)
                        if tmp < gs.EPS:
                            tmp = gs.EPS
                            rank = rank - 1
                        r = ti.sqrt(tmp)
                        c = r / Lkk
                        cinv = 1 / c
                        s = nt_vec[k, i_b] / Lkk
                        nt_H[i_b, k, k] = r

                        for i_ in range(k_):
                            i = jac_relevant_dofs[i_c, i_, i_b]
                            nt_H[i_b, i, k] = (nt_H[i_b, i, k] + s * nt_vec[i, i_b] * (flag_update * 2 - 1)) * cinv

                        for i_ in range(k_):
                            i = jac_relevant_dofs[i_c, i_, i_b]
                            nt_vec[i, i_b] = nt_vec[i, i_b] * c - s * nt_H[i_b, i, k]

                    if rank < n_dofs:
                        func_nt_hessian_direct(
                            island,
                            i_b,
                            n_dofs,
                            island_entity_n,
                            island_entity_start,
                            entity_id,
                            entities_info_dof_start,
                            entities_info_dof_end,
                            n_constraints,
                            jac_n_relevant_dofs,
                            jac_relevant_dofs,
                            jac,
                            efc_D,
                            active,
                            mass_mat,
                            nt_H,
                        )
                        updated = True
            else:
                if flag_update != -1:
                    for i_d in range(n_dofs):
                        nt_vec[i_d, i_b] = jac[i_c, i_d, i_b] * ti.sqrt(efc_D[i_c, i_b])

                    rank = n_dofs
                    for k in range(n_dofs):
                        if ti.abs(nt_vec[k, i_b]) > gs.EPS:
                            Lkk = nt_H[i_b, k, k]
                            tmp = Lkk * Lkk + nt_vec[k, i_b] * nt_vec[k, i_b] * (flag_update * 2 - 1)
                            if tmp < gs.EPS:
                                tmp = gs.EPS
                                rank = rank - 1
                            r = ti.sqrt(tmp)
                            c = r / Lkk
                            cinv = 1 / c
                            s = nt_vec[k, i_b] / Lkk
                            nt_H[i_b, k, k] = r

                            for i in range(k + 1, n_dofs):
                                nt_H[i_b, i, k] = (nt_H[i_b, i, k] + s * nt_vec[i, i_b] * (flag_update * 2 - 1)) * cinv

                            for i in range(k + 1, n_dofs):
                                nt_vec[i, i_b] = nt_vec[i, i_b] * c - s * nt_H[i_b, i, k]

                    if rank < n_dofs:
                        func_nt_hessian_direct(
                            island,
                            i_b,
                            n_dofs,
                            island_entity_n,
                            island_entity_start,
                            entity_id,
                            entities_info_dof_start,
                            entities_info_dof_end,
                            n_constraints,
                            jac_n_relevant_dofs,
                            jac_relevant_dofs,
                            jac,
                            efc_D,
                            active,
                            mass_mat,
                            nt_H,
                        )
                        updated = True


@ti.func
def func_init_solver(
    i_island: ti.i32,
    i_b: ti.i32,
    solver_type: ti.template(),
    sparse_solve: ti.template(),
    n_dofs: ti.i32,
    island_entity_n: ti.types.ndarray(),
    island_entity_start: ti.types.ndarray(),
    entity_id: ti.types.ndarray(),
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    n_entities: ti.i32,
    n_constraints: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    aref: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    qacc_ws: ti.types.ndarray(),
    Ma_ws: ti.types.ndarray(),
    cost_ws: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    cost: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    prev_cost: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    active: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    grad: ti.types.ndarray(),
    mass_mat_mask: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    nt_H: ti.types.ndarray(),
    search: ti.types.ndarray(),
):
    """Initialize the constraint solver for an island."""
    # Warm start check
    func_initialize_Jaref(
        i_b,
        n_constraints,
        sparse_solve,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        aref,
        Jaref,
        qacc_ws,
        n_dofs,
    )
    func_initialize_Ma(
        i_island,
        i_b,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        mass_mat,
        Ma_ws,
        qacc_ws,
    )
    func_update_constraint(
        i_island,
        i_b,
        solver_type,
        sparse_solve,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        n_constraints,
        prev_cost,
        cost_ws,
        gauss,
        prev_active,
        active,
        Jaref,
        efc_force,
        efc_D,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        qfrc_constraint,
        Ma_ws,
        dofs_state_force,
        qacc_ws,
        dofs_state_acc,
        n_dofs,
    )

    func_initialize_Jaref(
        i_b,
        n_constraints,
        sparse_solve,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        aref,
        Jaref,
        dofs_state_acc,
        n_dofs,
    )
    func_initialize_Ma(
        i_island,
        i_b,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        mass_mat,
        Ma,
        dofs_state_acc,
    )
    func_update_constraint(
        i_island,
        i_b,
        solver_type,
        sparse_solve,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        n_constraints,
        prev_cost,
        cost,
        gauss,
        prev_active,
        active,
        Jaref,
        efc_force,
        efc_D,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        qfrc_constraint,
        Ma,
        dofs_state_force,
        dofs_state_acc,
        dofs_state_acc,
        n_dofs,
    )

    for i_island_entity in range(island_entity_n[i_island, i_b]):
        i_e_ = island_entity_start[i_island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            if cost_ws[i_b] < cost[i_b]:
                qacc[i_d, i_b] = qacc_ws[i_d, i_b]
                Ma[i_d, i_b] = Ma_ws[i_d, i_b]
            else:
                qacc[i_d, i_b] = dofs_state_acc[i_d, i_b]

    func_initialize_Jaref(
        i_b,
        n_constraints,
        sparse_solve,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        aref,
        Jaref,
        qacc,
        n_dofs,
    )

    func_update_constraint(
        i_island,
        i_b,
        solver_type,
        sparse_solve,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        n_constraints,
        prev_cost,
        cost,
        gauss,
        prev_active,
        active,
        Jaref,
        efc_force,
        efc_D,
        jac_n_relevant_dofs,
        jac_relevant_dofs,
        jac,
        qfrc_constraint,
        Ma,
        dofs_state_force,
        qacc,
        dofs_state_acc,
        n_dofs,
    )

    if ti.static(solver_type == gs.constraint_solver.Newton):
        func_nt_hessian_direct(
            i_island,
            i_b,
            n_dofs,
            island_entity_n,
            island_entity_start,
            entity_id,
            entities_info_dof_start,
            entities_info_dof_end,
            n_constraints,
            jac_n_relevant_dofs,
            jac_relevant_dofs,
            jac,
            efc_D,
            active,
            mass_mat,
            nt_H,
        )

    func_update_gradient(
        i_island,
        i_b,
        solver_type,
        island_entity_n,
        island_entity_start,
        entity_id,
        entities_info_dof_start,
        entities_info_dof_end,
        n_entities,
        grad,
        Ma,
        dofs_state_force,
        qfrc_constraint,
        mass_mat_mask,
        Mgrad,
        nt_H,
    )

    for i_island_entity in range(island_entity_n[i_island, i_b]):
        i_e_ = island_entity_start[i_island, i_b] + i_island_entity
        i_e = entity_id[i_e_, i_b]
        for i_d in range(entities_info_dof_start[i_e], entities_info_dof_end[i_e]):
            search[i_d, i_b] = -Mgrad[i_d, i_b]
