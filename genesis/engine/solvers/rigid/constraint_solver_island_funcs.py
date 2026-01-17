"""
Standalone kernel and func definitions for ConstraintSolverIsland.

These functions are extracted to support GsTaichi ndarray compatibility.
When gs.use_ndarray=True, @ti.kernel/@ti.func methods in @ti.data_oriented
classes cannot access instance ndarray fields. These standalone functions
take all required arrays as arguments instead.

This is a comprehensive refactoring of constraint_solver_decomp_island.py
to work around GsTaichi limitations with nested struct access patterns like:
- self.contact_island.island_entity[island, i_b].n
- self.contact_island.island_col[island, i_b].start

These patterns fail with "__getitem__ cannot be called in GsTaichi-scope".
The solution is to use flattened fields (island_entity_n, island_entity_start, etc.)
and pass them as arguments to standalone kernel/func functions.

This module re-exports all functions from:
- constraint_solver_island_utils.py: Core utility functions
- constraint_solver_island_solver.py: Solver-specific functions

And contains the main kernel_resolve orchestration kernel.
"""

import gstaichi as ti
import genesis as gs
import genesis.utils.array_class as array_class

# Re-export all functions from submodules for backwards compatibility
from .constraint_solver_island_utils import (
    kernel_clear,
    kernel_reset,
    func_update_qacc,
    func_update_contact_force,
    func_initialize_Jaref,
    func_initialize_Ma,
    func_update_constraint,
    func_update_gradient,
    func_nt_chol_solve,
    func_nt_chol_factor,
    func_nt_hessian_direct,
    func_ls_init,
    func_ls_point_fn,
    func_update_bracket,
)

from .constraint_solver_island_solver import (
    func_add_collision_constraints_and_wakeup_entities,
    func_add_joint_limit_constraints,
    func_solve,
    func_solve_body,
    func_linesearch,
    func_nt_hessian_incremental,
    func_init_solver,
)

# Make all re-exported names available for "from module import *"
__all__ = [
    # From utils
    "kernel_clear",
    "kernel_reset",
    "func_update_qacc",
    "func_update_contact_force",
    "func_initialize_Jaref",
    "func_initialize_Ma",
    "func_update_constraint",
    "func_update_gradient",
    "func_nt_chol_solve",
    "func_nt_chol_factor",
    "func_nt_hessian_direct",
    "func_ls_init",
    "func_ls_point_fn",
    "func_update_bracket",
    # From solver
    "func_add_collision_constraints_and_wakeup_entities",
    "func_add_joint_limit_constraints",
    "func_solve",
    "func_solve_body",
    "func_linesearch",
    "func_nt_hessian_incremental",
    "func_init_solver",
    # Main kernel
    "kernel_resolve",
]


# =============================================================================
# Main kernel_resolve - orchestrates the full constraint resolution
# =============================================================================


@ti.kernel
def kernel_resolve(
    _B: ti.i32,
    n_dofs: ti.i32,
    n_dofs_: ti.i32,
    n_entities: ti.i32,
    iterations: ti.i32,
    ls_iterations: ti.i32,
    tolerance: gs.ti_float,
    ls_tolerance: gs.ti_float,
    solver_type: ti.template(),
    sparse_solve: ti.template(),
    use_hibernation: ti.template(),
    batch_links_info: ti.template(),
    batch_joints_info: ti.template(),
    batch_dofs_info: ti.template(),
    # Contact island arrays (flattened)
    ci_n_islands: ti.types.ndarray(),
    ci_island_hibernated: ti.types.ndarray(),
    ci_island_entity_n: ti.types.ndarray(),
    ci_island_entity_start: ti.types.ndarray(),
    ci_island_col_n: ti.types.ndarray(),
    ci_island_col_start: ti.types.ndarray(),
    ci_entity_id: ti.types.ndarray(),
    ci_constraint_id: ti.types.ndarray(),
    # Entities info
    entities_info_dof_start: ti.types.ndarray(),
    entities_info_dof_end: ti.types.ndarray(),
    entities_info_link_start: ti.types.ndarray(),
    entities_info_link_end: ti.types.ndarray(),
    # Links info
    links_info_invweight: ti.types.ndarray(),
    links_info_n_dofs: ti.types.ndarray(),
    links_info_dof_end: ti.types.ndarray(),
    links_info_parent_idx: ti.types.ndarray(),
    links_info_entity_idx: ti.types.ndarray(),
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
    # Collider state
    contact_data_link_a: ti.types.ndarray(),
    contact_data_link_b: ti.types.ndarray(),
    contact_data_normal: ti.types.ndarray(),
    contact_data_friction: ti.types.ndarray(),
    contact_data_pos: ti.types.ndarray(),
    contact_data_sol_params: ti.types.ndarray(),
    contact_data_penetration: ti.types.ndarray(),
    contact_data_force: ti.types.ndarray(),
    # Links state
    links_state_root_COM: ti.types.ndarray(),
    links_state_contact_force: ti.types.ndarray(),
    # Dofs state
    dofs_state_cdof_ang: ti.types.ndarray(),
    dofs_state_cdof_vel: ti.types.ndarray(),
    dofs_state_vel: ti.types.ndarray(),
    dofs_state_acc: ti.types.ndarray(),
    dofs_state_force: ti.types.ndarray(),
    # Entities state
    entities_state_hibernated: ti.types.ndarray(),
    # Solver state
    qpos: ti.types.ndarray(),
    meaninertia: ti.types.ndarray(),
    mass_mat: ti.types.ndarray(),
    mass_mat_mask: ti.types.ndarray(),
    # Constraint solver arrays
    n_constraints: ti.types.ndarray(),
    jac: ti.types.ndarray(),
    jac_n_relevant_dofs: ti.types.ndarray(),
    jac_relevant_dofs: ti.types.ndarray(),
    diag: ti.types.ndarray(),
    aref: ti.types.ndarray(),
    efc_D: ti.types.ndarray(),
    efc_force: ti.types.ndarray(),
    active: ti.types.ndarray(),
    prev_active: ti.types.ndarray(),
    qfrc_constraint: ti.types.ndarray(),
    qacc: ti.types.ndarray(),
    qacc_ws: ti.types.ndarray(),
    Jaref: ti.types.ndarray(),
    Ma: ti.types.ndarray(),
    Ma_ws: ti.types.ndarray(),
    grad: ti.types.ndarray(),
    Mgrad: ti.types.ndarray(),
    search: ti.types.ndarray(),
    improved: ti.types.ndarray(),
    cost: ti.types.ndarray(),
    cost_ws: ti.types.ndarray(),
    prev_cost: ti.types.ndarray(),
    gauss: ti.types.ndarray(),
    # Linesearch arrays
    mv: ti.types.ndarray(),
    jv: ti.types.ndarray(),
    quad_gauss: ti.types.ndarray(),
    quad: ti.types.ndarray(),
    gtol: ti.types.ndarray(),
    ls_it: ti.types.ndarray(),
    ls_result: ti.types.ndarray(),
    candidates: ti.types.ndarray(),
    # CG solver specific
    cg_prev_grad: ti.types.ndarray(),
    cg_prev_Mgrad: ti.types.ndarray(),
    cg_beta: ti.types.ndarray(),
    cg_pg_dot_pMg: ti.types.ndarray(),
    # Newton solver specific
    nt_H: ti.types.ndarray(),
    nt_vec: ti.types.ndarray(),
    # For wakeup - use proper struct type annotations
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    contact_island_state: array_class.ContactIslandState,
):
    """Main constraint resolution kernel."""
    for i_b in range(_B):
        for i_island in range(ci_n_islands[i_b]):
            is_active = True
            if ti.static(use_hibernation):
                is_active = not ci_island_hibernated[i_island, i_b]

            if is_active:
                # Add collision constraints
                func_add_collision_constraints_and_wakeup_entities(
                    i_island,
                    i_b,
                    use_hibernation,
                    sparse_solve,
                    batch_links_info,
                    n_dofs,
                    ci_island_col_n,
                    ci_island_col_start,
                    ci_constraint_id,
                    contact_data_link_a,
                    contact_data_link_b,
                    contact_data_normal,
                    contact_data_friction,
                    contact_data_pos,
                    contact_data_sol_params,
                    contact_data_penetration,
                    links_info_invweight,
                    links_info_n_dofs,
                    links_info_dof_end,
                    links_info_parent_idx,
                    links_info_entity_idx,
                    links_state_root_COM,
                    dofs_state_cdof_ang,
                    dofs_state_cdof_vel,
                    dofs_state_vel,
                    entities_state_hibernated,
                    n_constraints,
                    jac,
                    jac_n_relevant_dofs,
                    jac_relevant_dofs,
                    diag,
                    aref,
                    efc_D,
                    entities_state,
                    entities_info,
                    dofs_state,
                    links_state,
                    geoms_state,
                    rigid_global_info,
                    contact_island_state,
                )

                # Add joint limit constraints
                func_add_joint_limit_constraints(
                    i_island,
                    i_b,
                    sparse_solve,
                    batch_links_info,
                    batch_joints_info,
                    batch_dofs_info,
                    n_dofs,
                    ci_island_entity_n,
                    ci_island_entity_start,
                    ci_entity_id,
                    entities_info_link_start,
                    entities_info_link_end,
                    links_info_joint_start,
                    links_info_joint_end,
                    joints_info_type,
                    joints_info_q_start,
                    joints_info_dof_start,
                    joints_info_sol_params,
                    dofs_info_limit,
                    dofs_info_invweight,
                    qpos,
                    dofs_state_vel,
                    n_constraints,
                    jac,
                    jac_n_relevant_dofs,
                    jac_relevant_dofs,
                    diag,
                    aref,
                    efc_D,
                )

                # Initialize solver
                func_init_solver(
                    i_island,
                    i_b,
                    solver_type,
                    sparse_solve,
                    n_dofs,
                    ci_island_entity_n,
                    ci_island_entity_start,
                    ci_entity_id,
                    entities_info_dof_start,
                    entities_info_dof_end,
                    n_entities,
                    n_constraints,
                    jac_n_relevant_dofs,
                    jac_relevant_dofs,
                    jac,
                    aref,
                    Jaref,
                    qacc_ws,
                    Ma_ws,
                    cost_ws,
                    dofs_state_acc,
                    Ma,
                    cost,
                    qacc,
                    mass_mat,
                    prev_cost,
                    gauss,
                    prev_active,
                    active,
                    efc_force,
                    efc_D,
                    qfrc_constraint,
                    dofs_state_force,
                    grad,
                    mass_mat_mask,
                    Mgrad,
                    nt_H,
                    search,
                )

                # Solve constraints
                func_solve(
                    i_island,
                    i_b,
                    iterations,
                    tolerance,
                    n_dofs,
                    n_dofs_,
                    solver_type,
                    sparse_solve,
                    ci_island_entity_n,
                    ci_island_entity_start,
                    ci_entity_id,
                    entities_info_dof_start,
                    entities_info_dof_end,
                    meaninertia,
                    n_constraints,
                    improved,
                    grad,
                    prev_cost,
                    cost,
                    ls_iterations,
                    ls_tolerance,
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
                    active,
                    prev_active,
                    efc_force,
                    qfrc_constraint,
                    dofs_state_acc,
                    cg_prev_grad,
                    cg_prev_Mgrad,
                    cg_beta,
                    cg_pg_dot_pMg,
                    nt_H,
                    nt_vec,
                    n_entities,
                    mass_mat_mask,
                )

                # Update qacc
                func_update_qacc(
                    i_island,
                    i_b,
                    ci_island_entity_n,
                    ci_island_entity_start,
                    ci_entity_id,
                    entities_info_dof_start,
                    entities_info_dof_end,
                    qacc,
                    qacc_ws,
                    dofs_state_acc,
                )

                # Update contact force
                func_update_contact_force(
                    i_island,
                    i_b,
                    ci_island_entity_n,
                    ci_island_entity_start,
                    ci_entity_id,
                    ci_island_col_n,
                    ci_island_col_start,
                    ci_constraint_id,
                    entities_info_link_start,
                    entities_info_link_end,
                    links_state_contact_force,
                    contact_data_normal,
                    contact_data_friction,
                    contact_data_force,
                    contact_data_link_a,
                    contact_data_link_b,
                    efc_force,
                )
