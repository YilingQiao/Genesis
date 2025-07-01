from typing import Literal

import numpy as np
import torch
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import ti_field_to_torch, DeprecationError, ALLOCATE_TENSOR_WARNING
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.states.solvers import RigidSolverState
from genesis.styles import colors, formats

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .sdf_decomp import SDF

# TODO: use_ndarray as parameter
from genesis.engine.data_container import DofsState, DofsInfo, GlobalData, vec_types, get_array_type



############################################################ ndarray version
def make_kernel_step_1(
    use_ndarray: bool = True, para_level: bool = False, 
    batch_links_info: bool = False, 
    batch_joints_info: bool = False, 
    batch_dofs_info: bool = False):

    VT = vec_types(use_ndarray)
    is_serial = para_level < gs.PARA_LEVEL.PARTIAL

    @ti.kernel
    def _kernel_forward_kinematics(
        entities_info_link_start: VT.I,
        entities_info_link_end: VT.I,
        links_info_pos: VT.V3,
        links_info_quat: VT.V4,
        links_info_parent_idx: VT.I,
        links_info_joint_start: VT.I,
        links_info_joint_end: VT.I,
        links_info_is_fixed: VT.I,
        links_info_n_dofs: VT.I,
        links_info_root_idx: VT.I,
        links_info_inertial_pos: VT.V3,
        links_info_inertial_quat: VT.V4,
        links_info_inertial_mass: VT.F,
        links_info_inertial_i: VT.M3,
        joints_info_type: VT.I,
        joints_info_q_start: VT.I,
        joints_info_dof_start: VT.I,
        joints_info_dof_end: VT.I,
        joints_info_pos: VT.V3,
        dofs_info_motion_ang: VT.V3,
        dofs_info_motion_vel: VT.V3,
        links_state_pos: VT.V3,
        links_state_quat: VT.V4,
        joints_state_xanchor: VT.V3,
        joints_state_xaxis: VT.V3,
        dofs_state_pos: VT.F,
        rigid_qpos: VT.F,
        rigid_qpos0: VT.F,
        links_state_root_COM: VT.V3,
        links_state_mass_sum: VT.F,
        links_state_i_pos: VT.V3,
        links_state_i_quat: VT.V4,
        links_state_COM: VT.V3,
        links_state_cinr_inertial: VT.M3,
        links_state_cinr_pos: VT.V3,
        links_state_cinr_quat: VT.V4,
        links_state_cinr_mass: VT.F,
        links_state_j_pos: VT.V3,
        links_state_j_quat: VT.V4,
        links_state_i_pos_shift: VT.V3,
        links_state_mass_shift: VT.F,
        links_state_cd_vel: VT.V3,
        links_state_cd_ang: VT.V3,
        links_state_vel: VT.V3,
        links_state_ang: VT.V3,
        dofs_state_vel: VT.F,
        dofs_state_cdof_ang: VT.V3,
        dofs_state_cdof_vel: VT.V3,
        dofs_state_cdofvel_ang: VT.V3,
        dofs_state_cdofvel_vel: VT.V3,
        dofs_state_cdofd_ang: VT.V3,
        dofs_state_cdofd_vel: VT.V3,
        geoms_info_link_idx: VT.I,
        geoms_info_pos: VT.V3,
        geoms_info_quat: VT.V4,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        geoms_state_verts_updated: VT.I,
    ):

        ti.loop_config(serialize=is_serial)
        for i_b in range(links_state_pos.shape[1]):
            for i_e in range(entities_info_link_start.shape[0]):
                for i_l in range(entities_info_link_start[i_e], entities_info_link_end[i_e]):
                    I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                    pos = links_info_pos[I_l]
                    quat = links_info_quat[I_l]
                    # print("1~", i_l, quat)
                    if links_info_parent_idx[I_l] != -1:
                        parent_pos = links_state_pos[links_info_parent_idx[I_l], i_b]
                        parent_quat = links_state_quat[links_info_parent_idx[I_l], i_b]
                        pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
                        quat = gu.ti_transform_quat_by_quat(quat, parent_quat)
                    for i_j in range(links_info_joint_start[I_l], links_info_joint_end[I_l]):
                        I_j = [i_j, i_b] if ti.static(batch_joints_info) else i_j
                        joint_type = joints_info_type[I_j]
                        q_start = joints_info_q_start[I_j]
                        dof_start = joints_info_dof_start[I_j]
                        I_d = [dof_start, i_b] if ti.static(batch_dofs_info) else dof_start

                        # compute axis and anchor
                        if joint_type == gs.JOINT_TYPE.FREE:
                            joints_state_xanchor[i_j, i_b] = ti.Vector(
                                [
                                    rigid_qpos[q_start, i_b],
                                    rigid_qpos[q_start + 1, i_b],
                                    rigid_qpos[q_start + 2, i_b],
                                ]
                            )
                            joints_state_xaxis[i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
                        elif joint_type == gs.JOINT_TYPE.FIXED:
                            pass
                        else:
                            axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                                axis = dofs_info_motion_ang[I_d]
                            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                                axis = dofs_info_motion_vel[I_d]

                            joints_state_xanchor[i_j, i_b] = (
                                gu.ti_transform_by_quat(joints_info_pos[I_j], quat) + pos
                            )
                            joints_state_xaxis[i_j, i_b] = gu.ti_transform_by_quat(axis, quat)

                        if joint_type == gs.JOINT_TYPE.FREE:
                            pos = ti.Vector(
                                [
                                    rigid_qpos[q_start, i_b],
                                    rigid_qpos[q_start + 1, i_b],
                                    rigid_qpos[q_start + 2, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            quat = ti.Vector(
                                [
                                    rigid_qpos[q_start + 3, i_b],
                                    rigid_qpos[q_start + 4, i_b],
                                    rigid_qpos[q_start + 5, i_b],
                                    rigid_qpos[q_start + 6, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            quat = gu.ti_normalize(quat)
                            xyz = gu.ti_quat_to_xyz(quat)
                            for i in range(3):
                                dofs_state_pos[dof_start + i, i_b] = pos[i]
                                dofs_state_pos[dof_start + 3 + i, i_b] = xyz[i]
                        elif joint_type == gs.JOINT_TYPE.FIXED:
                            pass
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            qloc = ti.Vector(
                                [
                                    rigid_qpos[q_start, i_b],
                                    rigid_qpos[q_start + 1, i_b],
                                    rigid_qpos[q_start + 2, i_b],
                                    rigid_qpos[q_start + 3, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            xyz = gu.ti_quat_to_xyz(qloc)
                            for i in range(3):
                                dofs_state_pos[dof_start + i, i_b] = xyz[i]
                            quat = gu.ti_transform_quat_by_quat(qloc, quat)
                            pos = joints_state_xanchor[i_j, i_b] - gu.ti_transform_by_quat(
                                joints_info_pos[I_j], quat
                            )
                        elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                            axis = dofs_info_motion_ang[I_d]
                            dofs_state_pos[dof_start, i_b] = (
                                rigid_qpos[q_start, i_b] - rigid_qpos0[q_start, i_b]
                            )
                            qloc = gu.ti_rotvec_to_quat(axis * dofs_state_pos[dof_start, i_b])
                            quat = gu.ti_transform_quat_by_quat(qloc, quat)
                            pos = joints_state_xanchor[i_j, i_b] - gu.ti_transform_by_quat(
                                joints_info_pos[I_j], quat
                            )
                        else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                            dofs_state_pos[dof_start, i_b] = (
                                rigid_qpos[q_start, i_b] - rigid_qpos0[q_start, i_b]
                            )
                            pos = pos + joints_state_xaxis[i_j, i_b] * dofs_state_pos[dof_start, i_b]

                    # Skip link pose update for fixed root links to allow the user for manually overwriting them
                    if not (links_info_is_fixed[I_l] and links_info_joint_end[I_l] == -1):
                        links_state_pos[i_l, i_b] = pos
                        links_state_quat[i_l, i_b] = quat

            # ti.loop_config(serialize=is_serial)
            for i_l in range(links_state_root_COM.shape[0]):
                links_state_root_COM[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                links_state_mass_sum[i_l, i_b] = 0.0

            # ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                mass = links_info_inertial_mass[I_l] + links_state_mass_shift[i_l, i_b]
                (
                    links_state_i_pos[i_l, i_b],
                    links_state_i_quat[i_l, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    links_info_inertial_pos[I_l] + links_state_i_pos_shift[i_l, i_b],
                    links_info_inertial_quat[I_l],
                    links_state_pos[i_l, i_b],
                    links_state_quat[i_l, i_b],
                )

                i_r = links_info_root_idx[I_l]
                links_state_mass_sum[i_r, i_b] = links_state_mass_sum[i_r, i_b] + mass
                # ti.atomic_add(links_state_mass_sum[i_r, i_b], mass)

                COM = mass * links_state_i_pos[i_l, i_b]
                # links_state_root_COM[i_r, i_b] = links_state_root_COM[i_r, i_b] + COM
                ti.atomic_add(links_state_root_COM[i_r, i_b], COM)

            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                i_r = links_info_root_idx[I_l]
                if i_l == i_r:
                    links_state_root_COM[i_l, i_b] = (
                        links_state_root_COM[i_l, i_b] / links_state_mass_sum[i_l, i_b]
                    )
                    # self.links_state[i_l, i_b].root_COM = (
                    #     self.links_state[i_l, i_b].root_COM / self.links_state[i_l, i_b].mass_sum
                    # )

            # ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                i_r = links_info_root_idx[I_l]
                links_state_root_COM[i_l, i_b] = links_state_root_COM[i_r, i_b]

            # ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                i_r = links_info_root_idx[I_l]
                links_state_COM[i_l, i_b] = links_state_root_COM[i_r, i_b]
                links_state_i_pos[i_l, i_b] = links_state_i_pos[i_l, i_b] - links_state_COM[i_l, i_b]

                i_inertial = links_info_inertial_i[I_l]
                i_mass = links_info_inertial_mass[I_l] + links_state_mass_shift[i_l, i_b]
                (
                    links_state_cinr_inertial[i_l, i_b],
                    links_state_cinr_pos[i_l, i_b],
                    links_state_cinr_quat[i_l, i_b],
                    links_state_cinr_mass[i_l, i_b],
                ) = gu.ti_transform_inertia_by_trans_quat(
                    i_inertial, i_mass, links_state_i_pos[i_l, i_b], links_state_i_quat[i_l, i_b]
                )

            # ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l
                if links_info_n_dofs[I_l] == 0:
                    continue
                i_p = links_info_parent_idx[I_l]

                _i_j = links_info_joint_start[I_l]
                _I_j = [_i_j, i_b] if ti.static(batch_joints_info) else _i_j
                joint_type = joints_info_type[_I_j]

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = links_state_pos[i_p, i_b]
                    p_quat = links_state_quat[i_p, i_b]

                if joint_type == gs.JOINT_TYPE.FREE or (links_info_is_fixed[I_l] and i_p == -1):
                    links_state_j_pos[i_l, i_b] = links_state_pos[i_l, i_b]
                    links_state_j_quat[i_l, i_b] = links_state_quat[i_l, i_b]
                else:
                    (
                        links_state_j_pos[i_l, i_b],
                        links_state_j_quat[i_l, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        links_info_pos[I_l], links_info_quat[I_l], p_pos, p_quat
                    )

                    for i_j in range(links_info_joint_start[I_l], links_info_joint_end[I_l]):
                        I_j = [i_j, i_b] if ti.static(batch_joints_info) else i_j
                        j_info = joints_info_type[I_j]

                        (
                            links_state_j_pos[i_l, i_b],
                            links_state_j_quat[i_l, i_b],
                        ) = gu.ti_transform_pos_quat_by_trans_quat(
                            joints_info_pos[I_j],
                            gu.ti_identity_quat(),
                            links_state_j_pos[i_l, i_b],
                            links_state_j_quat[i_l, i_b],
                        )

            # cdof_fn
            # ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(links_state_root_COM.shape[0]):
                I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                for i_j in range(links_info_joint_start[I_l], links_info_joint_end[I_l]):
                    offset_pos = links_state_COM[i_l, i_b] - joints_state_xanchor[i_j, i_b]
                    I_j = [i_j, i_b] if ti.static(batch_joints_info) else i_j
                    joint_type = joints_info_type[I_j]

                    dof_start = joints_info_dof_start[I_j]

                    if joint_type == gs.JOINT_TYPE.REVOLUTE:
                        dofs_state_cdof_ang[dof_start, i_b] = joints_state_xaxis[i_j, i_b]
                        dofs_state_cdof_vel[dof_start, i_b] = joints_state_xaxis[i_j, i_b].cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                        dofs_state_cdof_ang[dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                        dofs_state_cdof_vel[dof_start, i_b] = joints_state_xaxis[i_j, i_b]
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        xmat_T = gu.ti_quat_to_R(links_state_quat[i_l, i_b]).transpose()
                        for i in range(3):
                            dofs_state_cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                            dofs_state_cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.FREE:
                        for i in range(3):
                            dofs_state_cdof_ang[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            dofs_state_cdof_vel[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            dofs_state_cdof_vel[i + dof_start + 3, i_b][i] = 1.0

                        xmat_T = gu.ti_quat_to_R(links_state_quat[i_l, i_b]).transpose()
                        for i in range(3):
                            dofs_state_cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                            dofs_state_cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                    for i_d in range(dof_start, joints_info_dof_end[I_j]):
                        dofs_state_cdofvel_ang[i_d, i_b] = (
                            dofs_state_cdof_ang[i_d, i_b] * dofs_state_vel[i_d, i_b]
                        )
                        dofs_state_cdofvel_vel[i_d, i_b] = (
                            dofs_state_cdof_vel[i_d, i_b] * dofs_state_vel[i_d, i_b]
                        )

            # forward velocity
            for i_e in range(entities_info_link_start.shape[0]):
                for i_l in range(entities_info_link_start[i_e], entities_info_link_end[i_e]):

                    I_l = [i_l, i_b] if ti.static(batch_links_info) else i_l

                    cvel_vel = ti.Vector.zero(gs.ti_float, 3)
                    cvel_ang = ti.Vector.zero(gs.ti_float, 3)
                    if links_info_parent_idx[I_l] != -1:
                        cvel_vel = links_state_cd_vel[links_info_parent_idx[I_l], i_b]
                        cvel_ang = links_state_cd_ang[links_info_parent_idx[I_l], i_b]

                    for i_j in range(links_info_joint_start[I_l], links_info_joint_end[I_l]):
                        I_j = [i_j, i_b] if ti.static(batch_joints_info) else i_j
                        joint_type = joints_info_type[I_j]
                        q_start = joints_info_q_start[I_j]
                        dof_start = joints_info_dof_start[I_j]

                        if joint_type == gs.JOINT_TYPE.FREE:
                            ## TODO: cdof_dots and cdof_ang_dot
                            for i_3 in range(3):
                                cvel_vel = (
                                    cvel_vel
                                    + dofs_state_cdof_vel[dof_start + i_3, i_b]
                                    * dofs_state_vel[dof_start + i_3, i_b]
                                )
                                cvel_ang = (
                                    cvel_ang
                                    + dofs_state_cdof_ang[dof_start + i_3, i_b]
                                    * dofs_state_vel[dof_start + i_3, i_b]
                                )

                            for i_3 in range(3):
                                (
                                    dofs_state_cdofd_ang[dof_start + i_3, i_b],
                                    dofs_state_cdofd_vel[dof_start + i_3, i_b],
                                ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                                (
                                    dofs_state_cdofd_ang[dof_start + i_3 + 3, i_b],
                                    dofs_state_cdofd_vel[dof_start + i_3 + 3, i_b],
                                ) = gu.motion_cross_motion(
                                    cvel_ang,
                                    cvel_vel,
                                    dofs_state_cdof_ang[dof_start + i_3 + 3, i_b],
                                    dofs_state_cdof_vel[dof_start + i_3 + 3, i_b],
                                )

                            for i_3 in range(3):
                                cvel_vel = (
                                    cvel_vel
                                    + dofs_state_cdof_vel[dof_start + i_3 + 3, i_b]
                                    * dofs_state_vel[dof_start + i_3 + 3, i_b]
                                )
                                cvel_ang = (
                                    cvel_ang
                                    + dofs_state_cdof_ang[dof_start + i_3 + 3, i_b]
                                    * dofs_state_vel[dof_start + i_3 + 3, i_b]
                                )

                        else:
                            for i_d in range(dof_start, joints_info_dof_end[I_j]):
                                (
                                    dofs_state_cdofd_ang[i_d, i_b],
                                    dofs_state_cdofd_vel[i_d, i_b],
                                ) = gu.motion_cross_motion(
                                    cvel_ang,
                                    cvel_vel,
                                    dofs_state_cdof_ang[i_d, i_b],
                                    dofs_state_cdof_vel[i_d, i_b],
                                )
                            for i_d in range(dof_start, joints_info_dof_end[I_j]):
                                cvel_vel = cvel_vel + dofs_state_cdof_vel[i_d, i_b] * dofs_state_vel[i_d, i_b]
                                cvel_ang = cvel_ang + dofs_state_cdof_ang[i_d, i_b] * dofs_state_vel[i_d, i_b]

                    links_state_cd_vel[i_l, i_b] = cvel_vel
                    links_state_cd_ang[i_l, i_b] = cvel_ang
                    links_state_vel[i_l, i_b] = cvel_vel
                    links_state_ang[i_l, i_b] = cvel_ang

            for i_g in range(geoms_info_link_idx.shape[0]):
                link_idx = geoms_info_link_idx[i_g]
                (
                    geoms_state_pos[i_g, i_b],
                    geoms_state_quat[i_g, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    geoms_info_pos[i_g],
                    geoms_info_quat[i_g],
                    links_state_pos[link_idx, i_b],
                    links_state_quat[link_idx, i_b],
                )

                geoms_state_verts_updated[i_g, i_b] = 0

    return _kernel_forward_kinematics
