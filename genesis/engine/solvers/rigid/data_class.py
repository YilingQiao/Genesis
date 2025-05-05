from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import taichi as ti

import genesis as gs


def vec_types(is_ndarray: bool):
    def choose(dtype):
        return ti.types.ndarray(dtype=dtype) if is_ndarray else ti.template()

    ns = SimpleNamespace(
        V2=choose(gs.ti_vec2),
        V3=choose(gs.ti_vec3),
        V7=choose(gs.ti_vec7),
        F=choose(gs.ti_float),
        I=choose(gs.ti_int),
        IV3=choose(gs.ti_ivec3),
    )

    return ns


@ti.data_oriented
class DataClass:
    def __init__(self, is_ndarray: bool):
        self.is_ndarray = is_ndarray

        if is_ndarray:
            self.VT = ti.ndarray
        else:
            self.VT = ti.field


@ti.data_oriented
class GlobalData(DataClass):
    def __init__(self, is_ndarray: bool, n_dofs: int, n_entities: int, f_batch: Callable):
        super().__init__(is_ndarray)

        self.mass_mat = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs, n_dofs)))
        self.mass_mat_L = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs, n_dofs)))
        self.mass_mat_D_inv = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs,)))

        self.mass_mat_mask = self.VT(dtype=gs.ti_int, shape=f_batch(n_entities))
        self.mass_parent_mask = self.VT(dtype=gs.ti_float, shape=(n_dofs, n_dofs))

        self.meaninertia = self.VT(dtype=gs.ti_float, shape=f_batch())


@ti.data_oriented
class VertsInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_verts: int):
        super().__init__(is_ndarray)

        self.init_pos = self.VT(dtype=gs.ti_vec3, shape=(n_verts,))
        self.init_normal = self.VT(dtype=gs.ti_vec3, shape=(n_verts,))
        self.geom_idx = self.VT(dtype=gs.ti_int, shape=(n_verts,))
        self.init_center_pos = self.VT(dtype=gs.ti_vec3, shape=(n_verts,))
        self.verts_state_idx = self.VT(dtype=gs.ti_int, shape=(n_verts,))
        self.is_free = self.VT(dtype=gs.ti_int, shape=(n_verts,))


@ti.data_oriented
class FacesInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_faces: int):
        super().__init__(is_ndarray)

        self.verts_idx = self.VT(dtype=gs.ti_ivec3, shape=(n_faces,))
        self.geom_idx = self.VT(dtype=gs.ti_int, shape=(n_faces,))


@ti.data_oriented
class VvertsInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_vverts: int):
        super().__init__(is_ndarray)

        self.init_pos = self.VT(dtype=gs.ti_vec3, shape=(n_vverts,))
        self.init_vnormal = self.VT(dtype=gs.ti_vec3, shape=(n_vverts,))
        self.vgeom_idx = self.VT(dtype=gs.ti_int, shape=(n_vverts,))


@ti.data_oriented
class VfacesInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_vfaces: int):
        super().__init__(is_ndarray)

        self.vverts_idx = self.VT(dtype=gs.ti_ivec3, shape=(n_vfaces,))
        self.vgeom_idx = self.VT(dtype=gs.ti_int, shape=(n_vfaces,))


# struct_vvert_info = ti.types.struct(
#     init_pos=gs.ti_vec3,
#     init_vnormal=gs.ti_vec3,
#     vgeom_idx=gs.ti_int,
# )
# struct_vface_info = ti.types.struct(
#     vverts_idx=gs.ti_ivec3,
#     vgeom_idx=gs.ti_int,
# )


@ti.data_oriented
class VertsState(DataClass):
    def __init__(self, is_ndarray: bool, n_verts: int, f_batch: Callable):
        super().__init__(is_ndarray)

        self.pos = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_verts,)))


@ti.data_oriented
class EdgesInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_edges: int):
        super().__init__(is_ndarray)

        self.v0 = self.VT(dtype=gs.ti_int, shape=(n_edges,))
        self.v1 = self.VT(dtype=gs.ti_int, shape=(n_edges,))
        self.length = self.VT(dtype=gs.ti_float, shape=(n_edges,))


# struct_vert_state = ti.types.struct(
#     pos=gs.ti_vec3,
# )

# self.verts_info = struct_vert_info.field(shape=(self.n_verts_), needs_grad=False, layout=ti.Layout.SOA)
# self.faces_info = struct_face_info.field(shape=(self.n_faces_), needs_grad=False, layout=ti.Layout.SOA)
# self.edges_info = struct_edge_info.field(shape=(self.n_edges_), needs_grad=False, layout=ti.Layout.SOA)

# if self.n_free_verts > 0:
#     self.free_verts_state = struct_vert_state.field(
#         shape=self._batch_shape(self.n_free_verts), needs_grad=False, layout=ti.Layout.SOA
#     )
# self.fixed_verts_state = struct_vert_state.field(
#     shape=(max(1, self.n_fixed_verts),), needs_grad=False, layout=ti.Layout.SOA
# )


@ti.data_oriented
class DofsInfo(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.stiffness = self.VT(dtype=gs.ti_float, shape=shape)
        self.sol_params = self.VT(dtype=gs.ti_vec7, shape=shape)
        self.invweight = self.VT(dtype=gs.ti_float, shape=shape)
        self.armature = self.VT(dtype=gs.ti_float, shape=shape)
        self.damping = self.VT(dtype=gs.ti_float, shape=shape)
        self.motion_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.motion_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.limit = self.VT(dtype=gs.ti_vec2, shape=shape)
        self.kp = self.VT(dtype=gs.ti_float, shape=shape)
        self.kv = self.VT(dtype=gs.ti_float, shape=shape)
        self.force_range = self.VT(dtype=gs.ti_vec2, shape=shape)


@ti.data_oriented
class DofsState(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        # self.force = self.VT(dtype=gs.ti_float, shape=shape)
        self.force = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_bias = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_passive = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_actuator = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_applied = self.VT(dtype=gs.ti_float, shape=shape)
        self.act_length = self.VT(dtype=gs.ti_float, shape=shape)
        self.pos = self.VT(dtype=gs.ti_float, shape=shape)
        self.vel = self.VT(dtype=gs.ti_float, shape=shape)
        self.acc = self.VT(dtype=gs.ti_float, shape=shape)
        self.acc_smooth = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_smooth = self.VT(dtype=gs.ti_float, shape=shape)
        self.qf_constraint = self.VT(dtype=gs.ti_float, shape=shape)
        self.cdof_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdof_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdofvel_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdofvel_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdofd_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdofd_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.f_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.f_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.ctrl_force = self.VT(dtype=gs.ti_float, shape=shape)
        self.ctrl_pos = self.VT(dtype=gs.ti_float, shape=shape)
        self.ctrl_vel = self.VT(dtype=gs.ti_float, shape=shape)
        self.ctrl_mode = self.VT(dtype=gs.ti_int, shape=shape)
        self.hibernated = self.VT(dtype=gs.ti_int, shape=shape)
