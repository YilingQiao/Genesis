from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import taichi as ti

import genesis as gs

_enable_self_collision = True
_enable_adjacent_collision = False
batch_links_info = False
max_collision_pairs = 218


def vec_types(is_ndarray: bool):
    def choose(dtype):
        return ti.types.ndarray(dtype=dtype) if is_ndarray else ti.template()

    ns = SimpleNamespace(
        V2=choose(gs.ti_vec2),
        V3=choose(gs.ti_vec3),
        V4=choose(gs.ti_vec4),
        V7=choose(gs.ti_vec7),
        F=choose(gs.ti_float),
        I=choose(gs.ti_int),
        IV3=choose(gs.ti_ivec3),
        M3=choose(gs.ti_mat3),
        G=ti.types.ndarray() if is_ndarray else ti.template(),
    )

    return ns

def get_array_type(is_ndarray: bool):
    return ti.ndarray if is_ndarray else ti.field

@ti.data_oriented
class DataClass:
    def __init__(self, is_ndarray: bool):
        self.is_ndarray = is_ndarray
        self.VT = get_array_type(is_ndarray)


@ti.data_oriented
class GlobalData(DataClass):
    def __init__(self, is_ndarray: bool, n_dofs: int, n_entities: int, n_geoms: int, f_batch: Callable):
        super().__init__(is_ndarray)

        self.mass_mat = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs, n_dofs)))
        self.mass_mat_L = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs, n_dofs)))
        self.mass_mat_D_inv = self.VT(dtype=gs.ti_float, shape=f_batch((n_dofs,)))
        self.mass_mat_mask = self.VT(dtype=gs.ti_int, shape=f_batch(n_entities))
        self.mass_parent_mask = self.VT(dtype=gs.ti_float, shape=(n_dofs, n_dofs))
        self.meaninertia = self.VT(dtype=gs.ti_float, shape=f_batch())
        self.geoms_init_AABB = self.VT(dtype=gs.ti_vec3, shape=(n_geoms, 8))




        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        # This buffer stores indexes of active geoms during SAP search



        self.sort_buffer_value = self.VT(dtype=gs.ti_float, shape=f_batch(2 * n_geoms))
        self.sort_buffer_i_g = self.VT(dtype=gs.ti_int, shape=f_batch(2 * n_geoms))
        self.sort_buffer_is_max = self.VT(dtype=gs.ti_int, shape=f_batch(2 * n_geoms))

        self.active_buffer = self.VT(dtype=gs.ti_int, shape=f_batch(n_geoms))
        self.n_broad_pairs = self.VT(dtype=gs.ti_int, shape=f_batch())
        self.broad_collision_pairs = self.VT(dtype=gs.ti_int, shape=f_batch((max_collision_pairs, 2)))

        self.first_time = self.VT(dtype=gs.ti_int, shape=f_batch())
    
        self.contact_cache_normal = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_geoms, n_geoms)))
        self.contact_cache_penetration = self.VT(dtype=gs.ti_float, shape=f_batch((n_geoms, n_geoms)))
        self.contact_cache_i_va_ws = self.VT(dtype=gs.ti_int, shape=f_batch((n_geoms, n_geoms)))
        

        ############## broad phase SAP ##############



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


# struct_geom_state = ti.types.struct(
#     pos=gs.ti_vec3,
#     quat=gs.ti_vec4,
#     aabb_min=gs.ti_vec3,
#     aabb_max=gs.ti_vec3,
#     verts_updated=gs.ti_int,
#     min_buffer_idx=gs.ti_int,
#     max_buffer_idx=gs.ti_int,
#     hibernated=gs.ti_int,
#     friction_ratio=gs.ti_float,
# )


@ti.data_oriented
class GeomsState(DataClass):
    def __init__(self, is_ndarray: bool, n_geoms: int, f_batch: Callable):
        super().__init__(is_ndarray)

        self.pos = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_geoms,)))
        self.quat = self.VT(dtype=gs.ti_vec4, shape=f_batch((n_geoms,)))
        self.aabb_min = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_geoms,)))
        self.aabb_max = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_geoms,)))
        self.verts_updated = self.VT(dtype=gs.ti_int, shape=f_batch((n_geoms,)))
        self.min_buffer_idx = self.VT(dtype=gs.ti_int, shape=f_batch((n_geoms,)))
        self.max_buffer_idx = self.VT(dtype=gs.ti_int, shape=f_batch((n_geoms,)))
        self.hibernated = self.VT(dtype=gs.ti_int, shape=f_batch((n_geoms,)))
        self.friction_ratio = self.VT(dtype=gs.ti_float, shape=f_batch((n_geoms,)))


@ti.data_oriented
class GeomsInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_geoms: int):
        super().__init__(is_ndarray)

        self.pos = self.VT(dtype=gs.ti_vec3, shape=(n_geoms,))
        self.center = self.VT(dtype=gs.ti_vec3, shape=(n_geoms,))
        self.quat = self.VT(dtype=gs.ti_vec4, shape=(n_geoms,))
        self.data = self.VT(dtype=gs.ti_vec7, shape=(n_geoms,))
        self.link_idx = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.type = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.friction = self.VT(dtype=gs.ti_float, shape=(n_geoms,))
        self.sol_params = self.VT(dtype=gs.ti_vec7, shape=(n_geoms,))
        self.vert_num = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.vert_start = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.vert_end = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.verts_state_start = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.verts_state_end = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.face_num = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.face_start = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.face_end = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.edge_num = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.edge_start = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.edge_end = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.is_convex = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.needs_coup = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.contype = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.conaffinity = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.coup_friction = self.VT(dtype=gs.ti_float, shape=(n_geoms,))
        self.coup_softness = self.VT(dtype=gs.ti_float, shape=(n_geoms,))
        self.coup_restitution = self.VT(dtype=gs.ti_float, shape=(n_geoms,))
        self.is_free = self.VT(dtype=gs.ti_int, shape=(n_geoms,))
        self.is_decomposed = self.VT(dtype=gs.ti_int, shape=(n_geoms,))


@ti.data_oriented
class VgeomsInfo(DataClass):
    def __init__(self, is_ndarray: bool, n_vgeoms: int):
        super().__init__(is_ndarray)

        self.pos = self.VT(dtype=gs.ti_vec3, shape=(n_vgeoms,))
        self.quat = self.VT(dtype=gs.ti_vec4, shape=(n_vgeoms,))
        self.link_idx = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vvert_num = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vvert_start = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vvert_end = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vface_num = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vface_start = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))
        self.vface_end = self.VT(dtype=gs.ti_int, shape=(n_vgeoms,))


@ti.data_oriented
class VgeomsState(DataClass):
    def __init__(self, is_ndarray: bool, n_vgeoms: int, f_batch: Callable):
        super().__init__(is_ndarray)

        self.pos = self.VT(dtype=gs.ti_vec3, shape=f_batch((n_vgeoms,)))
        self.quat = self.VT(dtype=gs.ti_vec4, shape=f_batch((n_vgeoms,)))


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


@ti.data_oriented
class LinksInfo(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.parent_idx = self.VT(dtype=gs.ti_int, shape=shape)
        self.root_idx = self.VT(dtype=gs.ti_int, shape=shape)
        self.q_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.dof_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.joint_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.q_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.dof_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.joint_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.n_dofs = self.VT(dtype=gs.ti_int, shape=shape)
        self.pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.invweight = self.VT(dtype=gs.ti_vec2, shape=shape)
        self.is_fixed = self.VT(dtype=gs.ti_int, shape=shape)
        self.inertial_pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.inertial_quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.inertial_i = self.VT(dtype=gs.ti_mat3, shape=shape)
        self.inertial_mass = self.VT(dtype=gs.ti_float, shape=shape)
        self.entity_idx = self.VT(dtype=gs.ti_int, shape=shape)


@ti.data_oriented
class JointsInfo(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.type = self.VT(dtype=gs.ti_int, shape=shape)
        self.sol_params = self.VT(dtype=gs.ti_vec7, shape=shape)
        self.q_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.dof_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.q_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.dof_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.n_dofs = self.VT(dtype=gs.ti_int, shape=shape)
        self.pos = self.VT(dtype=gs.ti_vec3, shape=shape)


@ti.data_oriented
class LinksState(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.cinr_inertial = self.VT(dtype=gs.ti_mat3, shape=shape)
        self.cinr_pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cinr_quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.cinr_mass = self.VT(dtype=gs.ti_float, shape=shape)
        self.crb_inertial = self.VT(dtype=gs.ti_mat3, shape=shape)
        self.crb_pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.crb_quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.crb_mass = self.VT(dtype=gs.ti_float, shape=shape)
        self.cdd_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cdd_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.i_pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.i_quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.j_pos = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.j_quat = self.VT(dtype=gs.ti_vec4, shape=shape)
        self.j_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.j_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cd_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cd_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.root_COM = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.mass_sum = self.VT(dtype=gs.ti_float, shape=shape)
        self.COM = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.mass_shift = self.VT(dtype=gs.ti_float, shape=shape)
        self.i_pos_shift = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cfrc_flat_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cfrc_flat_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cfrc_ext_ang = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.cfrc_ext_vel = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.contact_force = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.hibernated = self.VT(dtype=gs.ti_int, shape=shape)

@ti.data_oriented
class JointsState(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.xanchor = self.VT(dtype=gs.ti_vec3, shape=shape)
        self.xaxis = self.VT(dtype=gs.ti_vec3, shape=shape)

@ti.data_oriented
class EntityInfo(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.dof_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.dof_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.n_dofs = self.VT(dtype=gs.ti_int, shape=shape)
        self.link_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.link_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.n_links = self.VT(dtype=gs.ti_int, shape=shape)
        self.geom_start = self.VT(dtype=gs.ti_int, shape=shape)
        self.geom_end = self.VT(dtype=gs.ti_int, shape=shape)
        self.n_geoms = self.VT(dtype=gs.ti_int, shape=shape)
        self.gravity_compensation = self.VT(dtype=gs.ti_float, shape=shape)

@ti.data_oriented
class EntityState(DataClass):
    def __init__(self, is_ndarray: bool, shape: tuple):
        super().__init__(is_ndarray)

        self.hibernated = self.VT(dtype=gs.ti_int, shape=shape)
