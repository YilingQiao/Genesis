from .data_class import DofsState, DofsInfo, GlobalData, vec_types, get_array_type

import taichi as ti
import genesis as gs
import genesis.utils.geom as gu
from genesis.styles import colors, formats

_enable_self_collision = True
_enable_adjacent_collision = False
batch_links_info = False
max_collision_pairs = 8192


@ti.func
def _func_check_collision_valid(
    i_ga: gs.ti_int,
    i_gb: gs.ti_int,
    i_b: gs.ti_int,
    links_info_root_idx: ti.types.ndarray(),
    links_info_parent_idx: ti.types.ndarray(),
    geoms_info_link_idx: ti.types.ndarray(),
    geoms_info_contype: ti.types.ndarray(),
    geoms_info_conaffinity: ti.types.ndarray(),
    links_info_is_fixed: ti.types.ndarray(),
) -> bool:

    i_la = geoms_info_link_idx[i_ga]
    i_lb = geoms_info_link_idx[i_gb]
    I_la = [i_la, i_b] if ti.static(batch_links_info) else i_la
    I_lb = [i_lb, i_b] if ti.static(batch_links_info) else i_lb
    is_valid = True

    # geoms in the same link
    if i_la == i_lb:
        is_valid = False

    # self collision
    if ti.static(not _enable_self_collision) and links_info_root_idx[I_la] == links_info_root_idx[I_lb]:
        is_valid = False

    # adjacent links
    if ti.static(not _enable_adjacent_collision) and (
        links_info_parent_idx[I_la] == i_lb or links_info_parent_idx[I_lb] == i_la
    ):
        is_valid = False

    # contype and conaffinity
    if not (
        (geoms_info_contype[i_ga] & geoms_info_conaffinity[i_gb])
        or (geoms_info_contype[i_gb] & geoms_info_conaffinity[i_ga])
    ):
        is_valid = False

    # pair of fixed links wrt the world
    if links_info_is_fixed[I_la] and links_info_is_fixed[I_lb]:
        is_valid = False

    return is_valid


@ti.func
def _func_is_geom_aabbs_overlap(
    i_ga: gs.ti_int,
    i_gb: gs.ti_int,
    i_b: gs.ti_int,
    geoms_state_aabb_max: ti.types.ndarray(),
    geoms_state_aabb_min: ti.types.ndarray(),
) -> bool:
    return not (
        (geoms_state_aabb_max[i_ga, i_b] <= geoms_state_aabb_min[i_gb, i_b]).any()
        or (geoms_state_aabb_min[i_ga, i_b] >= geoms_state_aabb_max[i_gb, i_b]).any()
    )


def make_kernel_reset_collider(is_ndarray: bool, is_serial: bool = False):
    VT = vec_types(is_ndarray)

    @ti.kernel
    def _kernel_reset_collider(
        envs_idx: ti.types.ndarray(),
        contact_cache_i_va_ws: VT.I,
        contact_cache_penetration: VT.F,
        contact_cache_normal: VT.V3,
        geoms_info_link_idx: VT.I,
        first_time: VT.I,
    ):
        ti.loop_config(serialize=is_serial)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            first_time[i_b] = 1
            for i_ga in range(geoms_info_link_idx.shape[0]):
                for i_gb in range(geoms_info_link_idx.shape[0]):
                    contact_cache_i_va_ws[i_ga, i_gb, i_b] = -1
                    contact_cache_penetration[i_ga, i_gb, i_b] = 0.0
                    contact_cache_normal[i_ga, i_gb, i_b].fill(0.0)

    return _kernel_reset_collider


def make_kernel_update_aabbs(is_ndarray: bool, is_serial: bool = False):
    VT = vec_types(is_ndarray)

    @ti.kernel
    def _kernel_update_aabbs(
        links_info_root_idx: VT.I,
        links_info_parent_idx: VT.I,
        links_info_is_fixed: VT.I,
        geoms_info_link_idx: VT.I,
        geoms_info_pos: VT.V3,
        geoms_info_quat: VT.V4,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        geoms_state_verts_updated: VT.I,
        geoms_init_AABB: VT.V3,
        geoms_state_aabb_min: VT.V3,
        geoms_state_aabb_max: VT.V3,
        first_time: VT.I,
        sort_buffer_value: VT.F,
        sort_buffer_i_g: VT.I,
        sort_buffer_is_max: VT.I,
        geoms_state_min_buffer_idx: VT.I,
        geoms_state_max_buffer_idx: VT.I,
        n_broad_pairs: VT.I,
        active_buffer: VT.I,
        contact_cache_normal: VT.V3,
        contact_cache_penetration: VT.F,
        contact_cache_i_va_ws: VT.I,
        broad_collision_pairs: VT.I,
        geoms_info_contype: VT.I,
        geoms_info_conaffinity: VT.I,
    ):

        ti.loop_config(serialize=is_serial)
        for i_g, i_b in ti.ndrange(geoms_info_link_idx.shape[0], geoms_state_pos.shape[1]):

            lower = gu.ti_vec3(ti.math.inf)
            upper = gu.ti_vec3(-ti.math.inf)
            for i_corner in range(8):
                corner_pos = gu.ti_transform_by_trans_quat(
                    geoms_init_AABB[i_g, i_corner], geoms_state_pos[i_g, i_b], geoms_state_quat[i_g, i_b]
                )
                lower = ti.min(lower, corner_pos)
                upper = ti.max(upper, corner_pos)

            geoms_state_aabb_min[i_g, i_b] = lower
            geoms_state_aabb_max[i_g, i_b] = upper

        ti.loop_config(serialize=is_serial)
        for i_b in range(geoms_state_pos.shape[1]):
            axis = 0
            print("i_b", i_b, axis)

            # copy updated geom aabbs to buffer for sorting
            if first_time[i_b]:
                for i in range(geoms_state_pos.shape[0]):
                    sort_buffer_value[2 * i, i_b] = geoms_state_aabb_min[i, i_b][axis]
                    sort_buffer_i_g[2 * i, i_b] = i
                    sort_buffer_is_max[2 * i, i_b] = 0

                    sort_buffer_value[2 * i + 1, i_b] = geoms_state_aabb_max[i, i_b][axis]
                    sort_buffer_i_g[2 * i + 1, i_b] = i
                    sort_buffer_is_max[2 * i + 1, i_b] = 1

                    geoms_state_min_buffer_idx[i, i_b] = 2 * i
                    geoms_state_max_buffer_idx[i, i_b] = 2 * i + 1

                first_time[i_b] = False
            else:
                for i in range(geoms_state_pos.shape[0]):
                    if sort_buffer_is_max[i, i_b]:
                        sort_buffer_value[i, i_b] = geoms_state_aabb_max[i, i_b][axis]
                    else:
                        sort_buffer_value[i, i_b] = geoms_state_aabb_min[i, i_b][axis]

            # insertion sort, which has complexity near O(n) for nearly sorted array
            for i in range(1, 2 * geoms_state_pos.shape[0]):
                key_value = sort_buffer_value[i, i_b]
                key_i_g = sort_buffer_i_g[i, i_b]
                key_is_max = sort_buffer_is_max[i, i_b]

                j = i - 1
                while j >= 0 and key_value < sort_buffer_value[j, i_b]:
                    sort_buffer_value[j + 1, i_b] = sort_buffer_value[j, i_b]
                    sort_buffer_i_g[j + 1, i_b] = sort_buffer_i_g[j, i_b]
                    sort_buffer_is_max[j + 1, i_b] = sort_buffer_is_max[j, i_b]

                    j = j - 1
                sort_buffer_value[j + 1, i_b] = key_value
                sort_buffer_i_g[j + 1, i_b] = key_i_g
                sort_buffer_is_max[j + 1, i_b] = key_is_max

            # sweep over the sorted AABBs to find potential collision pairs
            n_broad_pairs[i_b] = 0
            n_active = 0
            for i in range(2 * geoms_state_pos.shape[0]):
                if not sort_buffer_is_max[i, i_b]:
                    for j in range(n_active):
                        i_ga = active_buffer[j, i_b]
                        i_gb = sort_buffer_i_g[i, i_b]
                        if i_ga > i_gb:
                            i_ga, i_gb = i_gb, i_ga

                        is_valid = _func_check_collision_valid(
                            i_ga,
                            i_gb,
                            i_b,
                            links_info_root_idx,
                            links_info_parent_idx,
                            geoms_info_link_idx,
                            geoms_info_contype,
                            geoms_info_conaffinity,
                            links_info_is_fixed,
                        )

                        if not is_valid:
                            continue
                        # --------------------- _func_check_collision_valid

                        # --------------------- _func_is_geom_aabbs_overlap
                        # if not _func_is_geom_aabbs_overlap(
                        #     i_ga,
                        #     i_gb,
                        #     i_b,
                        #     geoms_state_aabb_max,
                        #     geoms_state_aabb_min,
                        # ):
                        is_overlap = _func_is_geom_aabbs_overlap(
                            i_ga,
                            i_gb,
                            i_b,
                            geoms_state_aabb_max,
                            geoms_state_aabb_min,
                        )

                        if not is_overlap:
                            contact_cache_normal[i_ga, i_gb, i_b].fill(0.0)
                            continue
                        # --------------------- _func_is_geom_aabbs_overlap

                        if n_broad_pairs[i_b] == max_collision_pairs:
                            print(
                                f"{colors.YELLOW}[Genesis] [00:00:00] [WARNING] Ignoring collision pair to avoid "
                                f"exceeding max ({max_collision_pairs}). Please increase the value of "
                                f"RigidSolver's option 'max_collision_pairs'.{formats.RESET}"
                            )
                            break
                        broad_collision_pairs[n_broad_pairs[i_b], i_b, 0] = i_ga
                        broad_collision_pairs[n_broad_pairs[i_b], i_b, 1] = i_gb
                        n_broad_pairs[i_b] = n_broad_pairs[i_b] + 1
                        print("add", n_broad_pairs[i_b])
                    print("active", n_active)
                    active_buffer[n_active, i_b] = sort_buffer_i_g[i, i_b]
                    n_active = n_active + 1
                else:
                    i_g_to_remove = sort_buffer_i_g[i, i_b]
                    print("i_g_to_remove", i_g_to_remove)
                    for j in range(n_active):
                        if active_buffer[j, i_b] == i_g_to_remove:
                            if j < n_active - 1:
                                for k in range(j, n_active - 1):
                                    active_buffer[k, i_b] = active_buffer[k + 1, i_b]
                            n_active = n_active - 1
                            break

    return _kernel_update_aabbs


## mpr
_enable_multi_contact = True
_mc_perturbation = 1e-3
_mc_tolerance = 1e-2
_mpr_to_sdf_overlap_ratio = 0.5
_enable_mujoco_compatibility = False
_max_contact_pairs = 8192


@ti.func
def _func_compute_tolerance(
    i_ga: gs.ti_int,
    i_gb: gs.ti_int,
    i_b: gs.ti_int,
    geoms_init_AABB: ti.types.ndarray(),
):
    aabb_size_a = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
    aabb_size_b = geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]
    tolerance_abs = 0.5 * _mc_tolerance * ti.min(aabb_size_a.norm(), aabb_size_b.norm())
    return tolerance_abs


@ti.func
def _func_geom_overlap_ratio(i_ga, i_gb, i_b):
    # TODO ndarray
    return 0.0


@ti.func
def _func_rotate_frame(i_ga, contact_pos_0, qrot, i_b):
    # TODO
    pass




@ti.func
def _func_add_contact(
    i_ga,
    i_gb,
    normal,
    contact_pos,
    penetration,
    i_b,
    n_contacts: ti.types.ndarray(),
    contact_data_geom_a: ti.types.ndarray(),
    contact_data_geom_b: ti.types.ndarray(),
    contact_data_normal: ti.types.ndarray(),
    contact_data_pos: ti.types.ndarray(),
    contact_data_penetration: ti.types.ndarray(),
    contact_data_friction: ti.types.ndarray(),
    contact_data_sol_params: ti.types.ndarray(),
    contact_data_link_a: ti.types.ndarray(),
    contact_data_link_b: ti.types.ndarray(),
    geoms_info_friction: ti.types.ndarray(),
    geoms_state_friction_ratio: ti.types.ndarray(),
    geoms_info_sol_params: ti.types.ndarray(),
    geoms_info_link_idx: ti.types.ndarray(),
):

    i_col = n_contacts[i_b]

    if i_col == _max_contact_pairs:
        print(
            f"{colors.YELLOW}[Genesis] [00:00:00] [WARNING] Ignoring contact pair to avoid exceeding max "
            f"({_max_contact_pairs}). Please increase the value of RigidSolver's option "
            f"'max_collision_pairs'.{formats.RESET}"
        )
    else:
        friction_a = geoms_info_friction[i_ga] * geoms_state_friction_ratio[i_ga, i_b]
        friction_b = geoms_info_friction[i_gb] * geoms_state_friction_ratio[i_gb, i_b]

        # b to a
        contact_data_geom_a[i_col, i_b] = i_ga
        contact_data_geom_b[i_col, i_b] = i_gb
        contact_data_normal[i_col, i_b] = normal
        contact_data_pos[i_col, i_b] = contact_pos
        contact_data_penetration[i_col, i_b] = penetration
        contact_data_friction[i_col, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
        contact_data_sol_params[i_col, i_b] = 0.5 * (geoms_info_sol_params[i_ga] + geoms_info_sol_params[i_gb])
        contact_data_link_a[i_col, i_b] = geoms_info_link_idx[i_ga]
        contact_data_link_b[i_col, i_b] = geoms_info_link_idx[i_gb]

        n_contacts[i_b] = i_col + 1


@ti.func
def _func_contact_orthogonals(
    i_ga,
    i_gb,
    normal,
    i_b,
    geoms_init_AABB: ti.types.ndarray(),
    geoms_info_link_idx: ti.types.ndarray(),
    links_state_i_quat: ti.types.ndarray(),
):
    # The reference geometry is the one that will have the largest impact on the position of
    # the contact point. Basically, the smallest one between the two, which can be approximated
    # by the volume of their respective bounding box.
    size_ga = geoms_init_AABB[i_ga, 7]
    volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
    size_gb = geoms_init_AABB[i_gb, 7]
    volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
    i_g = i_ga if volume_ga < volume_gb else i_gb

    # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
    i_l = geoms_info_link_idx[i_g]
    rot = gu.ti_quat_to_R(links_state_i_quat[i_l, i_b])
    axis_idx = gs.ti_int(0)
    axis_angle_max = gs.ti_float(0.0)
    for i in ti.static(range(3)):
        axis_angle = ti.abs(rot[0, i] * normal[0] + rot[1, i] * normal[1] + rot[2, i] * normal[2])
        if axis_angle > axis_angle_max:
            axis_angle_max = axis_angle
            axis_idx = i
    axis_idx = (axis_idx + 1) % 3
    axis_0 = ti.Vector([rot[0, axis_idx], rot[1, axis_idx], rot[2, axis_idx]], dt=gs.ti_float)
    axis_0 -= normal.dot(axis_0) * normal
    axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


## only one mpr
@ti.func
def _func_mpr(
    i_ga: gs.ti_int,
    i_gb: gs.ti_int,
    i_b: gs.ti_int,
    geoms_info_type: ti.types.ndarray(),
    geoms_info_link_idx: ti.types.ndarray(),
    geoms_state_pos: ti.types.ndarray(),
    geoms_state_quat: ti.types.ndarray(),
    contact_cache_normal: ti.types.ndarray(),
    contact_data_pos: ti.types.ndarray(),
    n_contacts: ti.types.ndarray(),
    contact_data_geom_a: ti.types.ndarray(),
    contact_data_geom_b: ti.types.ndarray(),
    contact_data_normal: ti.types.ndarray(),
    contact_data_penetration: ti.types.ndarray(),
    contact_data_friction: ti.types.ndarray(),
    contact_data_sol_params: ti.types.ndarray(),
    contact_data_link_a: ti.types.ndarray(),
    contact_data_link_b: ti.types.ndarray(),
    geoms_info_friction: ti.types.ndarray(),
    geoms_state_friction_ratio: ti.types.ndarray(),
    geoms_info_sol_params: ti.types.ndarray(),
    geoms_init_AABB: ti.types.ndarray(),
    links_state_i_quat: ti.types.ndarray(),
):
    if geoms_info_type[i_ga] > geoms_info_type[i_gb]:
        i_gb, i_ga = i_ga, i_gb

    i_la = geoms_info_link_idx[i_ga]
    i_lb = geoms_info_link_idx[i_gb]

    # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
    # instabilities in the few cases where multiple contact points are actually need. Increasing the tolerance
    # criteria to get rid of redundant contact points seems to be a better option.
    multi_contact = (
        ti.static(_enable_multi_contact)
        # and not (self._solver.geoms_info[i_ga].is_decomposed and self._solver.geoms_info[i_gb].is_decomposed)
        and geoms_info_type[i_ga] != gs.GEOM_TYPE.SPHERE
        and geoms_info_type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
        and geoms_info_type[i_gb] != gs.GEOM_TYPE.SPHERE
        and geoms_info_type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
    )
    tolerance = _func_compute_tolerance(i_ga, i_gb, i_b, geoms_init_AABB)

    # Check if one geometry is partially enclosed in the other
    # TODO ndarray
    # overlap_ratio_a = _func_geom_overlap_ratio(i_ga, i_gb, i_b)
    # overlap_ratio_b = _func_geom_overlap_ratio(i_gb, i_ga, i_b)

    if geoms_info_type[i_ga] == gs.GEOM_TYPE.PLANE:
        # TODO
        # _func_plane_contact(i_ga, i_gb, multi_contact, i_b)
        pass
    else:
        is_col_0 = False
        penetration_0 = gs.ti_float(0.0)
        normal_0 = ti.Vector.zero(gs.ti_float, 3)
        contact_pos_0 = ti.Vector.zero(gs.ti_float, 3)

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        n_con = gs.ti_int(0)
        axis_0 = ti.Vector.zero(gs.ti_float, 3)
        axis_1 = ti.Vector.zero(gs.ti_float, 3)
        axis = ti.Vector.zero(gs.ti_float, 3)

        ga_pos, ga_quat = geoms_state_pos[i_ga, i_b], geoms_state_quat[i_ga, i_b]
        gb_pos, gb_quat = geoms_state_pos[i_gb, i_b], geoms_state_quat[i_gb, i_b]

        for i_detection in range(5):
            if multi_contact and is_col_0:
                # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                # otherwise it would be more sensitive to ill-conditionning.
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.ti_rotvec_to_quat(_mc_perturbation * axis)
                _func_rotate_frame(i_ga, contact_pos_0, qrot, i_b)
                _func_rotate_frame(i_gb, contact_pos_0, gu.ti_inv_quat(qrot), i_b)

            if (multi_contact and is_col_0) or (i_detection == 0):
                # MPR cannot handle collision detection for fully enclosed geometries. Falling back to SDF.
                # Note that SDF does not take into account to direction of interest. As such, it cannot be used
                # reliably for anything else than the point of deepest penetration.
                is_col, normal, penetration, contact_pos = func_mpr_contact(
                    i_ga, i_gb, i_b, contact_cache_normal[i_ga, i_gb, i_b]
                )

                # if (i_detection == 0) and overlap_ratio_a > _mpr_to_sdf_overlap_ratio:
                #     # FIXME: It is impossible to rely on `_func_contact_convex_convex_sdf` to get the contact
                #     # information because the compilation times skyrockets from 42s for `_func_contact_vertex_sdf`
                #     # to 2min51s on Apple Silicon M4 Max, which is not acceptable.
                #     # is_col, normal, penetration, contact_pos, i_va = self._func_contact_convex_convex_sdf(
                #     #     i_ga, i_gb, i_b, self.contact_cache[i_ga, i_gb, i_b].i_va_ws
                #     # )
                #     # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = i_va
                #     # is_col, normal, penetration, contact_pos = _func_contact_vertex_sdf(i_ga, i_gb, i_b)
                #     # TODO ndarray
                #     pass
                # elif (i_detection == 0) and overlap_ratio_b > _mpr_to_sdf_overlap_ratio:
                #     # is_col, normal, penetration, contact_pos = _func_contact_vertex_sdf(i_gb, i_ga, i_b)
                #     # normal = -normal
                #     # TODO ndarray
                #     pass
                # else:
                #     is_col, normal, penetration, contact_pos = _func_mpr_contact(
                #         i_ga, i_gb, i_b, contact_cache_normal[i_ga, i_gb, i_b]
                #     )

                #     # Fallback on SDF if collision is detected by MPR but no collision direction was cached but the
                #     # initial penetration is already quite large, because the contact information provided by MPR
                #     # may be unreliable in such a case.
                #     # Here it is assumed that generic SDF is much slower than MPR, so it is faster in average
                #     # to first make sure that the geometries are truly colliding and only after to run SDF if
                #     # necessary. This would probably not be the case anymore if it was possible to rely on
                #     # specialized SDF implementation for convex-convex collision detection in the first place.
                #     # if ti.static(not self._solver._enable_mujoco_compatibility):
                #     #     is_mpr_guess_direction_available = (
                #     #         ti.abs(self.contact_cache[i_ga, i_gb, i_b].normal) > gs.EPS
                #     #     ).any()
                #     #     if is_col and penetration > tolerance and not is_mpr_guess_direction_available:
                #     #         # Note that SDF may detect different collision points depending on geometry ordering.
                #     #         # Because of this, it is necessary to run it twice and take the contact information
                #     #         # associated with the point of deepest penetration.
                #     #         is_col_a, normal_a, penetration_a, contact_pos_a = self._func_contact_vertex_sdf(
                #     #             i_ga, i_gb, i_b
                #     #         )
                #     #         is_col_b, normal_b, penetration_b, contact_pos_b = self._func_contact_vertex_sdf(
                #     #             i_gb, i_ga, i_b
                #     #         )
                #     #         if is_col_a and (not is_col_b or penetration_a >= penetration_b):
                #     #             normal, penetration, contact_pos = normal_a, penetration_a, contact_pos_a
                #     #         elif is_col_b and (not is_col_a or penetration_b > penetration_a):
                #     #             normal, penetration, contact_pos = -normal_b, penetration_b, contact_pos_b

            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    _func_add_contact(i_ga, i_gb, normal_0, contact_pos_0, penetration_0, i_b)
                    if multi_contact:
                        # perturb geom_a around two orthogonal axes to find multiple contacts
                        axis_0, axis_1 = _func_contact_orthogonals(
                            i_ga, i_gb, normal_0, i_b, geoms_init_AABB, geoms_info_link_idx, links_state_i_quat
                        )
                        n_con = 1

                    # if ti.static(not _enable_mujoco_compatibility):
                    #     contact_cache_normal[i_ga, i_gb, i_b] = normal
                else:
                    # Clear collision normal cache if not in contact
                    contact_cache_normal[i_ga, i_gb, i_b].fill(0.0)

            elif multi_contact and is_col_0 > 0 and is_col > 0:
                repeated = False
                for i_con in range(n_con):
                    if not repeated:
                        idx_prev = n_contacts[i_b] - 1 - i_con
                        prev_contact = contact_data_pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                if not repeated:
                    # Apply first-order correction of small rotation perturbation.
                    # First, unrotate the normal direction, then cancel virtual penetation over-estimation.
                    # The way the contact normal gets twisted by applying perturbation of geometry poses is
                    # unpredictable as it depends on the final portal discovered by MPR. Alternatively, let
                    # compute the mininal rotation that makes the corrected twisted normal as closed as
                    # possible to the original one, up to the scale of the perturbation, then apply
                    # first-order Taylor expension of Rodrigues' rotation formula.
                    twist_rotvec = ti.math.clamp(normal.cross(normal_0), -_mc_perturbation, _mc_perturbation)
                    normal += twist_rotvec.cross(normal)
                    contact_shift = contact_pos - contact_pos_0
                    depth_lever = ti.abs(axis.cross(contact_shift).dot(normal))
                    penetration = ti.min(penetration - 2 * _mc_perturbation * depth_lever, penetration_0)

                    if penetration > 0.0:
                        _func_add_contact(
                            i_ga,
                            i_gb,
                            normal,
                            contact_pos,
                            penetration,
                            i_b,
                            n_contacts,
                            contact_data_geom_a,
                            contact_data_geom_b,
                            contact_data_normal,
                            contact_data_pos,
                            contact_data_penetration,
                            contact_data_friction,
                            contact_data_sol_params,
                            contact_data_link_a,
                            contact_data_link_b,
                            geoms_info_friction,
                            geoms_state_friction_ratio,
                            geoms_info_sol_params,
                            geoms_info_link_idx,
                        )
                        n_con = n_con + 1

                geoms_state_pos[i_ga, i_b] = ga_pos
                geoms_state_quat[i_ga, i_b] = ga_quat
                geoms_state_pos[i_gb, i_b] = gb_pos
                geoms_state_quat[i_gb, i_b] = gb_quat


def make_kernel_narrow_phase(is_ndarray: bool, is_serial: bool = False):
    VT = vec_types(is_ndarray)

    @ti.kernel
    def _kernel_narrow_phase(
        broad_collision_pairs: VT.I,
        n_broad_pairs: VT.I,
        geoms_info_link_idx: VT.I,
        geoms_info_type: VT.I,
        geoms_info_is_convex: VT.I,
        geoms_info_contype: VT.I,
        geoms_info_conaffinity: VT.I,
        geoms_info_is_fixed: VT.I,
    ):
        ti.loop_config(serialize=is_serial)
        for i_b in range(n_broad_pairs.shape[0]):
            for i_pair in range(n_broad_pairs[i_b]):
                i_ga = broad_collision_pairs[i_pair, i_b][0]
                i_gb = broad_collision_pairs[i_pair, i_b][1]

                _func_mpr(i_ga, i_gb, i_b)

    return _kernel_narrow_phase
