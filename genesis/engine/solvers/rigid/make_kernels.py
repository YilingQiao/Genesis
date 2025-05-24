from .data_class import DofsState, DofsInfo, GlobalData, vec_types, get_array_type
from math import pi
import taichi as ti
import genesis as gs
import genesis.utils.geom as gu
from genesis.styles import colors, formats

_enable_self_collision = True
_enable_adjacent_collision = False
batch_links_info = False
max_collision_pairs = 8192



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

    @ti.func
    def _func_is_geom_aabbs_overlap(
        i_ga: gs.ti_int,
        i_gb: gs.ti_int,
        i_b: gs.ti_int,
        geoms_state_aabb_max: VT.V3,
        geoms_state_aabb_min: VT.V3,
    ) -> bool:
        return not (
            (geoms_state_aabb_max[i_ga, i_b] <= geoms_state_aabb_min[i_gb, i_b]).any()
            or (geoms_state_aabb_min[i_ga, i_b] >= geoms_state_aabb_max[i_gb, i_b]).any()
        )

    @ti.func
    def _func_check_collision_valid(
        i_ga: gs.ti_int,
        i_gb: gs.ti_int,
        i_b: gs.ti_int,
        links_info_root_idx: VT.I,
        links_info_parent_idx: VT.I,
        geoms_info_link_idx: VT.I,
        geoms_info_contype: VT.I,
        geoms_info_conaffinity: VT.I,
        links_info_is_fixed: VT.I,
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
                        broad_collision_pairs[n_broad_pairs[i_b], 0, i_b] = i_ga
                        broad_collision_pairs[n_broad_pairs[i_b], 1, i_b] = i_gb
                        n_broad_pairs[i_b] = n_broad_pairs[i_b] + 1
                    active_buffer[n_active, i_b] = sort_buffer_i_g[i, i_b]
                    n_active = n_active + 1
                else:
                    i_g_to_remove = sort_buffer_i_g[i, i_b]
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
def _func_geom_overlap_ratio(i_ga, i_gb, i_b):
    pass
    # # Check if one geom 'i_ga' is likely fully (1) or partially (2) enclosed in another geom 'i_gb'.
    # # 1. 'Broad phase': Check if bounding box 'i_ga' is fully enclosed into the other bounding box
    # # 2. 'Mid phase': Check if rotated bounding box 'i_ga' is fully enclosed into the other bounding box
    # # 3. 'Narrow phase': Check if all corners of rotated bounding box 'i_ga' are inside the other true geom
    # overlap_ratio = gs.ti_float(0.0)

    # # Broad phase 1
    # is_enclosed_dims = (
    #     geoms_state_aabb_min[i_gb, i_b] < geoms_state_aabb_min[i_ga, i_b]
    # ) & (geoms_state_aabb_max[i_ga, i_b] < geoms_state_aabb_max[i_gb, i_b])

    # # Mid phase 2
    # if is_enclosed_dims.all():
    #     for i_corner in ti.static((0, 7)):
    #         corner_pos = gu.ti_inv_transform_by_trans_quat(
    #             gu.ti_transform_by_trans_quat(
    #                 geoms_init_AABB[i_ga, i_corner],
    #                 geoms_state_pos[i_ga, i_b],
    #                 geoms_state_quat[i_ga, i_b],
    #             ),
    #             geoms_state_pos[i_gb, i_b],
    #             geoms_state_quat[i_gb, i_b],
    #         )
    #         if i_corner == 0:
    #             is_enclosed_dims &= geoms_init_AABB[i_gb, i_corner] < corner_pos
    #         else:
    #             is_enclosed_dims &= corner_pos < geoms_init_AABB[i_gb, i_corner]

    # # Narrow phase 3
    # dists = ti.Vector.zero(gs.ti_float, 8)
    # if is_enclosed_dims.all():
    #     # Check whether the bound box 'i_ga' is fully enclosed
    #     is_enclosed = True
    #     for i_corner in range(8):
    #         corner_pos = gu.ti_transform_by_trans_quat(
    #             geoms_init_AABB[i_ga, i_corner],
    #             geoms_state_pos[i_ga, i_b],
    #             geoms_state_quat[i_ga, i_b],
    #         )
    #         dists[i_corner] = sdf_world(corner_pos, i_gb, i_b)
    #         if dists[i_corner] > 0.0:
    #             is_enclosed = False

    #     # Approximate the overlapping ratio.
    #     # It is defined as the ratio between the average signed distance of all the corners of the bounding box
    #     # 'i_ga' from the true convex geometry 'i_gb', and the length of the box along this specific direction.
    #     if is_enclosed:
    #         overlap_ratio = 1.0
    #     else:
    #         box_size = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
    #         dist_diff = ti.Vector(
    #             [
    #                 dists[4] + dists[5] + dists[6] + dists[7] - dists[0] - dists[1] - dists[2] - dists[3],
    #                 dists[2] + dists[3] + dists[6] + dists[7] - dists[0] - dists[1] - dists[4] - dists[5],
    #                 dists[1] + dists[3] + dists[5] + dists[7] - dists[0] - dists[2] - dists[4] - dists[6],
    #             ],
    #             dt=gs.ti_float,
    #         )
    #         overlap_dir = (dist_diff / box_size).normalized()
    #         overlap_length = box_size.dot(ti.abs(overlap_dir))
    #         overlap_ratio = ti.math.clamp(0.5 - (dists.sum() / 8) / overlap_length, 0.0, 1.0)

    # return overlap_ratio



def make_kernel_narrow_phase(is_ndarray: bool, is_serial: bool = False):
    VT = vec_types(is_ndarray)

    CCD_EPS = 1e-9
    CCD_TOLERANCE = 1e-6
    _enable_mujoco_compatibility = False
    CCD_ITERATIONS = 50



    @ti.func
    def _func_rotate_frame(
        i_ga, 
        contact_pos, 
        qrot, 
        i_b,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
    ):
        geoms_state_quat[i_ga, i_b] = gu.ti_transform_quat_by_quat(
            geoms_state_quat[i_ga, i_b], qrot
        )

        rel = contact_pos - geoms_state_pos[i_ga, i_b]
        vec = gu.ti_transform_by_quat(rel, qrot)
        vec = vec - rel
        geoms_state_pos[i_ga, i_b] = geoms_state_pos[i_ga, i_b] - vec




    @ti.func
    def _func_add_contact(
        i_ga,
        i_gb,
        normal,
        contact_pos,
        penetration,
        i_b,
        n_contacts: VT.I,
        contact_data_geom_a: VT.I,
        contact_data_geom_b: VT.I,
        contact_data_normal: VT.V3,
        contact_data_pos: VT.V3,
        contact_data_penetration: VT.F,
        contact_data_friction: VT.F,
        contact_data_sol_params: VT.V7,
        contact_data_link_a: VT.I,
        contact_data_link_b: VT.I,
        geoms_info_friction: VT.F,
        geoms_state_friction_ratio: VT.F,
        geoms_info_sol_params: VT.V7,
        geoms_info_link_idx: VT.I,
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
        geoms_init_AABB: VT.V3,
        geoms_info_link_idx: VT.I,
        links_state_i_quat: VT.V4,
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



    @ti.func
    def _func_support_world(
        d: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        """
        support position for a world direction
        """

        d_mesh = gu.ti_transform_by_quat(d, gu.ti_inv_quat(geoms_state_quat[i_g, i_b]))
        v, vid = _func_support_mesh(
            d_mesh=d_mesh, 
            i_g=i_g, 
            support_cell_start=support_cell_start, 
            support_vid=support_vid, 
            support_v=support_v)
        v_ = gu.ti_transform_by_trans_quat(v, geoms_state_pos[i_g, i_b], geoms_state_quat[i_g, i_b])
        return v_, vid


    @ti.func
    def _func_support_mesh(
        d_mesh: gs.ti_vec3, 
        i_g: ti.i32, 
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        """
        support point at mesh frame coordinate.
        """
        theta = ti.atan2(d_mesh[1], d_mesh[0])  # [-pi, pi]
        phi = ti.acos(d_mesh[2])  # [0, pi]

        support_res = gs.ti_int(180)
        dot_max = gs.ti_float(-1e20)
        v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        vid = 0

        ii = (theta + pi) / pi / 2 * support_res
        jj = phi / pi * support_res

        for i4 in range(4):
            i, j = gs.ti_int(0), gs.ti_int(0)
            if i4 % 2:
                i = gs.ti_int(ti.math.ceil(ii) % support_res)
            else:
                i = gs.ti_int(ti.math.floor(ii) % support_res)

            if i4 // 2 > 0:
                j = gs.ti_int(ti.math.clamp(ti.math.ceil(jj), 0, support_res - 1))
                if j == support_res - 1:
                    j = support_res - 2
            else:
                j = gs.ti_int(ti.math.clamp(ti.math.floor(jj), 0, support_res - 1))
                if j == 0:
                    j = 1

            support_idx = gs.ti_int(support_cell_start[i_g] + i * support_res + j)
            _vid = support_vid[support_idx]
            pos = support_v[support_idx]
            dot = pos.dot(d_mesh)

            if dot > dot_max:
                v = pos
                dot_max = dot
                vid = _vid

        return v, vid

    @ti.func
    def mpr_point_segment_dist2(
        P: gs.ti_vec3, 
        A: gs.ti_vec3, 
        B: gs.ti_vec3
    ):
        AB = B - A
        AP = P - A
        AB_AB = AB.dot(AB)
        AP_AB = AP.dot(AB)
        t = AP_AB / AB_AB
        if t < CCD_EPS:
            t = gs.ti_float(0.0)
        elif t > 1.0 - CCD_EPS:
            t = gs.ti_float(1.0)
        Q = A + AB * t

        return ((P - Q) ** 2).sum(), Q


    @ti.func
    def mpr_point_tri_depth(
        P: gs.ti_vec3, 
        x0: gs.ti_vec3, 
        B: gs.ti_vec3, 
        C: gs.ti_vec3
    ):
        d1 = B - x0
        d2 = C - x0
        a = x0 - P
        u = a.dot(a)
        v = d1.dot(d1)
        w = d2.dot(d2)
        p = a.dot(d1)
        q = a.dot(d2)
        r = d1.dot(d2)

        d = w * v - r * r
        dist = s = t = gs.ti_float(0.0)
        pdir = gs.ti_vec3([0.0, 0.0, 0.0])
        if ti.abs(d) < CCD_EPS:
            s = t = -1.0
        else:
            s = (q * r - w * p) / d
            t = (-s * r - q) / w

        if (
            (s > -CCD_EPS)
            and (s < 1.0 + CCD_EPS)
            and (t > -CCD_EPS)
            and (t < 1.0 + CCD_EPS)
            and (t + s < 1.0 + CCD_EPS)
        ):
            pdir = x0 + d1 * s + d2 * t
            dist = ((P - pdir) ** 2).sum()
        else:
            dist, pdir = mpr_point_segment_dist2(P, x0, B)
            dist2, pdir2 = mpr_point_segment_dist2(P, x0, C)
            if dist2 < dist:
                dist = dist2
                pdir = pdir2

            dist2, pdir2 = mpr_point_segment_dist2(P, B, C)
            if dist2 < dist:
                dist = dist2
                pdir = pdir2

        return ti.sqrt(dist), pdir


    @ti.func
    def compute_support(
        direction: gs.ti_vec3, 
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        v1 = support_driver(direction, i_ga, i_b, geoms_info_type, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat, support_cell_start, support_vid, support_v)
        v2 = support_driver(-direction, i_gb, i_b, geoms_info_type, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat, support_cell_start, support_vid, support_v)

        v = v1 - v2
        return v, v1, v2

    @ti.func
    def mpr_swap(
        i: ti.i32, 
        j: ti.i32, 
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v1: VT.V3, 
        simplex_support_v2: VT.V3, 
        simplex_support_v: VT.V3
        ):
        simplex_support_v1[i_ga, i_gb, i, i_b], simplex_support_v1[i_ga, i_gb, j, i_b] = (
            simplex_support_v1[i_ga, i_gb, j, i_b],
            simplex_support_v1[i_ga, i_gb, i, i_b],
        )
        simplex_support_v2[i_ga, i_gb, i, i_b], simplex_support_v2[i_ga, i_gb, j, i_b] = (
            simplex_support_v2[i_ga, i_gb, j, i_b],
            simplex_support_v2[i_ga, i_gb, i, i_b],
        )
        simplex_support_v[i_ga, i_gb, i, i_b], simplex_support_v[i_ga, i_gb, j, i_b] = (
            simplex_support_v[i_ga, i_gb, j, i_b],
            simplex_support_v[i_ga, i_gb, i, i_b],
        )


    @ti.func
    def mpr_discover_portal(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        normal_ws: gs.ti_vec3, 
        simplex_size: VT.I, 
        simplex_support_v1: VT.V3, 
        simplex_support_v2: VT.V3, 
        simplex_support_v: VT.V3, 
        geoms_init_AABB: VT.V3,
        geom_state_pos: VT.V3,
        geom_state_quat: VT.V4,
        geoms_info_center: VT.V3,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        ret = 0
        simplex_size[i_ga, i_gb, i_b] = 0

        # Completely different center logics depending on normal guess is provided or not
        if (ti.abs(normal_ws) < CCD_EPS).all():
            center_a = gu.ti_transform_by_trans_quat(geoms_info_center[i_ga], geom_state_pos[i_ga, i_b], geom_state_quat[i_ga, i_b])
            center_b = gu.ti_transform_by_trans_quat(geoms_info_center[i_gb], geom_state_pos[i_gb, i_b], geom_state_quat[i_gb, i_b])

            simplex_support_v1[i_ga, i_gb, 0, i_b] = center_a
            simplex_support_v2[i_ga, i_gb, 0, i_b] = center_b
            simplex_support_v[i_ga, i_gb, 0, i_b] = center_a - center_b
            simplex_size[i_ga, i_gb, i_b] = 1
        else:
            # Start with the center of the bounding box. They will be shifted if necessary anyway.
            center_a_local = 0.5 * (geoms_init_AABB[i_ga, 7] + geoms_init_AABB[i_ga, 0])
            center_a = gu.ti_transform_by_trans_quat(center_a_local, geom_state_pos[i_ga, i_b], geom_state_quat[i_ga, i_b])
            center_b_local = 0.5 * (geoms_init_AABB[i_gb, 7] + geoms_init_AABB[i_gb, 0])
            center_b = gu.ti_transform_by_trans_quat(center_b_local, geom_state_pos[i_gb, i_b], geom_state_quat[i_gb, i_b])
            delta = center_a - center_b

            # Offset the center of each geometry based on the desired search direction if provided
            # Skip if almost colinear already.
            normal = delta.normalized()
            if (ti.abs(normal_ws) > CCD_EPS).any() or normal_ws.cross(normal).norm() > CCD_TOLERANCE:
                # Compute the target offset
                offset = delta.dot(normal_ws) * normal_ws - delta
                offset_norm = offset.norm()

                if offset_norm > CCD_TOLERANCE:
                    # Compute the size of the bounding boxes along the target offset direction.
                    # First, move the direction in local box frame
                    dir_offset = offset / offset_norm
                    dir_offset_local_a = gu.ti_transform_by_quat(dir_offset, gu.ti_inv_quat(geom_state_quat[i_ga, i_b]))
                    dir_offset_local_b = gu.ti_transform_by_quat(dir_offset, gu.ti_inv_quat(geom_state_quat[i_gb, i_b]))
                    box_size_a = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
                    box_size_b = geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]
                    length_a = box_size_a.dot(ti.abs(dir_offset_local_a))
                    length_b = box_size_b.dot(ti.abs(dir_offset_local_b))

                    # Shift the center of each geometry
                    offset_ratio = ti.min(offset_norm / (length_a + length_b), 0.5)
                    simplex_support_v1[i_ga, i_gb, 0, i_b] = center_a + dir_offset * length_a * offset_ratio
                    simplex_support_v2[i_ga, i_gb, 0, i_b] = center_b - dir_offset * length_b * offset_ratio
                    simplex_support_v[i_ga, i_gb, 0, i_b] = (
                        simplex_support_v1[i_ga, i_gb, 0, i_b] - simplex_support_v2[i_ga, i_gb, 0, i_b]
                    )

        if (ti.abs(simplex_support_v[i_ga, i_gb, 0, i_b]) < CCD_EPS).all():
            simplex_support_v[i_ga, i_gb, 0, i_b][0] += 10.0 * CCD_EPS

        direction = -simplex_support_v[i_ga, i_gb, 0, i_b].normalized()

        v, v1, v2 = compute_support(
            direction, i_ga, i_gb, i_b, 
            geoms_info_type, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat, support_cell_start, support_vid, support_v)

        simplex_support_v1[i_ga, i_gb, 1, i_b] = v1
        simplex_support_v2[i_ga, i_gb, 1, i_b] = v2
        simplex_support_v[i_ga, i_gb, 1, i_b] = v
        simplex_size[i_ga, i_gb, i_b] = 2

        dot = v.dot(direction)

        if dot < CCD_EPS:
            ret = -1
        else:
            direction = simplex_support_v[i_ga, i_gb, 0, i_b].cross(simplex_support_v[i_ga, i_gb, 1, i_b])
            if direction.dot(direction) < CCD_EPS:
                if (ti.abs(simplex_support_v[i_ga, i_gb, 1, i_b]) < CCD_EPS).all():
                    ret = 1
                else:
                    ret = 2
            else:
                direction = direction.normalized()
                v, v1, v2 = compute_support(direction, i_ga, i_gb, i_b, geoms_info_type, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat, support_cell_start, support_vid, support_v)
                dot = v.dot(direction)
                if dot < CCD_EPS:
                    ret = -1
                else:
                    simplex_support_v1[i_ga, i_gb, 2, i_b] = v1
                    simplex_support_v2[i_ga, i_gb, 2, i_b] = v2
                    simplex_support_v[i_ga, i_gb, 2, i_b] = v
                    simplex_size[i_ga, i_gb, i_b] = 3

                    va = simplex_support_v[i_ga, i_gb, 1, i_b] - simplex_support_v[i_ga, i_gb, 0, i_b]
                    vb = simplex_support_v[i_ga, i_gb, 2, i_b] - simplex_support_v[i_ga, i_gb, 0, i_b]
                    direction = va.cross(vb)
                    direction = direction.normalized()

                    dot = direction.dot(simplex_support_v[i_ga, i_gb, 0, i_b])
                    if dot > 0:
                        mpr_swap(1, 2, i_ga, i_gb, i_b, simplex_support_v1, simplex_support_v2, simplex_support_v)
                        direction = -direction

                    while simplex_size[i_ga, i_gb, i_b] < 4:
                        v, v1, v2 = compute_support(direction, i_ga, i_gb, i_b, geoms_info_type, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat, support_cell_start, support_vid, support_v)
                        dot = v.dot(direction)
                        if dot < CCD_EPS:
                            ret = -1
                            break

                        cont = False

                        va = simplex_support_v[i_ga, i_gb, 1, i_b].cross(v)
                        dot = va.dot(simplex_support_v[i_ga, i_gb, 0, i_b])
                        if dot < -CCD_EPS:
                            simplex_support_v1[i_ga, i_gb, 2, i_b] = v1
                            simplex_support_v2[i_ga, i_gb, 2, i_b] = v2
                            simplex_support_v[i_ga, i_gb, 2, i_b] = v
                            cont = True

                        if not cont:
                            va = v.cross(simplex_support_v[i_ga, i_gb, 2, i_b])
                            dot = va.dot(simplex_support_v[i_ga, i_gb, 0, i_b])
                            if dot < -CCD_EPS:
                                simplex_support_v1[i_ga, i_gb, 1, i_b] = v1
                                simplex_support_v2[i_ga, i_gb, 1, i_b] = v2
                                simplex_support_v[i_ga, i_gb, 1, i_b] = v
                                cont = True

                        if cont:
                            va = simplex_support_v[i_ga, i_gb, 1, i_b] - simplex_support_v[i_ga, i_gb, 0, i_b]
                            vb = simplex_support_v[i_ga, i_gb, 2, i_b] - simplex_support_v[i_ga, i_gb, 0, i_b]
                            direction = va.cross(vb)
                            direction = direction.normalized()
                        else:
                            simplex_support_v1[i_ga, i_gb, 3, i_b] = v1
                            simplex_support_v2[i_ga, i_gb, 3, i_b] = v2
                            simplex_support_v[i_ga, i_gb, 3, i_b] = v
                            simplex_size[i_ga, i_gb, i_b] = 4

        return ret

    @ti.func
    def support_sphere(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_state_pos: VT.V3,
        geoms_info_data: VT.F,
    ):
        sphere_center = geoms_state_pos[i_g, i_b]
        sphere_radius = geoms_info_data[i_g, 0]
        return sphere_center + direction * sphere_radius

    @ti.func
    def support_ellipsoid(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        geoms_info_data: VT.F,
    ):
        ellipsoid_center = geoms_state_pos[i_g, i_b]
        ellipsoid_scaled_axis = ti.Vector(
            [
                geoms_info_data[i_g, 0] ** 2,
                geoms_info_data[i_g, 1] ** 2,
                geoms_info_data[i_g, 2] ** 2,
            ],
            dt=gs.ti_float,
        )
        ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, geoms_state_quat[i_g, i_b])
        dist = ellipsoid_scaled_axis / ti.sqrt(direction.dot(1.0 / ellipsoid_scaled_axis))
        return ellipsoid_center + direction * dist

    @ti.func
    def support_capsule(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        geoms_info_data: VT.F,
    ):
        capule_center = geoms_state_pos[i_g, i_b]
        capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), geoms_state_quat[i_g, i_b])
        capule_radius = geoms_info_data[i_g, 0]
        capule_halflength = 0.5 * geoms_info_data[i_g, 1]
        capule_endpoint_side = ti.math.sign(direction.dot(capsule_axis))
        capule_endpoint = capule_center + capule_halflength * capule_endpoint_side * capsule_axis
        return capule_endpoint + direction * capule_radius

    @ti.func
    def support_prism(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        collider_prism: VT.V3,
    ):
        istart = 3
        if direction[2] < 0:
            istart = 0

        ibest = istart
        best = collider_prism[istart, i_b].dot(direction)
        for i in range(istart + 1, istart + 3):
            dot = collider_prism[i, i_b].dot(direction)
            if dot > best:
                ibest = i
                best = dot

        return collider_prism[ibest, i_b], ibest

    @ti.func
    def support_box(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
    ):
        d_box = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(geoms_state_quat[i_g, i_b]))

        vid = (d_box[0] > 0) * 4 + (d_box[1] > 0) * 2 + (d_box[2] > 0) * 1
        v_ = ti.Vector(
            [
                ti.math.sign(d_box[0]) * geoms_info_data[i_g, 0] * 0.5,
                ti.math.sign(d_box[1]) * geoms_info_data[i_g, 1] * 0.5,
                ti.math.sign(d_box[2]) * geoms_info_data[i_g, 2] * 0.5,
            ],
            dt=gs.ti_float,
        )

        vid += geoms_info_vert_start[i_g]
        v = gu.ti_transform_by_trans_quat(v_, geoms_state_pos[i_g, i_b], geoms_state_quat[i_g, i_b])
        return v, vid

    @ti.func
    def support_driver(
        direction: gs.ti_vec3, 
        i_g: ti.i32, 
        i_b: ti.i32,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        v = ti.Vector.zero(gs.ti_float, 3)
        geom_type = geoms_info_type[i_g]
        if geom_type == gs.GEOM_TYPE.SPHERE:
            v = support_sphere(direction, i_g, i_b, geoms_state_pos, geoms_info_data)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            v = support_ellipsoid(direction, i_g, i_b, geoms_state_pos, geoms_state_quat, geoms_info_data)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            v = support_capsule(direction, i_g, i_b, geoms_state_pos, geoms_state_quat, geoms_info_data)
        elif geom_type == gs.GEOM_TYPE.BOX:
            v, _ = support_box(direction, i_g, i_b, geoms_info_data, geoms_info_vert_start, geoms_state_pos, geoms_state_quat)
        # TODO ndarray
        # elif geom_type == gs.GEOM_TYPE.TERRAIN:
        #     if ti.static(collider._has_terrain):
        #         v, _ = support_prism(direction, i_g, i_b)
        else:
            v, _ = _func_support_world(
                d=direction, 
                i_g=i_g, 
                i_b=i_b, 
                geoms_state_pos=geoms_state_pos, 
                geoms_state_quat=geoms_state_quat, 
                support_cell_start=support_cell_start, 
                support_vid=support_vid, support_v=support_v)
        return v


    @ti.func
    def mpr_find_penetr_touch(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3
    ):
        is_col = True
        penetration = gs.ti_float(0.0)
        normal = -simplex_support_v[i_ga, i_gb, 0, i_b].normalized()
        pos = (simplex_support_v1[i_ga, i_gb, 1, i_b] + simplex_support_v2[i_ga, i_gb, 1, i_b]) * 0.5
        return is_col, normal, penetration, pos

    @ti.func
    def mpr_find_penetr_segment(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3
    ):
        is_col = True
        penetration = simplex_support_v[i_ga, i_gb, 1, i_b].norm()
        normal = -simplex_support_v[i_ga, i_gb, 1, i_b].normalized()
        pos = (simplex_support_v1[i_ga, i_gb, 1, i_b] + simplex_support_v2[i_ga, i_gb, 1, i_b]) * 0.5

        return is_col, normal, penetration, pos

    @ti.func
    def mpr_portal_dir(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3
    ):
        v2v1 = simplex_support_v[i_ga, i_gb, 2, i_b] - simplex_support_v[i_ga, i_gb, 1, i_b]
        v3v1 = simplex_support_v[i_ga, i_gb, 3, i_b] - simplex_support_v[i_ga, i_gb, 1, i_b]
        direction = v2v1.cross(v3v1).normalized()
        return direction

    @ti.func
    def mpr_portal_encapsules_origin(
        direction: gs.ti_vec3, 
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3
    ):
        dot = simplex_support_v[i_ga, i_gb, 1, i_b].dot(direction)
        return dot > -CCD_EPS


    @ti.func
    def mpr_portal_can_encapsule_origin(v, direction):
        dot = v.dot(direction)
        return dot > -CCD_EPS


    @ti.func
    def mpr_portal_reach_tolerance(
        v: gs.ti_vec3, 
        direction: gs.ti_vec3, 
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3
    ):
        dv1 = simplex_support_v[i_ga, i_gb, 1, i_b].dot(direction)
        dv2 = simplex_support_v[i_ga, i_gb, 2, i_b].dot(direction)
        dv3 = simplex_support_v[i_ga, i_gb, 3, i_b].dot(direction)
        dv4 = v.dot(direction)
        dot1 = ti.min(dv4 - dv1, dv4 - dv2, dv4 - dv3)
        return dot1 < CCD_TOLERANCE + CCD_EPS * ti.max(1.0, dot1)

    @ti.func
    def mpr_expand_portal(
        v: gs.ti_vec3, 
        v1: gs.ti_vec3, 
        v2: gs.ti_vec3, 
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
    ):
        v4v0 = v.cross(simplex_support_v[i_ga, i_gb, 0, i_b])
        dot = simplex_support_v[i_ga, i_gb, 1, i_b].dot(v4v0)

        if dot > 0:
            dot = simplex_support_v[i_ga, i_gb, 2, i_b].dot(v4v0)
            if dot > 0:
                simplex_support_v1[i_ga, i_gb, 1, i_b] = v1
                simplex_support_v2[i_ga, i_gb, 1, i_b] = v2
                simplex_support_v[i_ga, i_gb, 1, i_b] = v

            else:
                simplex_support_v1[i_ga, i_gb, 3, i_b] = v1
                simplex_support_v2[i_ga, i_gb, 3, i_b] = v2
                simplex_support_v[i_ga, i_gb, 3, i_b] = v

        else:
            dot = simplex_support_v[i_ga, i_gb, 3, i_b].dot(v4v0)
            if dot > 0:
                simplex_support_v1[i_ga, i_gb, 2, i_b] = v1
                simplex_support_v2[i_ga, i_gb, 2, i_b] = v2
                simplex_support_v[i_ga, i_gb, 2, i_b] = v

            else:
                simplex_support_v1[i_ga, i_gb, 1, i_b] = v1
                simplex_support_v2[i_ga, i_gb, 1, i_b] = v2
                simplex_support_v[i_ga, i_gb, 1, i_b] = v
                
    @ti.func
    def mpr_refine_portal(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
        
    ):
        ret = 1
        while True:
            direction = mpr_portal_dir(i_ga, i_gb, i_b, simplex_support_v)
            
            if mpr_portal_encapsules_origin(direction, i_ga, i_gb, i_b, simplex_support_v):
                ret = 0
                break

            v, v1, v2 = compute_support(
                direction, 
                i_ga, 
                i_gb, 
                i_b,
                geoms_info_type,
                geoms_info_data,
                geoms_info_vert_start,
                geoms_state_pos,
                geoms_state_quat,
                support_cell_start,
                support_vid,
                support_v,
            )
            if not mpr_portal_can_encapsule_origin(v, direction) or mpr_portal_reach_tolerance(
                v, 
                direction, 
                i_ga, 
                i_gb, 
                i_b,
                simplex_support_v
            ):
                ret = -1
                break

            mpr_expand_portal(
                v=v, 
                v1=v1, 
                v2=v2, 
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b, 
                simplex_support_v=simplex_support_v, 
                simplex_support_v1=simplex_support_v1, 
                simplex_support_v2=simplex_support_v2)
        return ret

    @ti.func
    def mpr_find_pos(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
    ):
        b = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.ti_float)

        # Only look into the direction of the portal for consistency with penetration depth computation
        # TODO ndarray
        # if ti.static(_enable_mujoco_compatibility):
        #     for i in range(4):
        #         i1, i2, i3 = (i % 2) + 1, (i + 2) % 4, 3 * ((i + 1) % 2)
        #         vec = simplex_support_v[i_ga, i_gb, i1, i_b].v.cross(simplex_support_v[i_ga, i_gb, i2, i_b].v)
        #         b[i] = vec.dot(simplex_support_v[i_ga, i_gb, i3, i_b].v) * (1 - 2 * (((i + 1) // 2) % 2))

        sum_ = b.sum()

        if sum_ < CCD_EPS:
            direction = mpr_portal_dir(i_ga, i_gb, i_b, simplex_support_v)
            b[0] = 0.0
            for i in range(1, 4):
                i1, i2 = i % 3 + 1, (i + 1) % 3 + 1
                vec = simplex_support_v[i_ga, i_gb, i1, i_b].cross(simplex_support_v[i_ga, i_gb, i2, i_b])
                b[i] = vec.dot(direction)
            sum_ = b.sum()

        p1 = gs.ti_vec3([0.0, 0.0, 0.0])
        p2 = gs.ti_vec3([0.0, 0.0, 0.0])
        for i in range(4):
            p1 += b[i] * simplex_support_v1[i_ga, i_gb, i, i_b]
            p2 += b[i] * simplex_support_v2[i_ga, i_gb, i, i_b]

        return (0.5 / sum_) * (p1 + p2)


    @ti.func
    def mpr_find_penetration(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        simplex_support_v: VT.V3,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
    ):
        iterations = 0

        is_col = False
        pos = gs.ti_vec3([0.0, 0.0, 0.0])
        normal = gs.ti_vec3([0.0, 0.0, 0.0])
        penetration = gs.ti_float(0.0)

        while True:
            direction = mpr_portal_dir(i_ga, i_gb, i_b, simplex_support_v)
            v, v1, v2 = compute_support(
                direction=direction, 
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b, 
                geoms_info_type=geoms_info_type, 
                geoms_info_data=geoms_info_data, 
                geoms_info_vert_start=geoms_info_vert_start, 
                geoms_state_pos=geoms_state_pos, 
                geoms_state_quat=geoms_state_quat, 
                support_cell_start=support_cell_start, 
                support_vid=support_vid, 
                support_v=support_v)
            

            if mpr_portal_reach_tolerance(v, direction, i_ga, i_gb, i_b, simplex_support_v) or iterations > CCD_ITERATIONS:
                # The contact point is defined as the projection of the origin onto the portal, i.e. the closest point
                # to the origin that lies inside the portal.
                # Let's consider the portal as an infinite plane rather than a face triangle. This makes sense because
                # the projection of the origin must be strictly included into the portal triangle for it to correspond
                # to the true penetration depth.
                # For reference about this propery, see 'Collision Handling with Variable-Step Integrators' Theorem 4.2:
                # https://modiasim.github.io/Modia3D.jl/resources/documentation/CollisionHandling_Neumayr_Otter_2017.pdf
                #
                # In theory, the center should have been shifted until to end up with the one and only portal satisfying
                # this condition. However, a native implementation of this process must be avoided because it would be
                # very costly. In practice, assuming the portal is infinite provides a decent approximation of the true
                # penetration depth (it is actually a lower-bound estimate according to Theorem 4.3) and normal without
                # requiring any additional computations.
                # See: https://github.com/danfis/libccd/issues/71#issuecomment-660415008
                #
                # An improved version of MPR has been proposed to find the right portal in an efficient way.
                # See: https://arxiv.org/pdf/2304.07357
                # Implementation: https://github.com/weigao95/mind-fcl/blob/main/include/fcl/cvx_collide/mpr.h
                #
                # The original paper introducing MPR algorithm is available here:
                # https://archive.org/details/game-programming-gems-7
                if ti.static(_enable_mujoco_compatibility):
                    penetration, pdir = mpr_point_tri_depth(
                        gs.ti_vec3([0.0, 0.0, 0.0]),
                        simplex_support_v1[i_ga, i_gb, 1, i_b],
                        simplex_support_v2[i_ga, i_gb, 2, i_b],
                        simplex_support_v2[i_ga, i_gb, 3, i_b],
                    )
                    normal = -pdir.normalized()
                else:
                    penetration = direction.dot(simplex_support_v[i_ga, i_gb, 1, i_b])
                    normal = -direction

                is_col = True
                pos = mpr_find_pos(
                    i_ga, 
                    i_gb, 
                    i_b, 
                    simplex_support_v,
                    simplex_support_v1,
                    simplex_support_v2)
                break

            mpr_expand_portal(
                v=v, 
                v1=v1, 
                v2=v2, 
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b,
                simplex_support_v=simplex_support_v,
                simplex_support_v1=simplex_support_v1,
                simplex_support_v2=simplex_support_v2
            )
            iterations += 1

        return is_col, normal, penetration, pos

    @ti.func
    def func_mpr_contact(
        i_ga: ti.i32, 
        i_gb: ti.i32, 
        i_b: ti.i32, 
        normal_ws: gs.ti_vec3,
        simplex_size: VT.I,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3,
        geoms_init_AABB: VT.V3,
        geom_state_pos: VT.V3,
        geom_state_quat: VT.V4,
        geoms_info_center: VT.V3,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,

    ):

        res = mpr_discover_portal(
            i_ga=i_ga, 
            i_gb=i_gb, 
            i_b=i_b, 
            normal_ws=normal_ws,
            simplex_size=simplex_size,
            simplex_support_v1=simplex_support_v1,
            simplex_support_v2=simplex_support_v2,
            simplex_support_v=simplex_support_v,
            geoms_init_AABB=geoms_init_AABB,
            geom_state_pos=geom_state_pos,
            geom_state_quat=geom_state_quat,
            geoms_info_center=geoms_info_center,
            geoms_info_type=geoms_info_type,
            geoms_info_data=geoms_info_data,
            geoms_info_vert_start=geoms_info_vert_start,
            geoms_state_pos=geoms_state_pos,
            geoms_state_quat=geoms_state_quat,
            support_cell_start=support_cell_start,
            support_vid=support_vid,
            support_v=support_v
        )
        is_col = False
        pos = gs.ti_vec3([0.0, 0.0, 0.0])
        normal = gs.ti_vec3([0.0, 0.0, 0.0])
        penetration = gs.ti_float(0.0)

        if res == 1:
            is_col, normal, penetration, pos = mpr_find_penetr_touch(
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b,
                simplex_support_v1=simplex_support_v1,
                simplex_support_v2=simplex_support_v2,
                simplex_support_v=simplex_support_v
            )
        elif res == 2:
            is_col, normal, penetration, pos = mpr_find_penetr_segment(
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b,
                simplex_support_v1=simplex_support_v1,
                simplex_support_v2=simplex_support_v2,
                simplex_support_v=simplex_support_v
            )
        elif res == 0:
            res = mpr_refine_portal(
                i_ga=i_ga, 
                i_gb=i_gb, 
                i_b=i_b, 
                simplex_support_v1=simplex_support_v1,
                simplex_support_v2=simplex_support_v2,
                simplex_support_v=simplex_support_v,
                geoms_info_type=geoms_info_type,
                geoms_info_data=geoms_info_data,
                geoms_info_vert_start=geoms_info_vert_start,
                geoms_state_pos=geoms_state_pos,
                geoms_state_quat=geoms_state_quat,
                support_cell_start=support_cell_start,
                support_vid=support_vid,
                support_v=support_v
            )
            if res >= 0:
                is_col, normal, penetration, pos = mpr_find_penetration(
                    i_ga=i_ga, 
                    i_gb=i_gb, 
                    i_b=i_b,
                    simplex_support_v1=simplex_support_v1,
                    simplex_support_v2=simplex_support_v2,
                    simplex_support_v=simplex_support_v,
                    geoms_info_type=geoms_info_type,
                    geoms_info_data=geoms_info_data,
                    geoms_info_vert_start=geoms_info_vert_start,
                    geoms_state_pos=geoms_state_pos,
                    geoms_state_quat=geoms_state_quat,
                    support_cell_start=support_cell_start,
                    support_vid=support_vid,
                    support_v=support_v
                )

        return is_col, normal, penetration, pos

    @ti.func
    def _func_compute_tolerance(
        i_ga: gs.ti_int,
        i_gb: gs.ti_int,
        i_b: gs.ti_int,
        geoms_init_AABB: VT.V3,
    ):
        aabb_size_a = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
        aabb_size_b = geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]
        tolerance_abs = 0.5 * _mc_tolerance * ti.min(aabb_size_a.norm(), aabb_size_b.norm())
        return tolerance_abs


    ## only one mpr
    @ti.func
    def _func_mpr(
        i_ga: gs.ti_int,
        i_gb: gs.ti_int,
        i_b: gs.ti_int,
        geoms_info_type: VT.I,
        geoms_info_link_idx: VT.I,
        geoms_info_data: VT.F,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        contact_cache_normal: VT.V3,
        contact_data_pos: VT.V3,
        n_contacts: VT.I,
        contact_data_geom_a: VT.I,
        contact_data_geom_b: VT.I,
        contact_data_normal: VT.V3,
        contact_data_penetration: VT.F,
        contact_data_friction: VT.F,
        contact_data_sol_params: VT.V7,
        contact_data_link_a: VT.I,
        contact_data_link_b: VT.I,
        geoms_info_friction: VT.F,
        geoms_state_friction_ratio: VT.F,
        geoms_info_sol_params: VT.V7,
        geoms_info_vert_start: VT.I,
        geoms_init_AABB: VT.V3,
        links_state_i_quat: VT.V4,
        simplex_size: VT.I,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
        geoms_info_center: VT.V3,
    ):
        if geoms_info_type[i_ga] > geoms_info_type[i_gb]:
            i_gb, i_ga = i_ga, i_gb


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

            for i_detection in range(1): # TODO ndaary: 5
                if multi_contact and is_col_0:
                    # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                    # otherwise it would be more sensitive to ill-conditionning.
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.ti_rotvec_to_quat(_mc_perturbation * axis)
                    _func_rotate_frame(i_ga, contact_pos_0, qrot, i_b, geoms_state_pos, geoms_state_quat)
                    _func_rotate_frame(i_gb, contact_pos_0, gu.ti_inv_quat(qrot), i_b, geoms_state_pos, geoms_state_quat)

                if (multi_contact and is_col_0) or (i_detection == 0):
                    # MPR cannot handle collision detection for fully enclosed geometries. Falling back to SDF.
                    # Note that SDF does not take into account to direction of interest. As such, it cannot be used
                    # reliably for anything else than the point of deepest penetration.
                    
                    is_col, normal, penetration, contact_pos = func_mpr_contact(
                        i_ga, 
                        i_gb, 
                        i_b, 
                        contact_cache_normal[i_ga, i_gb, i_b],
                        simplex_size,
                        simplex_support_v1,
                        simplex_support_v2,
                        simplex_support_v,
                        geoms_init_AABB,
                        geoms_state_pos,
                        geoms_state_quat,
                        geoms_info_center,
                        geoms_info_type,
                        geoms_info_data,
                        geoms_info_vert_start,
                        geoms_state_pos,
                        geoms_state_quat,
                        support_cell_start,
                        support_vid,
                        support_v,
                        
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
                        _func_add_contact(
                            i_ga=i_ga, 
                            i_gb=i_gb, 
                            normal=normal_0, 
                            contact_pos=contact_pos_0, 
                            penetration=penetration_0, 
                            i_b=i_b, 
                            n_contacts=n_contacts, 
                            contact_data_geom_a=contact_data_geom_a, 
                            contact_data_geom_b=contact_data_geom_b, 
                            contact_data_normal=contact_data_normal, 
                            contact_data_pos=contact_data_pos, 
                            contact_data_penetration=contact_data_penetration, 
                            contact_data_friction=contact_data_friction, 
                            contact_data_sol_params=contact_data_sol_params, 
                            contact_data_link_a=contact_data_link_a, 
                            contact_data_link_b=contact_data_link_b, 
                            geoms_info_friction=geoms_info_friction, 
                            geoms_state_friction_ratio=geoms_state_friction_ratio, 
                            geoms_info_sol_params=geoms_info_sol_params, 
                            geoms_info_link_idx=geoms_info_link_idx
                        )
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

    @ti.kernel
    def _kernel_narrow_phase(
        broad_collision_pairs: VT.I,
        n_broad_pairs: VT.I,
        geoms_info_link_idx: VT.I,
        geoms_info_type: VT.I,
        geoms_info_data: VT.F,
        geoms_info_vert_start: VT.I,
        geoms_info_is_convex: VT.I,
        geoms_info_contype: VT.I,
        geoms_info_conaffinity: VT.I,
        geoms_state_pos: VT.V3,
        geoms_state_quat: VT.V4,
        contact_cache_normal: VT.V3,
        contact_data_pos: VT.V3,
        n_contacts: VT.I,
        contact_data_geom_a: VT.I,
        contact_data_geom_b: VT.I,
        contact_data_normal: VT.V3,
        contact_data_penetration: VT.F,
        contact_data_friction: VT.F,
        contact_data_sol_params: VT.V7,
        contact_data_link_a: VT.I,
        contact_data_link_b: VT.I,
        geoms_info_friction: VT.F,
        geoms_state_friction_ratio: VT.F,
        geoms_info_sol_params: VT.V7,
        geoms_init_AABB: VT.V3,
        links_state_i_quat: VT.V4,
        simplex_size: VT.I,
        simplex_support_v1: VT.V3,
        simplex_support_v2: VT.V3,
        simplex_support_v: VT.V3,
        support_cell_start: VT.I,
        support_vid: VT.I,
        support_v: VT.V3,
        geoms_info_center: VT.V3,
    ):
        ti.loop_config(serialize=is_serial)
        for i_b in range(n_broad_pairs.shape[0]):
            for i_pair in range(n_broad_pairs[i_b]):
                i_ga = broad_collision_pairs[i_pair, 0, i_b]
                i_gb = broad_collision_pairs[i_pair, 1, i_b]

                if geoms_info_is_convex[i_ga] and geoms_info_is_convex[i_gb]:
                    _func_mpr(
                        i_ga, 
                        i_gb, 
                        i_b,
                        geoms_info_type,
                        geoms_info_link_idx,
                        geoms_info_data,
                        geoms_state_pos,
                        geoms_state_quat,
                        contact_cache_normal,
                        contact_data_pos,
                        n_contacts,
                        contact_data_geom_a,
                        contact_data_geom_b,
                        contact_data_normal,
                        contact_data_penetration,
                        contact_data_friction,
                        contact_data_sol_params,
                        contact_data_link_a,
                        contact_data_link_b,
                        geoms_info_friction,
                        geoms_state_friction_ratio,
                        geoms_info_sol_params,
                        geoms_info_vert_start,
                        geoms_init_AABB,
                        links_state_i_quat,
                        simplex_size,
                        simplex_support_v1,
                        simplex_support_v2,
                        simplex_support_v,
                        support_cell_start,
                        support_vid,
                        support_v,
                        geoms_info_center,
                    )


    return _kernel_narrow_phase
