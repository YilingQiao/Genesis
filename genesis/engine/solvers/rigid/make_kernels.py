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
    i_ga,
    i_gb,
    i_b,
    links_info_root_idx,
    links_info_parent_idx,
    geoms_info_link_idx,
    geoms_info_contype,
    geoms_info_conaffinity,
    links_info_is_fixed,
):

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
def _func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, geoms_state_aabb_max, geoms_state_aabb_min):
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

                        # --------------------- _func_check_collision_valid
                        # check if the pair is valid
                        # if not _func_check_collision_valid(
                        #     i_ga,
                        #     i_gb,
                        #     i_b,
                        #     links_info_root_idx,
                        #     links_info_parent_idx,
                        #     geoms_info_link_idx,
                        #     geoms_info_contype,
                        #     geoms_info_conaffinity,
                        #     links_info_is_fixed,
                        # ):
                        #     continue

                        i_la = geoms_info_link_idx[i_ga]
                        i_lb = geoms_info_link_idx[i_gb]
                        I_la = [i_la, i_b] if ti.static(batch_links_info) else i_la
                        I_lb = [i_lb, i_b] if ti.static(batch_links_info) else i_lb
                        is_valid = True

                        # geoms in the same link
                        if i_la == i_lb:
                            is_valid = False

                        # self collision
                        if (
                            ti.static(not _enable_self_collision)
                            and links_info_root_idx[I_la] == links_info_root_idx[I_lb]
                        ):
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

                        is_overlap = not (
                            (geoms_state_aabb_max[i_ga, i_b] <= geoms_state_aabb_min[i_gb, i_b]).any()
                            or (geoms_state_aabb_min[i_ga, i_b] >= geoms_state_aabb_max[i_gb, i_b]).any()
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


@ti.kernel
def _func_narrow_phase(self):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collision_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(self._solver._B):
        for i_pair in range(self.n_broad_pairs[i_b]):
            i_ga = self.broad_collision_pairs[i_pair, i_b][0]
            i_gb = self.broad_collision_pairs[i_pair, i_b][1]

            if (
                self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.TERRAIN
                or self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.TERRAIN
            ):
                pass
            elif self._solver.geoms_info[i_ga].is_convex and self._solver.geoms_info[i_gb].is_convex:
                if ti.static(self._solver._box_box_detection):
                    if (
                        self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.BOX
                        and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
                    ):
                        self._func_box_box_contact(i_ga, i_gb, i_b)
                    else:
                        self._func_mpr(i_ga, i_gb, i_b)
                else:
                    self._func_mpr(i_ga, i_gb, i_b)


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

                self._func_mpr(i_ga, i_gb, i_b)

    return _kernel_narrow_phase
