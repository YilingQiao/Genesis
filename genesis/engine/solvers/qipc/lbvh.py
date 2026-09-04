"""LBVH -- Linear Bounding Volume Hierarchy (Karras 2012).

One ``LBVH`` instance per primitive type (triangles, edges).
All buffers are ``qd.tensor`` (default backend=NDARRAY) to enable fastcache
kernel sharing across instances with different sizes.  Build and query methods
are ``@qd.func(requires_top_level=True)`` for use inside graph kernels.

Faithfully ports cgq ``BVHContext`` + ``lbvh_kernels.cu`` + ``bvh_subgraph.h``.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import sort, sort_scratch_slots
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as qd_block

from genesis.engine.solvers.qipc.bvh_math import (
    aabb_combine_aabb,
    aabb_combine_point,
    aabb_expand,
    aabb_init,
    determine_range,
    find_split,
    morton_code_30bit,
)

# structural, host-only: traversal stack depth. Read once to shape `stack_pool`
# and never from device code, so it may stay a module constant.
_BVH_STACK_CAPACITY = 64

_LBVH_INIT_DATA: dict[int, tuple[int, int]] = {}
_LBVH_BACKEND = qd.Backend.NDARRAY


# ---------------------------------------------------------------------------
# Two-phase AABB reduce (equivalent to cgq CUB DeviceReduce + AABBReduceOp)
# ---------------------------------------------------------------------------


@qd.func
def _bin_min_f64(a: qd.f64, b: qd.f64):
    return qd.min(a, b)


@qd.func
def _bin_max_f64(a: qd.f64, b: qd.f64):
    return qd.max(a, b)


@qd.func
def _aabb_reduce_phase1(
    aabbs: qd.template(),
    partials: qd.template(),
    n_rt: qd.template(),
    total_threads: qd.i32,
    block: qd.template(),
):
    """Phase 1: each block reduces its tile of leaf AABBs into per-block partials.

    One kernel launch, each block processes ``block`` leaves, writes 6 f64
    per block into partials[block_id, 0..5].
    """
    loop_config(block_dim=block)
    for i in range(total_threads):
        qd_block.sync()
        n = n_rt[0]
        tid = i % block
        block_id = i // block
        leaf = n - 1 + i

        for comp in qd.static(range(3)):
            v = qd.f64(1e32)
            if i < n:
                v = aabbs[leaf, comp]
            agg = qd_block.reduce(v, block, _bin_min_f64, qd.f64)
            if tid == 0:
                partials[block_id, comp] = agg
        for comp in qd.static(range(3, 6)):
            v = qd.f64(-1e32)
            if i < n:
                v = aabbs[leaf, comp]
            agg = qd_block.reduce(v, block, _bin_max_f64, qd.f64)
            if tid == 0:
                partials[block_id, comp] = agg


@qd.func
def _aabb_reduce_phase2(
    partials: qd.template(),
    aabbs: qd.template(),
    n_blocks_rt: qd.template(),
    block: qd.template(),
):
    """Phase 2: single block reduces ALL per-block partials into aabbs[0].

    Uses grid-stride accumulation so a single block of ``block`` threads
    can handle arbitrary n_blocks (not limited to one tile).  Each thread
    accumulates its stripe of partials, then block-reduces the accumulated
    values.
    """
    loop_config(block_dim=block)
    for i in range(block):
        qd_block.sync()
        nb = n_blocks_rt[0]

        for comp in qd.static(range(3)):
            v = qd.f64(1e32)
            j = i
            while j < nb:
                v = qd.min(v, partials[j, comp])
                j = j + block
            agg = qd_block.reduce(v, block, _bin_min_f64, qd.f64)
            if i == 0:
                aabbs[0, comp] = agg
        for comp in qd.static(range(3, 6)):
            v = qd.f64(-1e32)
            j = i
            while j < nb:
                v = qd.max(v, partials[j, comp])
                j = j + block
            agg = qd_block.reduce(v, block, _bin_max_f64, qd.f64)
            if i == 0:
                aabbs[0, comp] = agg


@qd.data_oriented
class LBVH:
    """Self-contained LBVH for *n* primitives (faces or edges).

    Tree layout (Karras 2012):
    - ``2n - 1`` nodes total; internals ``[0, n-1)``, leaves ``[n-1, 2n-1)``.
    - Root at index 0.
    - Leaf for sorted primitive *i* is at ``n - 1 + i``.
    - ``element_idx == 0xFFFFFFFF`` for internal nodes.
    """

    def __init__(self, n_prims: int, max_queries: int = 0) -> None:
        """Create LBVH for *n_prims* primitives.

        Args:
            n_prims: Number of primitives (faces or edges).
            max_queries: Max query count for stack_pool sizing.  For PT query
                this is n_surf_verts; for EE query this equals n_prims.
                If 0, defaults to n_prims.
        """
        assert n_prims > 0, "LBVH requires n_prims > 0"
        # Radix-sort pass geometry, the u32 "no node" marker and the reduction
        # block width. All fix unroll counts or block shapes, so none can be a
        # runtime device scalar. They are instance attributes rather than module
        # constants so their values reach device code as compile-time template
        # arguments and enter the fastcache key.
        self.sort_end_bit = 64
        self.sort_log256_max_n = 4
        self.sentinel = 0xFFFFFFFF
        self.bvh_block = 256

        if max_queries <= 0:
            max_queries = n_prims
        n_nodes = 2 * n_prims - 1
        padded = ((n_prims + 63) // 64) * 64
        _b = _LBVH_BACKEND

        self.n_prims_rt = qd.tensor(qd.i32, (1,), backend=_b)

        # --- Tree buffers ---
        self.aabbs = qd.tensor(qd.f64, (n_nodes, 6), backend=_b)
        self.temp_aabbs = qd.tensor(qd.f64, (n_prims, 6), backend=_b)
        self.indices = qd.tensor(qd.u32, (n_prims,), backend=_b)
        self.nodes_parent = qd.tensor(qd.u32, (n_nodes,), backend=_b)
        self.nodes_left = qd.tensor(qd.u32, (n_nodes,), backend=_b)
        self.nodes_right = qd.tensor(qd.u32, (n_nodes,), backend=_b)
        self.nodes_element = qd.tensor(qd.u32, (n_nodes,), backend=_b)
        self.flags = qd.tensor(qd.u32, (max(n_prims - 1, 1),), backend=_b)

        # --- Sort workspace ---
        sort_scratch = max(sort_scratch_slots(padded, self.sort_log256_max_n), 1)
        self.morton = qd.tensor(qd.u64, (padded,), backend=_b)
        self.morton_tmp = qd.tensor(qd.u64, (padded,), backend=_b)
        self.srt_perm = qd.tensor(qd.u32, (padded,), backend=_b)
        self.srt_tmp_perm = qd.tensor(qd.u32, (padded,), backend=_b)
        self.srt_scratch = qd.tensor(qd.u32, (sort_scratch,), backend=_b)
        self.srt_n = qd.tensor(qd.i32, (), backend=_b)

        # --- Scene AABB reduce workspace ---
        n_blocks = (n_prims + self.bvh_block - 1) // self.bvh_block
        self.red_partials = qd.tensor(qd.f64, (max(n_blocks, 1), 6), backend=_b)
        self.n_reduce_blocks_rt = qd.tensor(qd.i32, (1,), backend=_b)

        # --- BVH query stack workspace (per-thread) ---
        stack_rows = max(n_prims, max_queries)
        self.stack_pool = qd.tensor(qd.u32, (max(stack_rows, 1), _BVH_STACK_CAPACITY), backend=_b)

        # Host-only init data stored outside the instance dict so
        # @qd.data_oriented doesn't bake them as template primitives.
        _LBVH_INIT_DATA[id(self)] = (n_prims, n_blocks)

    def do_init(self) -> None:
        n_prims, n_blocks = _LBVH_INIT_DATA.pop(id(self))
        padded = ((n_prims + 63) // 64) * 64
        self.n_prims_rt.from_numpy(np.array([n_prims], dtype=np.int32))
        self.srt_n.from_numpy(np.array(padded, dtype=np.int32))
        self.n_reduce_blocks_rt.from_numpy(np.array([n_blocks], dtype=np.int32))

    # ======================================================================
    # BUILD pipeline
    # ======================================================================

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_tri(self, surf_mgr: qd.template(), vtx_mgr: qd.template()):
        """Compute swept leaf AABBs for triangles (stride=3).

        Matches cgq ``calc_leaf_aabb`` with stride=3.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf)
            for k in qd.static(range(3)):
                vi = surf_mgr.surf_triangles[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px, py, pz)
                dx = vtx_mgr.displacements[vi, 0]
                dy = vtx_mgr.displacements[vi, 1]
                dz = vtx_mgr.displacements[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px + dx, py + dy, pz + dz)

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_edge(self, surf_mgr: qd.template(), vtx_mgr: qd.template()):
        """Compute swept leaf AABBs for edges (stride=2).

        Matches cgq ``calc_leaf_aabb`` with stride=2.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf)
            for k in qd.static(range(2)):
                vi = surf_mgr.surf_edges[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px, py, pz)
                dx = vtx_mgr.displacements[vi, 0]
                dy = vtx_mgr.displacements[vi, 1]
                dz = vtx_mgr.displacements[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px + dx, py + dy, pz + dz)

    # --- Scene AABB reduce (cgq: cub::DeviceReduce::Reduce + AABBReduceOp) ---
    # Decomposed into 6 scalar reduce_min/reduce_max via qd.static unrolling.

    @qd.func(requires_top_level=True)
    def reduce_scene_aabb(self):
        """Parallel reduce of leaf AABBs into aabbs[0] (scene bounding box).

        Matches cgq ``cub::DeviceReduce::Reduce(AABBReduceOp)``.
        Two-phase block reduce: phase 1 (multi-block) writes per-block
        partials, phase 2 (single-block) reduces partials into aabbs[0].
        Total: 2 kernel launches (vs cgq's 1 CUB launch).
        """
        n_blocks = self.n_reduce_blocks_rt[0]
        total_threads = n_blocks * self.bvh_block
        _aabb_reduce_phase1(self.aabbs, self.red_partials, self.n_prims_rt, total_threads, self.bvh_block)
        _aabb_reduce_phase2(self.red_partials, self.aabbs, self.n_reduce_blocks_rt, self.bvh_block)

    @qd.func(requires_top_level=True)
    def calc_morton(self):
        """Compute Morton codes from leaf AABB centers, normalized to scene AABB.

        Matches cgq ``calc_morton``: each thread reads scene AABB independently.
        Padding slots filled with max u64 in the same loop.
        """
        padded_n = self.morton.shape[0]
        n = self.n_prims_rt[0]
        for idx in range(padded_n):
            if idx < n:
                scene_lx = self.aabbs[0, 0]
                scene_ly = self.aabbs[0, 1]
                scene_lz = self.aabbs[0, 2]
                scene_sx = self.aabbs[0, 3] - scene_lx
                scene_sy = self.aabbs[0, 4] - scene_ly
                scene_sz = self.aabbs[0, 5] - scene_lz
                if scene_sx < 1e-30:
                    scene_sx = 1e-30
                if scene_sy < 1e-30:
                    scene_sy = 1e-30
                if scene_sz < 1e-30:
                    scene_sz = 1e-30

                leaf = n - 1 + idx
                cx = (self.aabbs[leaf, 0] + self.aabbs[leaf, 3]) * 0.5
                cy = (self.aabbs[leaf, 1] + self.aabbs[leaf, 4]) * 0.5
                cz = (self.aabbs[leaf, 2] + self.aabbs[leaf, 5]) * 0.5
                nx = (cx - scene_lx) / scene_sx
                ny = (cy - scene_ly) / scene_sy
                nz = (cz - scene_lz) / scene_sz
                mc32 = morton_code_30bit(nx, ny, nz)
                self.morton[idx] = (qd.u64(mc32) << qd.u64(32)) | qd.u64(qd.u32(idx))
            else:
                self.morton[idx] = qd.u64(0xFFFFFFFFFFFFFFFF)

    @qd.func(requires_top_level=True)
    def sort_morton(self):
        """Radix sort Morton keys (u64, all 64 bits).

        Matches cgq ``cub::DeviceRadixSort::SortKeys``.
        """
        sort(
            self.morton,
            self.morton_tmp,
            self.srt_perm,
            self.srt_tmp_perm,
            self.srt_scratch,
            self.srt_n,
            qd.u64,
            True,
            self.sort_end_bit,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def extract_indices(self):
        """Extract original primitive index from lower 32 bits of sorted Morton keys.

        Matches cgq ``extract_indices``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            self.indices[idx] = qd.u32(self.morton[idx] & qd.u64(0xFFFFFFFF))

    @qd.func(requires_top_level=True)
    def copy_leaf_aabb_to_temp(self):
        """Copy leaf AABBs to temp before reorder.

        Matches cgq ``bvh_copy_leaf_aabb_kernel``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            for k in range(6):
                self.temp_aabbs[idx, k] = self.aabbs[leaf, k]

    @qd.func(requires_top_level=True)
    def reorder_leaf_aabb(self):
        """Reorder leaf AABBs to Morton-sorted order.

        Matches cgq ``reorder_leaf_aabb``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            src = qd.i32(self.indices[idx])
            for k in range(6):
                self.aabbs[leaf, k] = self.temp_aabbs[src, k]

    @qd.func(requires_top_level=True)
    def calc_leaf_nodes(self):
        """Initialize all BVH nodes: internals get sentinel, leaves get element_idx.

        Matches cgq ``calc_leaf_nodes``.
        """
        n = self.n_prims_rt[0]
        n_nodes = 2 * n - 1
        for idx in range(n_nodes):
            self.nodes_parent[idx] = qd.u32(self.sentinel)
            self.nodes_left[idx] = qd.u32(self.sentinel)
            self.nodes_right[idx] = qd.u32(self.sentinel)
            if idx < n - 1:
                self.nodes_element[idx] = qd.u32(self.sentinel)
            else:
                leaf_i = idx - (n - 1)
                self.nodes_element[idx] = self.indices[leaf_i]

    @qd.func(requires_top_level=True)
    def calc_internal_nodes(self):
        """Karras 2012 internal node construction.

        Matches cgq ``calc_internal_nodes``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n - 1):
            ij = determine_range(self.morton, n, idx)
            first = ij[0]
            last = ij[1]
            gamma = find_split(self.morton, n, first, last)

            left_child = gamma
            right_child = gamma + 1
            ij_min = qd.min(first, last)
            ij_max = qd.max(first, last)
            if ij_min == gamma:
                left_child = left_child + n - 1
            if ij_max == gamma + 1:
                right_child = right_child + n - 1

            self.nodes_left[idx] = qd.u32(left_child)
            self.nodes_right[idx] = qd.u32(right_child)
            self.nodes_parent[left_child] = qd.u32(idx)
            self.nodes_parent[right_child] = qd.u32(idx)

    @qd.func(requires_top_level=True)
    def memset_flags(self):
        """Reset flags to 0xFFFFFFFF sentinel for bottom-up AABB refit.

        Matches cgq ``bvh_memset_flags_kernel``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n - 1):
            self.flags[idx] = qd.u32(self.sentinel)

    @qd.func(requires_top_level=True)
    def calc_internal_aabb(self):
        """Bottom-up parallel AABB refit using atomicCAS on flags.

        Matches cgq ``calc_internal_aabb`` exactly:
        - flags initialized to 0xFFFFFFFF
        - atomicCAS(flags[parent], 0xFFFFFFFF, 0): first child returns, second merges
        - qd.simt.grid.mem_fence() after merge (= __threadfence())
        - walk terminates when parent == 0xFFFFFFFF (root's parent)
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            node_idx = idx + n - 1
            parent = self.nodes_parent[node_idx]
            while parent != qd.u32(self.sentinel):
                old = qd.atomic_cas(self.flags[qd.i32(parent)], qd.u32(self.sentinel), qd.u32(0))
                if old == qd.u32(self.sentinel):
                    parent = qd.u32(self.sentinel)
                else:
                    lidx = qd.i32(self.nodes_left[qd.i32(parent)])
                    ridx = qd.i32(self.nodes_right[qd.i32(parent)])
                    aabb_init(self.aabbs, qd.i32(parent))
                    aabb_combine_aabb(self.aabbs, qd.i32(parent), self.aabbs, lidx)
                    aabb_combine_aabb(self.aabbs, qd.i32(parent), self.aabbs, ridx)
                    qd.simt.grid.mem_fence()
                    parent = self.nodes_parent[qd.i32(parent)]

    # ======================================================================
    # TOY-MODE leaf AABB + query (matches cgq toy/lbvh.cu for ipctk tests)
    # Leaf AABBs expanded by d_hat; query AABB expanded by d_hat; plain aabb_overlap.
    # ======================================================================

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_tri_toy(self, surf_mgr: qd.template(), vtx_mgr: qd.template(), d_hat: qd.f64):
        """Toy-mode: leaf AABBs for triangles with d_hat expansion.

        Matches cgq ``toy/lbvh.cu::calc_leaf_aabb_face_kernel``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf)
            for k in qd.static(range(3)):
                vi = surf_mgr.surf_triangles[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px, py, pz)
            aabb_expand(self.aabbs, leaf, d_hat)

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_edge_toy(self, surf_mgr: qd.template(), vtx_mgr: qd.template(), d_hat: qd.f64):
        """Toy-mode: leaf AABBs for edges with d_hat expansion.

        Matches cgq ``toy/lbvh.cu::calc_leaf_aabb_edge_kernel``.
        """
        n = self.n_prims_rt[0]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf)
            for k in qd.static(range(2)):
                vi = surf_mgr.surf_edges[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(self.aabbs, leaf, px, py, pz)
            aabb_expand(self.aabbs, leaf, d_hat)

    @qd.func(requires_top_level=True)
    def query_pt_toy(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
    ):
        """Toy-mode PT query: query AABB expanded by d_hat + plain aabb_overlap.

        Matches cgq ``toy/lbvh.cu::query_pt_kernel``.
        """
        n_queries = surf_mgr.n_surf_verts_rt[0]

        for idx in range(n_queries):
            vidx = surf_mgr.surf_verts[idx]
            vx = vtx_mgr.positions[vidx, 0]
            vy = vtx_mgr.positions[vidx, 1]
            vz = vtx_mgr.positions[vidx, 2]

            q_lx = vx - d_hat
            q_ly = vy - d_hat
            q_lz = vz - d_hat
            q_ux = vx + d_hat
            q_uy = vy + d_hat
            q_uz = vz + d_hat

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child — plain aabb_overlap
                L_overlap = qd.i32(1)
                if self.aabbs[L_idx, 0] > q_ux:
                    L_overlap = 0
                if q_lx > self.aabbs[L_idx, 3]:
                    L_overlap = 0
                if self.aabbs[L_idx, 1] > q_uy:
                    L_overlap = 0
                if q_ly > self.aabbs[L_idx, 4]:
                    L_overlap = 0
                if self.aabbs[L_idx, 2] > q_uz:
                    L_overlap = 0
                if q_lz > self.aabbs[L_idx, 5]:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx = qd.i32(L_elem)
                        bi = vtx_mgr.body_id[vidx]
                        fv0 = surf_mgr.surf_triangles[face_idx, 0]
                        fv1 = surf_mgr.surf_triangles[face_idx, 1]
                        fv2 = surf_mgr.surf_triangles[face_idx, 2]
                        bj = vtx_mgr.body_id[fv0]
                        accept = qd.i32(1)
                        if bi == bj:
                            if bi >= 0:
                                accept = 0
                        if vidx == fv0:
                            accept = 0
                        if vidx == fv1:
                            accept = 0
                        if vidx == fv2:
                            accept = 0
                        if accept != 0:
                            cp_idx = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx < max_pairs_val:
                                pairs[cp_idx, 0] = idx
                                pairs[cp_idx, 1] = face_idx

                # Process right child — plain aabb_overlap
                R_overlap = qd.i32(1)
                if self.aabbs[R_idx, 0] > q_ux:
                    R_overlap = 0
                if q_lx > self.aabbs[R_idx, 3]:
                    R_overlap = 0
                if self.aabbs[R_idx, 1] > q_uy:
                    R_overlap = 0
                if q_ly > self.aabbs[R_idx, 4]:
                    R_overlap = 0
                if self.aabbs[R_idx, 2] > q_uz:
                    R_overlap = 0
                if q_lz > self.aabbs[R_idx, 5]:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx2 = qd.i32(R_elem)
                        bi2 = vtx_mgr.body_id[vidx]
                        fv02 = surf_mgr.surf_triangles[face_idx2, 0]
                        fv12 = surf_mgr.surf_triangles[face_idx2, 1]
                        fv22 = surf_mgr.surf_triangles[face_idx2, 2]
                        bj2 = vtx_mgr.body_id[fv02]
                        accept2 = qd.i32(1)
                        if bi2 == bj2:
                            if bi2 >= 0:
                                accept2 = 0
                        if vidx == fv02:
                            accept2 = 0
                        if vidx == fv12:
                            accept2 = 0
                        if vidx == fv22:
                            accept2 = 0
                        if accept2 != 0:
                            cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx2 < max_pairs_val:
                                pairs[cp_idx2, 0] = idx
                                pairs[cp_idx2, 1] = face_idx2

    @qd.func(requires_top_level=True)
    def query_ee_toy(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
    ):
        """Toy-mode EE self-query: leaf AABB already expanded, plain aabb_overlap.

        Matches cgq ``toy/lbvh.cu::query_ee_kernel``.
        """
        n = self.n_prims_rt[0]

        for idx in range(n):
            leaf_idx = idx + n - 1
            self_eid = self.nodes_element[leaf_idx]

            q_lx = self.aabbs[leaf_idx, 0]
            q_ly = self.aabbs[leaf_idx, 1]
            q_lz = self.aabbs[leaf_idx, 2]
            q_ux = self.aabbs[leaf_idx, 3]
            q_uy = self.aabbs[leaf_idx, 4]
            q_uz = self.aabbs[leaf_idx, 5]

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child — plain aabb_overlap
                L_overlap = qd.i32(1)
                if self.aabbs[L_idx, 0] > q_ux:
                    L_overlap = 0
                if q_lx > self.aabbs[L_idx, 3]:
                    L_overlap = 0
                if self.aabbs[L_idx, 1] > q_uy:
                    L_overlap = 0
                if q_ly > self.aabbs[L_idx, 4]:
                    L_overlap = 0
                if self.aabbs[L_idx, 2] > q_uz:
                    L_overlap = 0
                if q_lz > self.aabbs[L_idx, 5]:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx = L_elem
                        if obj_idx > self_eid:
                            ea0 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea1 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb0 = surf_mgr.surf_edges[qd.i32(obj_idx), 0]
                            eb1 = surf_mgr.surf_edges[qd.i32(obj_idx), 1]
                            bi = vtx_mgr.body_id[ea0]
                            bj = vtx_mgr.body_id[eb0]
                            accept = qd.i32(1)
                            if bi == bj:
                                if bi >= 0:
                                    accept = 0
                            if ea0 == eb0:
                                accept = 0
                            if ea0 == eb1:
                                accept = 0
                            if ea1 == eb0:
                                accept = 0
                            if ea1 == eb1:
                                accept = 0
                            if accept != 0:
                                cp_idx = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx < max_pairs_val:
                                    pairs[cp_idx, 0] = qd.i32(self_eid)
                                    pairs[cp_idx, 1] = qd.i32(obj_idx)

                # Process right child — plain aabb_overlap
                R_overlap = qd.i32(1)
                if self.aabbs[R_idx, 0] > q_ux:
                    R_overlap = 0
                if q_lx > self.aabbs[R_idx, 3]:
                    R_overlap = 0
                if self.aabbs[R_idx, 1] > q_uy:
                    R_overlap = 0
                if q_ly > self.aabbs[R_idx, 4]:
                    R_overlap = 0
                if self.aabbs[R_idx, 2] > q_uz:
                    R_overlap = 0
                if q_lz > self.aabbs[R_idx, 5]:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx2 = R_elem
                        if obj_idx2 > self_eid:
                            ea02 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea12 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb02 = surf_mgr.surf_edges[qd.i32(obj_idx2), 0]
                            eb12 = surf_mgr.surf_edges[qd.i32(obj_idx2), 1]
                            bi2 = vtx_mgr.body_id[ea02]
                            bj2 = vtx_mgr.body_id[eb02]
                            accept2 = qd.i32(1)
                            if bi2 == bj2:
                                if bi2 >= 0:
                                    accept2 = 0
                            if ea02 == eb02:
                                accept2 = 0
                            if ea02 == eb12:
                                accept2 = 0
                            if ea12 == eb02:
                                accept2 = 0
                            if ea12 == eb12:
                                accept2 = 0
                            if accept2 != 0:
                                cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx2 < max_pairs_val:
                                    pairs[cp_idx2, 0] = qd.i32(self_eid)
                                    pairs[cp_idx2, 1] = qd.i32(obj_idx2)

    # ======================================================================
    # Full build sequence (convenience, calls all steps in order)
    # Note: query_ee_toy above has no overflow_flag (harness-only path)
    # ======================================================================

    def build_tri(self, surf_mgr, vtx_mgr):
        """Full BVH build for triangles (call from ``_step_kernel`` top level)."""
        self.calc_leaf_aabb_tri(surf_mgr, vtx_mgr)
        self.reduce_scene_aabb()
        self.calc_morton()
        self.sort_morton()
        self.extract_indices()
        self.copy_leaf_aabb_to_temp()
        self.reorder_leaf_aabb()
        self.calc_leaf_nodes()
        self.calc_internal_nodes()
        self.memset_flags()
        self.calc_internal_aabb()

    def build_edge(self, surf_mgr, vtx_mgr):
        """Full BVH build for edges (call from ``_step_kernel`` top level)."""
        self.calc_leaf_aabb_edge(surf_mgr, vtx_mgr)
        self.reduce_scene_aabb()
        self.calc_morton()
        self.sort_morton()
        self.extract_indices()
        self.copy_leaf_aabb_to_temp()
        self.reorder_leaf_aabb()
        self.calc_leaf_nodes()
        self.calc_internal_nodes()
        self.memset_flags()
        self.calc_internal_aabb()

    # ======================================================================
    # QUERY: PT broadphase (surface vertices vs triangle BVH)
    # ======================================================================

    @qd.func(requires_top_level=True)
    def query_pt(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
        overflow_flag: qd.template(),
    ):
        """PT swept broadphase: each surface vertex queries the triangle BVH.

        Matches cgq ``query_pt_swept``.  Sets ``overflow_flag`` to 1 when the
        candidate count exceeds ``max_pairs_val``.
        Output pairs: ``(surf_vert_idx, face_idx)``.
        """
        n_queries = surf_mgr.n_surf_verts_rt[0]

        for idx in range(n_queries):
            vidx = surf_mgr.surf_verts[idx]
            vx = vtx_mgr.positions[vidx, 0]
            vy = vtx_mgr.positions[vidx, 1]
            vz = vtx_mgr.positions[vidx, 2]
            dx = vtx_mgr.displacements[vidx, 0]
            dy = vtx_mgr.displacements[vidx, 1]
            dz = vtx_mgr.displacements[vidx, 2]

            q_lx = qd.min(vx, vx + dx)
            q_ly = qd.min(vy, vy + dy)
            q_lz = qd.min(vz, vz + dz)
            q_ux = qd.max(vx, vx + dx)
            q_uy = qd.max(vy, vy + dy)
            q_uz = qd.max(vz, vz + dz)

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child
                L_lo_x = self.aabbs[L_idx, 0]
                L_lo_y = self.aabbs[L_idx, 1]
                L_lo_z = self.aabbs[L_idx, 2]
                L_hi_x = self.aabbs[L_idx, 3]
                L_hi_y = self.aabbs[L_idx, 4]
                L_hi_z = self.aabbs[L_idx, 5]
                L_overlap = qd.i32(1)
                if (L_lo_x - q_ux) >= d_hat:
                    L_overlap = 0
                if (q_lx - L_hi_x) >= d_hat:
                    L_overlap = 0
                if (L_lo_y - q_uy) >= d_hat:
                    L_overlap = 0
                if (q_ly - L_hi_y) >= d_hat:
                    L_overlap = 0
                if (L_lo_z - q_uz) >= d_hat:
                    L_overlap = 0
                if (q_lz - L_hi_z) >= d_hat:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx = qd.i32(L_elem)
                        bi = vtx_mgr.body_id[vidx]
                        fv0 = surf_mgr.surf_triangles[face_idx, 0]
                        fv1 = surf_mgr.surf_triangles[face_idx, 1]
                        fv2 = surf_mgr.surf_triangles[face_idx, 2]
                        bj = vtx_mgr.body_id[fv0]
                        accept = qd.i32(1)
                        if bi == bj:
                            if bi >= 0:
                                accept = 0
                        if vidx == fv0:
                            accept = 0
                        if vidx == fv1:
                            accept = 0
                        if vidx == fv2:
                            accept = 0
                        if accept != 0:
                            cp_idx = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx < max_pairs_val:
                                pairs[cp_idx, 0] = idx
                                pairs[cp_idx, 1] = face_idx
                            else:
                                overflow_flag[()] = 1

                # Process right child
                R_lo_x = self.aabbs[R_idx, 0]
                R_lo_y = self.aabbs[R_idx, 1]
                R_lo_z = self.aabbs[R_idx, 2]
                R_hi_x = self.aabbs[R_idx, 3]
                R_hi_y = self.aabbs[R_idx, 4]
                R_hi_z = self.aabbs[R_idx, 5]
                R_overlap = qd.i32(1)
                if (R_lo_x - q_ux) >= d_hat:
                    R_overlap = 0
                if (q_lx - R_hi_x) >= d_hat:
                    R_overlap = 0
                if (R_lo_y - q_uy) >= d_hat:
                    R_overlap = 0
                if (q_ly - R_hi_y) >= d_hat:
                    R_overlap = 0
                if (R_lo_z - q_uz) >= d_hat:
                    R_overlap = 0
                if (q_lz - R_hi_z) >= d_hat:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx2 = qd.i32(R_elem)
                        bi2 = vtx_mgr.body_id[vidx]
                        fv02 = surf_mgr.surf_triangles[face_idx2, 0]
                        fv12 = surf_mgr.surf_triangles[face_idx2, 1]
                        fv22 = surf_mgr.surf_triangles[face_idx2, 2]
                        bj2 = vtx_mgr.body_id[fv02]
                        accept2 = qd.i32(1)
                        if bi2 == bj2:
                            if bi2 >= 0:
                                accept2 = 0
                        if vidx == fv02:
                            accept2 = 0
                        if vidx == fv12:
                            accept2 = 0
                        if vidx == fv22:
                            accept2 = 0
                        if accept2 != 0:
                            cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx2 < max_pairs_val:
                                pairs[cp_idx2, 0] = idx
                                pairs[cp_idx2, 1] = face_idx2
                            else:
                                overflow_flag[()] = 1

    # ======================================================================
    # QUERY: EE self-query (edge vs edge BVH)
    # ======================================================================

    @qd.func(requires_top_level=True)
    def query_ee(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
        overflow_flag: qd.template(),
    ):
        """EE self-query: each edge queries the edge BVH.

        Matches cgq ``query_ee``.  Sets ``overflow_flag`` to 1 when the
        candidate count exceeds ``max_pairs_val``.
        Output pairs: ``(edge_i, edge_j)`` with ``edge_i < edge_j``.
        """
        n = self.n_prims_rt[0]

        for idx in range(n):
            leaf_idx = idx + n - 1
            self_eid = self.nodes_element[leaf_idx]

            q_lx = self.aabbs[leaf_idx, 0]
            q_ly = self.aabbs[leaf_idx, 1]
            q_lz = self.aabbs[leaf_idx, 2]
            q_ux = self.aabbs[leaf_idx, 3]
            q_uy = self.aabbs[leaf_idx, 4]
            q_uz = self.aabbs[leaf_idx, 5]

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child
                L_lo_x = self.aabbs[L_idx, 0]
                L_lo_y = self.aabbs[L_idx, 1]
                L_lo_z = self.aabbs[L_idx, 2]
                L_hi_x = self.aabbs[L_idx, 3]
                L_hi_y = self.aabbs[L_idx, 4]
                L_hi_z = self.aabbs[L_idx, 5]
                L_overlap = qd.i32(1)
                if (L_lo_x - q_ux) >= d_hat:
                    L_overlap = 0
                if (q_lx - L_hi_x) >= d_hat:
                    L_overlap = 0
                if (L_lo_y - q_uy) >= d_hat:
                    L_overlap = 0
                if (q_ly - L_hi_y) >= d_hat:
                    L_overlap = 0
                if (L_lo_z - q_uz) >= d_hat:
                    L_overlap = 0
                if (q_lz - L_hi_z) >= d_hat:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx = L_elem
                        if obj_idx > self_eid:
                            ea0 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea1 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb0 = surf_mgr.surf_edges[qd.i32(obj_idx), 0]
                            eb1 = surf_mgr.surf_edges[qd.i32(obj_idx), 1]
                            bi = vtx_mgr.body_id[ea0]
                            bj = vtx_mgr.body_id[eb0]
                            accept = qd.i32(1)
                            if bi == bj:
                                if bi >= 0:
                                    accept = 0
                            if ea0 == eb0:
                                accept = 0
                            if ea0 == eb1:
                                accept = 0
                            if ea1 == eb0:
                                accept = 0
                            if ea1 == eb1:
                                accept = 0
                            if accept != 0:
                                cp_idx = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx < max_pairs_val:
                                    pairs[cp_idx, 0] = qd.i32(self_eid)
                                    pairs[cp_idx, 1] = qd.i32(obj_idx)
                                else:
                                    overflow_flag[()] = 1

                # Process right child
                R_lo_x = self.aabbs[R_idx, 0]
                R_lo_y = self.aabbs[R_idx, 1]
                R_lo_z = self.aabbs[R_idx, 2]
                R_hi_x = self.aabbs[R_idx, 3]
                R_hi_y = self.aabbs[R_idx, 4]
                R_hi_z = self.aabbs[R_idx, 5]
                R_overlap = qd.i32(1)
                if (R_lo_x - q_ux) >= d_hat:
                    R_overlap = 0
                if (q_lx - R_hi_x) >= d_hat:
                    R_overlap = 0
                if (R_lo_y - q_uy) >= d_hat:
                    R_overlap = 0
                if (q_ly - R_hi_y) >= d_hat:
                    R_overlap = 0
                if (R_lo_z - q_uz) >= d_hat:
                    R_overlap = 0
                if (q_lz - R_hi_z) >= d_hat:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx2 = R_elem
                        if obj_idx2 > self_eid:
                            ea02 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea12 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb02 = surf_mgr.surf_edges[qd.i32(obj_idx2), 0]
                            eb12 = surf_mgr.surf_edges[qd.i32(obj_idx2), 1]
                            bi2 = vtx_mgr.body_id[ea02]
                            bj2 = vtx_mgr.body_id[eb02]
                            accept2 = qd.i32(1)
                            if bi2 == bj2:
                                if bi2 >= 0:
                                    accept2 = 0
                            if ea02 == eb02:
                                accept2 = 0
                            if ea02 == eb12:
                                accept2 = 0
                            if ea12 == eb02:
                                accept2 = 0
                            if ea12 == eb12:
                                accept2 = 0
                            if accept2 != 0:
                                cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx2 < max_pairs_val:
                                    pairs[cp_idx2, 0] = qd.i32(self_eid)
                                    pairs[cp_idx2, 1] = qd.i32(obj_idx2)
                                else:
                                    overflow_flag[()] = 1
