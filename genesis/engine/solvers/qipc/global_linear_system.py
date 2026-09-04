"""GlobalLinearSystem -- BCOO triplet storage, sort+reduce, and solution vectors.

Mirrors cgq ``GlobalLinearSystem`` + ``GlobalLinearSystemContext``.
Owns triplet input buffers, radix sort workspace, FSR (fast segmented reduce)
workspace, unique BCOO output, plus RHS gradient (b_rhs) and solution (x_sol).
"""

from __future__ import annotations

import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    sort,
    sort_scratch_slots,
)

from genesis.engine.solvers.qipc.sim_system import SimSystem


@qd.data_oriented
class GlobalLinearSystem(SimSystem):
    """BCOO block sparse matrix (3x3 blocks) with sort+reduce pipeline.

    The triplet buffer is partitioned into a fixed-size elastic region
    ``[0, n_elastic)`` and a contact region ``[n_elastic, capacity)``. The
    elastic region is topology-determined, so its extent is known at build
    time; the contact region's extent is not, because it is
    ``16 * n_unique_triplets`` and the unique-triplet count only exists on the
    device after narrowphase. The sort+reduce pipeline therefore runs over a
    *live* count ``n_live_rt`` that the graph itself computes each Newton
    iteration (:meth:`set_live_from_contact`), mirroring how cgq derives the
    live extent from ``n_elastic_device_view()`` plus a category offset.

    Sort+reduce then merges duplicate ``(row, col)`` entries into unique BCOO.
    A slot that a distribute pass decides contributes nothing is *discarded*
    rather than skipped: it is written as a zero block at global ``(0, 0)``, so
    the reduce merges it away and no downstream pass needs to know which slots
    are dead (cgq ``TripletMatrixView::discard``).
    """

    def __init__(
        self,
        n_block_rows: int,
        n_elastic_triplets: int,
        n_contact_triplets: int = 0,
    ) -> None:
        super().__init__()
        # Radix-sort / scan pass geometry. These fix `qd.static` unroll counts in
        # the sort and scan device funcs, so they cannot be runtime device
        # scalars. They are instance attributes rather than module constants so
        # that their values reach device code as compile-time template arguments
        # and enter the fastcache key -- a module global would neither recompile
        # nor invalidate the cache when edited.
        self.sort_end_bit = 64
        self.sort_log256_max_n = 4
        self.scan_log256_max_n = 4

        self.n_block_rows = n_block_rows
        self.ndof = n_block_rows * 3
        self.n_elastic = n_elastic_triplets
        self.n_contact_cap = n_contact_triplets
        self.capacity = n_elastic_triplets + n_contact_triplets
        max_tri = self.capacity
        padded = ((max_tri + 63) // 64) * 64

        # Runtime scalars for loop bounds (avoids recompilation)
        self.ndof_rt = qd.ndarray(qd.i32, shape=(1,))
        self.capacity_rt = qd.ndarray(qd.i32, shape=(1,))
        self.n_elastic_rt = qd.ndarray(qd.i32, shape=(1,))
        # Live triplet extent for this Newton iteration: elastic + contact.
        # Recomputed inside the graph, so it is a device scalar, not a host int.
        self.n_live_rt = qd.ndarray(qd.i32, shape=(1,))
        self.padded_live_rt = qd.ndarray(qd.i32, shape=(1,))

        # Overflow detection (GPU sets to 1 when triplet count > capacity)
        self.triplet_overflow = qd.ndarray(qd.i32, shape=())

        # --- Triplet input ---
        self.tri_row = qd.ndarray(qd.i32, shape=(max_tri,))
        self.tri_col = qd.ndarray(qd.i32, shape=(max_tri,))
        self.tri_val = qd.ndarray(qd.f64, shape=(max_tri * 9,))

        # --- Sort workspace (u64 composite keys: row<<32|col) ---
        sort_scratch = max(sort_scratch_slots(padded, self.sort_log256_max_n), 1)
        self.srt_keys = qd.ndarray(qd.u64, shape=(padded,))
        self.srt_tmp_keys = qd.ndarray(qd.u64, shape=(padded,))
        self.srt_perm = qd.ndarray(qd.u32, shape=(padded,))
        self.srt_tmp_perm = qd.ndarray(qd.u32, shape=(padded,))
        self.srt_scratch = qd.ndarray(qd.u32, shape=(sort_scratch,))
        self.srt_n = qd.ndarray(qd.i32, shape=())

        # --- FSR workspace ---
        self.seg_flags = qd.ndarray(qd.u32, shape=(padded,))
        scan_scratch = max(exclusive_scan_scratch_slots(padded, self.scan_log256_max_n), 1)
        self.seg_ids = qd.ndarray(qd.u32, shape=(padded,))
        self.scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch,))

        # --- Unique BCOO output ---
        self.bcoo_row = qd.ndarray(qd.i32, shape=(max_tri,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(max_tri,))
        self.bcoo_val = qd.ndarray(qd.f64, shape=(max_tri * 9,))
        self.bcoo_nnz = qd.ndarray(qd.i32, shape=())

        # --- RHS + solution ---
        self.b_rhs = qd.ndarray(qd.f64, shape=(self.ndof,))
        self.x_sol = qd.ndarray(qd.f64, shape=(self.ndof,))

    def do_build(self) -> None:
        import numpy as np

        self.ndof_rt.from_numpy(np.array([self.ndof], dtype=np.int32))
        self.capacity_rt.from_numpy(np.array([self.capacity], dtype=np.int32))
        self.n_elastic_rt.from_numpy(np.array([self.n_elastic], dtype=np.int32))
        # Elastic-only until a narrowphase raises it; a contact-free scene never
        # touches these again.
        self.n_live_rt.from_numpy(np.array([self.n_elastic], dtype=np.int32))
        padded = ((self.n_elastic + 63) // 64) * 64
        self.padded_live_rt.from_numpy(np.array([padded], dtype=np.int32))
        self.srt_n.from_numpy(np.array(self.n_elastic, dtype=np.int32))

    def realloc_triplet_buffers(self, new_cap: int) -> None:
        """Grow triplet + sort + BCOO buffers. Mirrors cgq realloc_triplet_buffers.

        ``new_cap`` is the total capacity (elastic + contact); the elastic
        region keeps its base at 0 and its size, so only the contact region
        grows.
        """
        import numpy as np

        if new_cap < self.n_elastic:
            raise ValueError(f"realloc_triplet_buffers: capacity {new_cap} below elastic region {self.n_elastic}")
        padded = ((new_cap + 63) // 64) * 64
        self.tri_row = qd.ndarray(qd.i32, shape=(new_cap,))
        self.tri_col = qd.ndarray(qd.i32, shape=(new_cap,))
        self.tri_val = qd.ndarray(qd.f64, shape=(new_cap * 9,))

        sort_scratch = max(sort_scratch_slots(padded, self.sort_log256_max_n), 1)
        self.srt_keys = qd.ndarray(qd.u64, shape=(padded,))
        self.srt_tmp_keys = qd.ndarray(qd.u64, shape=(padded,))
        self.srt_perm = qd.ndarray(qd.u32, shape=(padded,))
        self.srt_tmp_perm = qd.ndarray(qd.u32, shape=(padded,))
        self.srt_scratch = qd.ndarray(qd.u32, shape=(sort_scratch,))

        self.seg_flags = qd.ndarray(qd.u32, shape=(padded,))
        scan_scratch = max(exclusive_scan_scratch_slots(padded, self.scan_log256_max_n), 1)
        self.seg_ids = qd.ndarray(qd.u32, shape=(padded,))
        self.scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch,))

        self.bcoo_row = qd.ndarray(qd.i32, shape=(new_cap,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(new_cap,))
        self.bcoo_val = qd.ndarray(qd.f64, shape=(new_cap * 9,))

        self.capacity = new_cap
        self.n_contact_cap = new_cap - self.n_elastic
        self.capacity_rt.from_numpy(np.array([new_cap], dtype=np.int32))
        self.padded_live_rt.from_numpy(np.array([padded], dtype=np.int32))

    # ------------------------------------------------------------------
    # Graph-kernel methods
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def zero_rhs(self):
        """Zero the gradient vector and triplet values before assembly.

        Runs over the whole capacity rather than the live extent: it executes
        before narrowphase, so the live extent for this iteration is not known
        yet.
        """
        for i in range(self.ndof_rt[0]):
            self.b_rhs[i] = qd.f64(0.0)
        for i in range(self.capacity_rt[0] * 9):
            self.tri_val[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def set_live_elastic_only(self):
        """Set the live triplet extent to the elastic region alone."""
        for _ in range(1):
            live = self.n_elastic_rt[0]
            self.n_live_rt[0] = live
            self.padded_live_rt[0] = ((live + 63) // 64) * 64
            self.srt_n[()] = live

    @qd.func(requires_top_level=True)
    def set_live_from_contact(self, csys: qd.template()):
        """Extend the live triplet extent to cover this iteration's contact blocks.

        Each unique contact triplet expands to 16 body-DOF blocks (a 4x4 grid
        of 3x3 blocks for the 12-DOF row body against the 12-DOF column body).
        On overflow the exact desired extent is preserved and
        ``triplet_overflow`` is raised. The host grows to that demand and
        resumes at the following solve checkpoint before any writer runs.
        """
        for _ in range(1):
            live = self.n_elastic_rt[0] + csys.n_unique_triplets[()] * 16
            if live > self.capacity_rt[0]:
                self.triplet_overflow[()] = 1
            self.n_live_rt[0] = live
            self.padded_live_rt[0] = ((live + 63) // 64) * 64
            self.srt_n[()] = live

    @qd.func(requires_top_level=True)
    def sort_seed(self):
        """Seed sort keys = composite (row<<32|col) as u64, perm = original index.

        Pads unused slots to max u64 so they sort to the end.
        """
        for k in range(self.padded_live_rt[0]):
            if k < self.n_live_rt[0]:
                r = qd.u64(self.tri_row[k])
                c = qd.u64(self.tri_col[k])
                self.srt_keys[k] = (r << 32) | c
                self.srt_perm[k] = qd.u32(k)
            else:
                self.srt_keys[k] = qd.u64(0xFFFFFFFFFFFFFFFF)
                self.srt_perm[k] = qd.u32(k)

    @qd.func(requires_top_level=True)
    def sort_radix(self):
        """Radix sort the composite u64 keys."""
        sort(
            self.srt_keys,
            self.srt_tmp_keys,
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
    def sort_segment_flags(self):
        """Compute segment tail flags: 1 where key changes, 0 otherwise."""
        for k in range(self.padded_live_rt[0]):
            if k < self.n_live_rt[0]:
                if k == self.n_live_rt[0] - 1:
                    self.seg_flags[k] = qd.u32(1)
                else:
                    k1 = self.srt_keys[k]
                    k2 = self.srt_keys[k + 1]
                    if k1 != k2:
                        self.seg_flags[k] = qd.u32(1)
                    else:
                        self.seg_flags[k] = qd.u32(0)
            else:
                self.seg_flags[k] = qd.u32(0)

    @qd.func(requires_top_level=True)
    def sort_scan(self):
        """Exclusive scan on segment flags -> segment IDs."""
        exclusive_scan_add(
            self.seg_flags,
            self.seg_ids,
            self.scan_scratch,
            self.n_live_rt[0],
            qd.u32,
            self.scan_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def sort_zero_bcoo(self):
        """Zero BCOO output values before merge-scatter."""
        for i in range(self.n_live_rt[0] * 9):
            self.bcoo_val[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def sort_fsr_merge(self):
        """Fast segmented reduce: sum 3x3 blocks by segment ID into BCOO."""
        for k in range(self.n_live_rt[0]):
            src = qd.i32(self.srt_perm[k])
            seg = qd.i32(self.seg_ids[k])
            for c in range(9):
                qd.atomic_add(self.bcoo_val[seg * 9 + c], self.tri_val[src * 9 + c])

    @qd.func(requires_top_level=True)
    def sort_extract_unique(self):
        """Extract unique (row, col) from tail-flag positions, record nnz."""
        for k in range(self.n_live_rt[0]):
            if self.seg_flags[k] == qd.u32(1):
                seg = qd.i32(self.seg_ids[k])
                key = self.srt_keys[k]
                self.bcoo_row[seg] = qd.i32(key >> 32)
                self.bcoo_col[seg] = qd.i32(key & qd.u64(0xFFFFFFFF))
                if k == self.n_live_rt[0] - 1:
                    self.bcoo_nnz[()] = seg + 1
