"""ContactSystem -- growable contact buffers for narrowphase + assembly.

Mirrors cgq ``ContactSystem`` / ``GlobalContactContext``.  Owns:
- Collision pair buffers (PT, EE, PH) — written by BVH query
- Contact doublet/triplet buffers — written by fused filter+assemble
- Sort workspace for doublet/triplet dedup
- Unique output buffers (post sort+reduce)
- CCD alpha + partial reduction buffers
- Barrier energy + partial reduction buffers
- Overflow flags for pair query and assembly capacity

Capacity formulas from cgq:
  ``doublets = total_pairs * 4``  (8 with friction)
  ``triplets = total_pairs * 10`` (20 with friction)

Growth factor: 1.2x on overflow.

Device-scalar convention: every dynamic value reachable from device code lives
as a ``qd.ndarray`` on ``self`` using the cgq canonical name — no ``_rt``
suffix, no Python host mirror.  Only ``qd.static``-consumed flags and
structural invariants (sort digit count, halfplane count, barrier formulation
selectors) remain as plain instance attributes.
"""

from __future__ import annotations

import math

import numpy as np
import quadrants as qd
from quadrants.algorithms import exclusive_scan_add, sort, sort_scratch_slots

from genesis.engine.solvers.qipc.sim_system import SimSystem

_GROW_FACTOR = 1.2
_CCD_BLOCK_DIM = 256
_CONTACT_BLOCK_DIM = 256


def _padded64(n: int) -> int:
    return ((n + 63) // 64) * 64


@qd.data_oriented
class ContactSystem(SimSystem):
    """Scene-wide contact buffers with overflow/realloc support.

    Construction allocates all buffers and uploads device scalars immediately.
    ``do_build()`` is a no-op kept for lifecycle compatibility.

    On overflow, ``realloc_pair_buffers`` / ``realloc_assembly_buffers`` grow
    the relevant buffers.  Because ndarray members are replaced on the same
    ``@qd.data_oriented`` instance, the graph kernel picks up the new
    pointers without recompilation.
    """

    def __init__(
        self,
        max_pairs_pt: int,
        max_pairs_ee: int,
        *,
        d_hat: float = 0.01,
        kappa: float = 1e5,
        gass_threshold: float = 1e-6,
        eps_x_coeff: float = 1e-3,
        n_verts: int = 0,
        dt: float = 1.0,
        barrier: str = "consistent_ipc",
        n_halfplanes: int = 0,
        max_pairs_ph: int | None = None,
        n_codim_verts: int = 0,
        ccd_eta: float = 0.2,
    ) -> None:
        super().__init__()

        # ---- compile-time structural invariants (enter fastcache key) ----

        self.sort_log256_max_n = 4
        self.ccd_max_iters = 50000

        if barrier not in ("consistent_ipc", "gipc_rank2"):
            raise ValueError(f"unknown barrier {barrier!r}: expected 'consistent_ipc' or 'gipc_rank2'")
        self.use_area_weight = 1 if barrier == "consistent_ipc" else 0
        self.use_mollifier = 1 if barrier == "consistent_ipc" else 0
        self.use_cipc_distance_flag = 1 if barrier == "consistent_ipc" else 0

        n_hp = max(int(n_halfplanes), 0)
        self.n_halfplanes = n_hp
        self.has_halfplanes = 1 if n_hp > 0 else 0
        self.hp_off = max(n_verts - n_hp, 0)

        self.has_codim = 0
        if n_codim_verts > 0:
            raise NotImplementedError(
                f"PE/PP codim contact: pending migration (got n_codim_verts={n_codim_verts}), "
                "see docs/design/road-map.md M6"
            )

        # ---- local capacity computation (not stored as host mirrors) ----

        cap_pt = max(max_pairs_pt, 1)
        cap_ee = max(max_pairs_ee, 1)
        cap_ph = max(max_pairs_ph if max_pairs_ph is not None else 1, 1) if self.has_halfplanes else 0

        mesh_pairs = cap_pt + cap_ee
        cap_doublets = max(mesh_pairs * 4 + cap_ph, 64)
        cap_triplets = max(mesh_pairs * 10 + cap_ph, 64)
        total_pairs = mesh_pairs + cap_ph

        d_hat_sq = d_hat * d_hat
        dt_sq = dt * dt
        gass_ln = math.log(gass_threshold)
        gass_a = gass_ln + (gass_threshold - 1.0) / gass_threshold

        # ---- device scalars (canonical cgq names, no _rt suffix) ----

        self.max_pairs_pt = qd.ndarray(qd.i32, shape=(1,))
        self.max_pairs_ee = qd.ndarray(qd.i32, shape=(1,))
        self.max_pairs_ph = qd.ndarray(qd.i32, shape=(1,))
        self.max_contact_doublets = qd.ndarray(qd.i32, shape=(1,))
        self.max_contact_triplets = qd.ndarray(qd.i32, shape=(1,))
        self.n_verts = qd.ndarray(qd.i32, shape=(1,))

        self.d_hat = qd.ndarray(qd.f64, shape=(1,))
        self.d_hat_sq = qd.ndarray(qd.f64, shape=(1,))
        self.dt_sq = qd.ndarray(qd.f64, shape=(1,))
        self.kappa = qd.ndarray(qd.f64, shape=(1,))
        self.gass_t = qd.ndarray(qd.f64, shape=(1,))
        self.gass_ln = qd.ndarray(qd.f64, shape=(1,))
        self.gass_a = qd.ndarray(qd.f64, shape=(1,))
        self.eps_x_coeff = qd.ndarray(qd.f64, shape=(1,))

        padded_d = _padded64(cap_doublets)
        padded_t = _padded64(cap_triplets)
        self.cd_padded = qd.ndarray(qd.i32, shape=(1,))
        self.ct_padded = qd.ndarray(qd.i32, shape=(1,))
        self.ccd_eta = qd.ndarray(qd.f64, shape=(1,))

        # Upload device scalars immediately
        self.max_pairs_pt.from_numpy(np.array([cap_pt], dtype=np.int32))
        self.max_pairs_ee.from_numpy(np.array([cap_ee], dtype=np.int32))
        self.max_pairs_ph.from_numpy(np.array([cap_ph], dtype=np.int32))
        self.max_contact_doublets.from_numpy(np.array([cap_doublets], dtype=np.int32))
        self.max_contact_triplets.from_numpy(np.array([cap_triplets], dtype=np.int32))
        self.n_verts.from_numpy(np.array([n_verts], dtype=np.int32))
        self.d_hat.from_numpy(np.array([d_hat], dtype=np.float64))
        self.d_hat_sq.from_numpy(np.array([d_hat_sq], dtype=np.float64))
        self.dt_sq.from_numpy(np.array([dt_sq], dtype=np.float64))
        self.kappa.from_numpy(np.array([kappa], dtype=np.float64))
        self.gass_t.from_numpy(np.array([gass_threshold], dtype=np.float64))
        self.gass_ln.from_numpy(np.array([gass_ln], dtype=np.float64))
        self.gass_a.from_numpy(np.array([gass_a], dtype=np.float64))
        self.eps_x_coeff.from_numpy(np.array([eps_x_coeff], dtype=np.float64))
        self.cd_padded.from_numpy(np.array([padded_d], dtype=np.int32))
        self.ct_padded.from_numpy(np.array([padded_t], dtype=np.int32))
        self.ccd_eta.from_numpy(np.array([ccd_eta], dtype=np.float64))

        # ---- collision pair buffers (BVH query output) ----

        self.pairs_pt = qd.ndarray(qd.i32, shape=(cap_pt, 2))
        self.pairs_ee = qd.ndarray(qd.i32, shape=(cap_ee, 2))
        self.n_pairs_pt = qd.ndarray(qd.i32, shape=())
        self.n_pairs_ee = qd.ndarray(qd.i32, shape=())

        # ---- contact doublet output (per-vertex gradient contributions) ----

        self.doublet_vert = qd.ndarray(qd.i32, shape=(cap_doublets,))
        self.doublet_grad = qd.ndarray(qd.f64, shape=(cap_doublets * 3,))
        self.n_doublets = qd.ndarray(qd.i32, shape=())

        # ---- contact triplet output (3x3 Hessian blocks, upper triangle) ----

        self.triplet_row = qd.ndarray(qd.i32, shape=(cap_triplets,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(cap_triplets,))
        self.triplet_val = qd.ndarray(qd.f64, shape=(cap_triplets * 9,))
        self.n_triplets = qd.ndarray(qd.i32, shape=())

        # ---- unique output (post sort+reduce) ----

        self.unique_doublet_vert = qd.ndarray(qd.i32, shape=(cap_doublets,))
        self.unique_doublet_grad = qd.ndarray(qd.f64, shape=(cap_doublets * 3,))
        self.n_unique_doublets = qd.ndarray(qd.i32, shape=())

        self.unique_triplet_row = qd.ndarray(qd.i32, shape=(cap_triplets,))
        self.unique_triplet_col = qd.ndarray(qd.i32, shape=(cap_triplets,))
        self.unique_triplet_val = qd.ndarray(qd.f64, shape=(cap_triplets * 9,))
        self.n_unique_triplets = qd.ndarray(qd.i32, shape=())

        # ---- sort workspace for doublet dedup (u32 keys = vertex id) ----

        sort_scratch_d = max(sort_scratch_slots(padded_d, self.sort_log256_max_n), 1)
        self.cd_sort_keys = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_keys_tmp = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_perm = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_perm_tmp = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_d,))

        # ---- sort workspace for triplet dedup (u64 keys = row<<32|col) ----

        sort_scratch_t = max(sort_scratch_slots(padded_t, self.sort_log256_max_n), 1)
        self.ct_sort_keys = qd.ndarray(qd.u64, shape=(padded_t,))
        self.ct_sort_keys_tmp = qd.ndarray(qd.u64, shape=(padded_t,))
        self.ct_sort_perm = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_sort_perm_tmp = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_t,))

        # ---- FSR workspace (segment flags + scan) ----

        from quadrants.algorithms import exclusive_scan_scratch_slots

        self.cd_seg_flags = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_seg_ids = qd.ndarray(qd.u32, shape=(padded_d,))
        scan_scratch_d = max(exclusive_scan_scratch_slots(padded_d, self.sort_log256_max_n), 1)
        self.cd_scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_d,))

        self.ct_seg_flags = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_seg_ids = qd.ndarray(qd.u32, shape=(padded_t,))
        scan_scratch_t = max(exclusive_scan_scratch_slots(padded_t, self.sort_log256_max_n), 1)
        self.ct_scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_t,))

        # ---- barrier energy ----

        self.pair_energy = qd.ndarray(qd.f64, shape=(max(total_pairs, 1),))
        self.barrier_energy = qd.ndarray(qd.f64, shape=())
        n_barrier_blocks = max((total_pairs + _CONTACT_BLOCK_DIM - 1) // _CONTACT_BLOCK_DIM, 1)
        self.barrier_partial = qd.ndarray(qd.f64, shape=(n_barrier_blocks,))
        self.n_active_pairs = qd.ndarray(qd.i32, shape=())

        # ---- CCD ----

        self.ccd_alpha = qd.ndarray(qd.f64, shape=())
        n_ccd_blocks = max((max(cap_pt, cap_ee) + _CCD_BLOCK_DIM - 1) // _CCD_BLOCK_DIM, 1)
        self.ccd_partial_pt = qd.ndarray(qd.f64, shape=(n_ccd_blocks,))
        self.ccd_partial_ee = qd.ndarray(qd.f64, shape=(n_ccd_blocks,))

        # ---- halfplane (PH) contact ----

        n_hp_buf = max(n_hp, 1)
        self.hp_position = qd.ndarray(qd.f64, shape=(n_hp_buf, 3))
        self.hp_normal = qd.ndarray(qd.f64, shape=(n_hp_buf, 3))
        self.pairs_ph = qd.ndarray(qd.i32, shape=(max(cap_ph, 1), 2))
        self.n_pairs_ph = qd.ndarray(qd.i32, shape=())

        # ---- count-only pre-sizing (cgq count_active) ----

        self.n_counted_doublets = qd.ndarray(qd.i32, shape=())
        self.n_counted_triplets = qd.ndarray(qd.i32, shape=())

        # ---- overflow flags ----

        self.overflow_flag = qd.ndarray(qd.i32, shape=())
        self.assembly_overflow_flag = qd.ndarray(qd.i32, shape=())

    def do_build(self) -> None:
        """No-op — device scalars are uploaded in ``__init__``."""

    def wire_halfplanes(self, positions: np.ndarray, normals: np.ndarray) -> None:
        """Upload the plane reference points and outward normals.

        Mirrors cgq ``ContactSystem::wire_halfplanes``. The count is fixed at
        construction because it gates a compile-time branch, so this only fills
        in the values.
        """
        P = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
        N = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(P) != self.n_halfplanes or len(N) != self.n_halfplanes:
            raise ValueError(
                f"wire_halfplanes: got {len(P)} positions and {len(N)} normals, "
                f"but the system was built for n_halfplanes={self.n_halfplanes}"
            )
        if self.n_halfplanes == 0:
            return
        norms = np.linalg.norm(N, axis=1)
        if not np.all(norms > 0.0):
            raise ValueError("wire_halfplanes: every normal must be non-zero")
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"wire_halfplanes: normals must be unit length, got norms {norms}")
        self.hp_position.from_numpy(P)
        self.hp_normal.from_numpy(N)

    def set_dt(self, dt: float) -> None:
        """Update the timestep used for the Consistent-IPC pair scale.

        Mirrors cgq ``ContactSystem::set_dt_sq``.
        """
        self.dt_sq.from_numpy(np.array([dt * dt], dtype=np.float64))

    # ------------------------------------------------------------------
    # Realloc
    # ------------------------------------------------------------------

    def _ensure_energy_capacity(self, required_pairs: int) -> None:
        """Grow per-pair barrier-energy workspace without preserving contents."""
        current = self.pair_energy.shape[0]
        capacity = max(required_pairs, current, 1)
        if capacity > current:
            self.pair_energy = qd.ndarray(qd.f64, shape=(capacity,))

        required_blocks = max(
            (capacity + _CONTACT_BLOCK_DIM - 1) // _CONTACT_BLOCK_DIM,
            1,
        )
        if required_blocks > self.barrier_partial.shape[0]:
            self.barrier_partial = qd.ndarray(qd.f64, shape=(required_blocks,))

    def realloc_pair_buffers(self, new_max_pt: int, new_max_ee: int) -> None:
        """Grow collision pair buffers after BVH query overflow.

        Mirrors cgq ``ContactSystem::realloc_pair_buffers``. Only grows, never
        shrinks. Does NOT touch assembly (doublet/triplet) buffers — use
        ``realloc_assembly_buffers`` for that.
        """
        cur_pt = self.pairs_pt.shape[0]
        cur_ee = self.pairs_ee.shape[0]
        cap_pt = max(math.ceil(new_max_pt * _GROW_FACTOR), cur_pt)
        cap_ee = max(math.ceil(new_max_ee * _GROW_FACTOR), cur_ee)

        self.pairs_pt = qd.ndarray(qd.i32, shape=(cap_pt, 2))
        self.pairs_ee = qd.ndarray(qd.i32, shape=(cap_ee, 2))
        self.max_pairs_pt.from_numpy(np.array([cap_pt], dtype=np.int32))
        self.max_pairs_ee.from_numpy(np.array([cap_ee], dtype=np.int32))

        n_ccd_blocks = max((max(cap_pt, cap_ee) + _CCD_BLOCK_DIM - 1) // _CCD_BLOCK_DIM, 1)
        self.ccd_partial_pt = qd.ndarray(qd.f64, shape=(n_ccd_blocks,))
        self.ccd_partial_ee = qd.ndarray(qd.f64, shape=(n_ccd_blocks,))

        cap_ph = self.pairs_ph.shape[0] if self.has_halfplanes else 0
        self._ensure_energy_capacity(cap_pt + cap_ee + cap_ph)
        self.overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def realloc_ph_pairs(self, new_max_ph: int) -> None:
        """Grow the halfplane pair buffer after query overflow."""
        cur_ph = self.pairs_ph.shape[0]
        cap_ph = max(math.ceil(new_max_ph * _GROW_FACTOR), cur_ph)
        self.pairs_ph = qd.ndarray(qd.i32, shape=(cap_ph, 2))
        self.max_pairs_ph.from_numpy(np.array([cap_ph], dtype=np.int32))
        self._ensure_energy_capacity(self.pairs_pt.shape[0] + self.pairs_ee.shape[0] + cap_ph)
        self.overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def realloc_assembly_buffers(self, need_doublets: int, need_triplets: int) -> None:
        """Grow doublet/triplet/sort/unique/energy buffers after count-only overflow.

        Mirrors cgq ``ContactSystem::realloc_assembly_buffers``. Sized to
        ``ceil(need * 1.2)``; only grows, never shrinks.
        """
        from quadrants.algorithms import exclusive_scan_scratch_slots

        cur_d = self.doublet_vert.shape[0]
        cur_t = self.triplet_row.shape[0]
        cap_d = max(math.ceil(need_doublets * _GROW_FACTOR), cur_d)
        cap_t = max(math.ceil(need_triplets * _GROW_FACTOR), cur_t)

        self.doublet_vert = qd.ndarray(qd.i32, shape=(cap_d,))
        self.doublet_grad = qd.ndarray(qd.f64, shape=(cap_d * 3,))
        self.triplet_row = qd.ndarray(qd.i32, shape=(cap_t,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(cap_t,))
        self.triplet_val = qd.ndarray(qd.f64, shape=(cap_t * 9,))

        self.unique_doublet_vert = qd.ndarray(qd.i32, shape=(cap_d,))
        self.unique_doublet_grad = qd.ndarray(qd.f64, shape=(cap_d * 3,))
        self.unique_triplet_row = qd.ndarray(qd.i32, shape=(cap_t,))
        self.unique_triplet_col = qd.ndarray(qd.i32, shape=(cap_t,))
        self.unique_triplet_val = qd.ndarray(qd.f64, shape=(cap_t * 9,))

        padded_d = _padded64(cap_d)
        padded_t = _padded64(cap_t)

        sort_scratch_d = max(sort_scratch_slots(padded_d, self.sort_log256_max_n), 1)
        self.cd_sort_keys = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_keys_tmp = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_perm = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_perm_tmp = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_d,))
        self.cd_seg_flags = qd.ndarray(qd.u32, shape=(padded_d,))
        self.cd_seg_ids = qd.ndarray(qd.u32, shape=(padded_d,))
        scan_scratch_d = max(exclusive_scan_scratch_slots(padded_d, self.sort_log256_max_n), 1)
        self.cd_scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_d,))

        sort_scratch_t = max(sort_scratch_slots(padded_t, self.sort_log256_max_n), 1)
        self.ct_sort_keys = qd.ndarray(qd.u64, shape=(padded_t,))
        self.ct_sort_keys_tmp = qd.ndarray(qd.u64, shape=(padded_t,))
        self.ct_sort_perm = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_sort_perm_tmp = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_t,))
        self.ct_seg_flags = qd.ndarray(qd.u32, shape=(padded_t,))
        self.ct_seg_ids = qd.ndarray(qd.u32, shape=(padded_t,))
        scan_scratch_t = max(exclusive_scan_scratch_slots(padded_t, self.sort_log256_max_n), 1)
        self.ct_scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_t,))

        # Every active pair contributes at least one doublet, but pair-buffer
        # growth may already have established a larger energy upper bound.
        self._ensure_energy_capacity(cap_d)

        self.max_contact_doublets.from_numpy(np.array([cap_d], dtype=np.int32))
        self.max_contact_triplets.from_numpy(np.array([cap_t], dtype=np.int32))
        self.cd_padded.from_numpy(np.array([padded_d], dtype=np.int32))
        self.ct_padded.from_numpy(np.array([padded_t], dtype=np.int32))

        self.assembly_overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def realloc_buffers(self, new_max_pt: int, new_max_ee: int) -> None:
        """Grow all buffers after overflow — compatibility wrapper.

        Existing harnesses (``align_utils.DCDPipelineContext``) call this.
        Production code should use ``realloc_pair_buffers`` and
        ``realloc_assembly_buffers`` separately, driven by exact demand from
        the count-only pre-pass.
        """
        self.realloc_pair_buffers(new_max_pt, new_max_ee)

        cap_pt = self.pairs_pt.shape[0]
        cap_ee = self.pairs_ee.shape[0]
        cap_ph = self.pairs_ph.shape[0]
        mesh_pairs = cap_pt + cap_ee
        need_d = max(mesh_pairs * 4 + cap_ph, 64)
        need_t = max(mesh_pairs * 10 + cap_ph, 64)
        self.realloc_assembly_buffers(need_d, need_t)

    # ------------------------------------------------------------------
    # Kernel methods
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def zero_counters(self):
        """Reset all pair/doublet/triplet/count counters to zero before query+assemble."""
        self.n_pairs_pt[()] = 0
        self.n_pairs_ee[()] = 0
        self.n_pairs_ph[()] = 0
        self.n_doublets[()] = 0
        self.n_triplets[()] = 0
        self.n_active_pairs[()] = 0
        self.n_counted_doublets[()] = 0
        self.n_counted_triplets[()] = 0
        self.overflow_flag[()] = 0
        self.assembly_overflow_flag[()] = 0

    @qd.func(requires_top_level=True)
    def reset_collision_counts(self):
        """Reset broadphase candidate counts and overflow state."""
        self.n_pairs_pt[()] = 0
        self.n_pairs_ee[()] = 0
        self.n_pairs_ph[()] = 0
        self.overflow_flag[()] = 0

    @qd.func(requires_top_level=True)
    def reset_counted_demand(self):
        """Reset count-only assembly demand and its overflow state."""
        self.n_counted_doublets[()] = 0
        self.n_counted_triplets[()] = 0
        self.assembly_overflow_flag[()] = 0

    @qd.func(requires_top_level=True)
    def reset_assembly_counts(self):
        """Reset raw and unique contact assembly counts, preserving candidates."""
        self.n_doublets[()] = 0
        self.n_triplets[()] = 0
        self.n_active_pairs[()] = 0
        self.n_unique_doublets[()] = 0
        self.n_unique_triplets[()] = 0

    # ------------------------------------------------------------------
    # CCD helpers
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def init_ccd(self):
        """Initialize CCD alpha to 1.0 before per-pair sweep."""
        self.ccd_alpha[()] = qd.f64(1.0)

    # ------------------------------------------------------------------
    # Doublet sort+unique (m7)
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def doublet_sort_seed(self):
        """Seed doublet sort: key = vertex id (u32), perm = original index."""
        for k in range(self.cd_padded[0]):
            if k < self.n_doublets[()]:
                self.cd_sort_keys[k] = qd.u32(self.doublet_vert[k])
                self.cd_sort_perm[k] = qd.u32(k)
            else:
                self.cd_sort_keys[k] = qd.u32(0xFFFFFFFF)
                self.cd_sort_perm[k] = qd.u32(k)

    @qd.func(requires_top_level=True)
    def doublet_sort_radix(self):
        """Radix sort doublet keys (u32 vertex ids)."""
        sort(
            self.cd_sort_keys,
            self.cd_sort_keys_tmp,
            self.cd_sort_perm,
            self.cd_sort_perm_tmp,
            self.cd_sort_scratch,
            self.n_doublets,
            qd.u32,
            True,
            32,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def doublet_segment_flags(self):
        """Compute segment tail flags for doublet dedup."""
        n = self.n_doublets[()]
        for k in range(self.cd_padded[0]):
            if k < n:
                if k == n - 1:
                    self.cd_seg_flags[k] = qd.u32(1)
                else:
                    if self.cd_sort_keys[k] != self.cd_sort_keys[k + 1]:
                        self.cd_seg_flags[k] = qd.u32(1)
                    else:
                        self.cd_seg_flags[k] = qd.u32(0)
            else:
                self.cd_seg_flags[k] = qd.u32(0)

    @qd.func(requires_top_level=True)
    def doublet_scan(self):
        """Exclusive scan on doublet segment flags -> segment IDs."""
        exclusive_scan_add(
            self.cd_seg_flags,
            self.cd_seg_ids,
            self.cd_scan_scratch,
            self.n_doublets[()],
            qd.u32,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def doublet_zero_unique(self):
        """Zero unique doublet grad before merge-scatter."""
        for i in range(self.max_contact_doublets[0] * 3):
            self.unique_doublet_grad[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def doublet_fsr_merge(self):
        """Fast segmented reduce: sum gradient 3-vectors by segment ID."""
        n = self.n_doublets[()]
        for k in range(n):
            src = qd.i32(self.cd_sort_perm[k])
            seg = qd.i32(self.cd_seg_ids[k])
            for c in qd.static(range(3)):
                qd.atomic_add(
                    self.unique_doublet_grad[seg * 3 + c],
                    self.doublet_grad[src * 3 + c],
                )

    @qd.func(requires_top_level=True)
    def doublet_extract_unique(self):
        """Extract unique vertex IDs from tail-flag positions."""
        n = self.n_doublets[()]
        for k in range(n):
            if self.cd_seg_flags[k] == qd.u32(1):
                seg = qd.i32(self.cd_seg_ids[k])
                self.unique_doublet_vert[seg] = qd.i32(self.cd_sort_keys[k])
                if k == n - 1:
                    self.n_unique_doublets[()] = seg + 1

    # ------------------------------------------------------------------
    # Triplet sort+unique (m7)
    # ------------------------------------------------------------------

    @qd.func(requires_top_level=True)
    def triplet_sort_seed(self):
        """Seed triplet sort: key = (row<<32|col) as u64, perm = original index."""
        for k in range(self.ct_padded[0]):
            if k < self.n_triplets[()]:
                r = qd.u64(self.triplet_row[k])
                c = qd.u64(self.triplet_col[k])
                self.ct_sort_keys[k] = (r << 32) | c
                self.ct_sort_perm[k] = qd.u32(k)
            else:
                self.ct_sort_keys[k] = qd.u64(0xFFFFFFFFFFFFFFFF)
                self.ct_sort_perm[k] = qd.u32(k)

    @qd.func(requires_top_level=True)
    def triplet_sort_radix(self):
        """Radix sort triplet keys (u64 composite row:col)."""
        sort(
            self.ct_sort_keys,
            self.ct_sort_keys_tmp,
            self.ct_sort_perm,
            self.ct_sort_perm_tmp,
            self.ct_sort_scratch,
            self.n_triplets,
            qd.u64,
            True,
            64,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def triplet_segment_flags(self):
        """Compute segment tail flags for triplet dedup."""
        n = self.n_triplets[()]
        for k in range(self.ct_padded[0]):
            if k < n:
                if k == n - 1:
                    self.ct_seg_flags[k] = qd.u32(1)
                else:
                    if self.ct_sort_keys[k] != self.ct_sort_keys[k + 1]:
                        self.ct_seg_flags[k] = qd.u32(1)
                    else:
                        self.ct_seg_flags[k] = qd.u32(0)
            else:
                self.ct_seg_flags[k] = qd.u32(0)

    @qd.func(requires_top_level=True)
    def triplet_scan(self):
        """Exclusive scan on triplet segment flags -> segment IDs."""
        exclusive_scan_add(
            self.ct_seg_flags,
            self.ct_seg_ids,
            self.ct_scan_scratch,
            self.n_triplets[()],
            qd.u32,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def triplet_zero_unique(self):
        """Zero unique triplet val before merge-scatter."""
        for i in range(self.max_contact_triplets[0] * 9):
            self.unique_triplet_val[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def triplet_fsr_merge(self):
        """Fast segmented reduce: sum 3x3 blocks by segment ID."""
        n = self.n_triplets[()]
        for k in range(n):
            src = qd.i32(self.ct_sort_perm[k])
            seg = qd.i32(self.ct_seg_ids[k])
            for c in range(9):
                qd.atomic_add(
                    self.unique_triplet_val[seg * 9 + c],
                    self.triplet_val[src * 9 + c],
                )

    @qd.func(requires_top_level=True)
    def triplet_extract_unique(self):
        """Extract unique (row, col) from tail-flag positions, record nnz."""
        n = self.n_triplets[()]
        for k in range(n):
            if self.ct_seg_flags[k] == qd.u32(1):
                seg = qd.i32(self.ct_seg_ids[k])
                key = self.ct_sort_keys[k]
                self.unique_triplet_row[seg] = qd.i32(key >> 32)
                self.unique_triplet_col[seg] = qd.i32(key & qd.u64(0xFFFFFFFF))
                if k == n - 1:
                    self.n_unique_triplets[()] = seg + 1
