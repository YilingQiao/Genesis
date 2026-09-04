"""SimEngine -- top-level simulation orchestrator with graph kernel step."""

from __future__ import annotations

import weakref

import numpy as np
from typing import TypeVar

import quadrants as qd

from genesis.engine.solvers.qipc.affine_body_preconditioner import ABDPreconditioner
from genesis.engine.solvers.qipc.affine_body_dynamics import AffineBodyDynamics
from genesis.engine.solvers.qipc.contact_function.filter_assemble_device import (
    ccd_alpha_ee_device,
    ccd_alpha_ph_device,
    ccd_alpha_pt_device,
    count_active_ee_device,
    count_active_ph_device,
    count_active_pt_device,
    filter_energy_ee_device,
    filter_energy_ph_device,
    filter_energy_pt_device,
    fused_filter_assemble_ee_device,
    fused_filter_assemble_ph_device,
    fused_filter_assemble_pt_device,
    halfplane_query_device,
    reduce_contact_energy_device,
)
from genesis.engine.solvers.qipc.contact_system import ContactSystem
from genesis.engine.solvers.qipc.global_linear_system import GlobalLinearSystem
from genesis.engine.solvers.qipc.global_surface_manager import GlobalSurfaceManager
from genesis.engine.solvers.qipc.global_vertex_manager import GlobalVertexManager
from genesis.engine.solvers.qipc.lbvh import LBVH
from genesis.engine.solvers.qipc.linear_pcg import LinearPCG
from genesis.engine.solvers.qipc.sim_config import SimConfig
from genesis.engine.solvers.qipc.sim_system import SimSystem

T = TypeVar("T", bound=SimSystem)


class _EngineState:
    """Host-only bookkeeping for one SimEngine, held outside its ``__dict__``."""

    __slots__ = ("initialized", "systems")

    def __init__(self) -> None:
        self.systems: dict[type, SimSystem] = {}
        self.initialized: bool = False


_ENGINE_STATE: dict[int, _EngineState] = {}


@qd.data_oriented
class SimEngine:
    """Top-level simulation orchestrator.  Mirrors cgq SimEngine.

    After ``init()``, call ``step()`` each frame.  The step kernel is a single
    ``@qd.kernel(graph=True)`` containing a ``qd.graph.do_while`` Newton loop
    with nested PCG and line search loops.

    The Newton pipeline follows cgq:
    positions → BVH → query → count → assemble → sort+reduce → precond → PCG →
    recover_dq → displacements → convergence → CCD → line_search → finalize.
    """

    def __init__(self) -> None:
        key = id(self)
        _ENGINE_STATE[key] = _EngineState()
        weakref.finalize(self, _ENGINE_STATE.pop, key, None)

        self.has_contact = 0

        self.ncond = qd.ndarray(qd.i32, shape=())
        self.newton_iter = qd.ndarray(qd.i32, shape=())

        self.ls_cond = qd.ndarray(qd.i32, shape=())
        self.ls_iter = qd.ndarray(qd.i32, shape=())
        self.alpha = qd.ndarray(qd.f64, shape=())
        self.energy = qd.ndarray(qd.f64, shape=())
        self.energy_trial = qd.ndarray(qd.f64, shape=())

        self.converged = qd.ndarray(qd.i32, shape=())
        self.max_disp = qd.ndarray(qd.f64, shape=())

        # Stays zero forever: the flag named by the checkpoints that never yield, since qd.checkpoint requires
        # 'yield_on' to reference an ndarray (a bare 'None' is rejected at compile time by quadrants 1.3.0).
        self.never_yield = qd.ndarray(qd.i32, shape=())

    @property
    def _state(self) -> _EngineState:
        state = _ENGINE_STATE.get(id(self))
        if state is None:
            raise RuntimeError("SimEngine has no state; __init__ was not called")
        return state

    def add_system(self, system: SimSystem) -> None:
        key = type(system)
        systems = self._state.systems
        assert key not in systems, f"Duplicate system: {key.__name__}"
        system._set_engine(self)
        systems[key] = system
        setattr(self, key.__name__, system)

    def find(self, system_type: type[T]) -> T | None:
        sys = self._state.systems.get(system_type)
        if sys is not None and sys.is_valid():
            return sys  # type: ignore
        return None

    def require(self, system_type: type[T]) -> T:
        result = self.find(system_type)
        assert result is not None, f"Required system {system_type.__name__} not found"
        return result

    def build_systems(self) -> None:
        for system in self._state.systems.values():
            system.do_build()

    def init(self) -> None:
        """Full initialization: build all systems, cache subsystem references."""
        self.build_systems()
        self._abd = self.require(AffineBodyDynamics)
        self._config = self.require(SimConfig)
        self._lsys = self.require(GlobalLinearSystem)
        self._pcg = self.require(LinearPCG)
        self._precond = self.require(ABDPreconditioner)

        vtx_mgr = self.find(GlobalVertexManager)
        if vtx_mgr is not None:
            self._vtx_mgr = vtx_mgr
        surf_mgr = self.find(GlobalSurfaceManager)
        if surf_mgr is not None:
            self._surf_mgr = surf_mgr
            if surf_mgr.n_surf_tri > 0:
                self._tri_bvh = LBVH(surf_mgr.n_surf_tri, max_queries=surf_mgr.n_surf_verts)
                self._tri_bvh.do_init()
            if surf_mgr.n_surf_edges > 0:
                self._edge_bvh = LBVH(surf_mgr.n_surf_edges)
                self._edge_bvh.do_init()

        csys = self.find(ContactSystem)
        if csys is not None:
            self._csys = csys
            self.has_contact = 1

        self._state.initialized = True
        if csys is not None:
            self._initialize_contact()

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _step_kernel(
        self,
        triplet_overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        """One contact-free simulation timestep as a CUDA graph."""
        self._abd.predict(self._config.dt)

        for _ in range(1):
            self.energy[()] = qd.f64(0.0)
        self._abd.compute_total_energy(self.energy, self._config.dt)

        for _ in range(1):
            self.ncond[()] = 1
            self.newton_iter[()] = 0

        while qd.graph.do_while(self.ncond):
            self._lsys.zero_rhs()
            self._abd.assemble_kinetic(self._lsys, self._config.dt)
            self._abd.assemble_shape(self._lsys, self._config.dt)

            with qd.checkpoint(2, yield_on=triplet_overflow):
                for _ in range(1):
                    live = self._lsys.n_elastic_rt[0]
                    self._lsys.n_live_rt[0] = live
                    self._lsys.padded_live_rt[0] = ((live + 63) // 64) * 64
                    self._lsys.srt_n[()] = live

            self._lsys.sort_seed()
            self._lsys.sort_radix()
            self._lsys.sort_segment_flags()
            self._lsys.sort_scan()
            self._lsys.sort_zero_bcoo()
            self._lsys.sort_fsr_merge()
            self._lsys.sort_extract_unique()

            self._precond.build(self._lsys)

            self._pcg.init(self._lsys, self._precond)
            for _ in range(1):
                self._pcg.pcond[()] = 1
                self._pcg.iter_buf[0] = 0
            while qd.graph.do_while(self._pcg.pcond):
                self._pcg.iteration(
                    self._lsys,
                    self._precond,
                    self._config.pcg_tol,
                    self._config.pcg_max_iter,
                )

            self._abd.recover_dq(self._lsys)
            self._abd.record_q_temp()

            for _ in range(1):
                self.max_disp[()] = qd.f64(0.0)
                self.converged[()] = 0
            self._abd.compute_max_displacement(self.max_disp, 1.0 / self._config.dt)
            for _ in range(1):
                vel_tol = self._config.vel_tol
                if self.max_disp[()] < vel_tol:
                    if self.newton_iter[()] >= self._config.newton_min:
                        self.converged[()] = 1

            for _ in range(1):
                self.alpha[()] = qd.f64(1.0)
                self.ls_cond[()] = 1
                self.ls_iter[()] = 0

            while qd.graph.do_while(self.ls_cond):
                self._abd.step_forward(self.alpha)
                for _ in range(1):
                    self.energy_trial[()] = qd.f64(0.0)
                self._abd.compute_total_energy(self.energy_trial, self._config.dt)
                for _ in range(1):
                    if self.converged[()] != 0:
                        self.ls_cond[()] = 0
                    elif self.energy_trial[()] <= self.energy[()]:
                        self.ls_cond[()] = 0
                    else:
                        self.alpha[()] = self.alpha[()] * 0.5
                        self.ls_iter[()] = self.ls_iter[()] + 1
                        if self.ls_iter[()] >= self._config.max_ls_iter:
                            self.ls_cond[()] = 0

            for _ in range(1):
                self.energy[()] = self.energy_trial[()]
                self.newton_iter[()] = self.newton_iter[()] + 1
                if self.converged[()] != 0:
                    self.ncond[()] = 0
                elif self.newton_iter[()] >= self._config.max_newton:
                    self.ncond[()] = 0
                else:
                    self.ncond[()] = 1

        self._abd.velocity_update(1.0 / self._config.dt)
        self._abd.copy_q_prev()

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _init_contact_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        """Bootstrap current-configuration candidates before the first step."""
        with qd.checkpoint(0, yield_on=self.never_yield):
            if qd.static(self.has_contact):
                self._vtx_mgr.compute_positions(self._abd)
                self._vtx_mgr.zero_displacements()
                self._tri_bvh.calc_leaf_aabb_tri(self._surf_mgr, self._vtx_mgr)
                self._tri_bvh.reduce_scene_aabb()
                self._tri_bvh.calc_morton()
                self._tri_bvh.sort_morton()
                self._tri_bvh.extract_indices()
                self._tri_bvh.copy_leaf_aabb_to_temp()
                self._tri_bvh.reorder_leaf_aabb()
                self._tri_bvh.calc_leaf_nodes()
                self._tri_bvh.calc_internal_nodes()
                self._tri_bvh.memset_flags()
                self._tri_bvh.calc_internal_aabb()
                self._edge_bvh.calc_leaf_aabb_edge(self._surf_mgr, self._vtx_mgr)
                self._edge_bvh.reduce_scene_aabb()
                self._edge_bvh.calc_morton()
                self._edge_bvh.sort_morton()
                self._edge_bvh.extract_indices()
                self._edge_bvh.copy_leaf_aabb_to_temp()
                self._edge_bvh.reorder_leaf_aabb()
                self._edge_bvh.calc_leaf_nodes()
                self._edge_bvh.calc_internal_nodes()
                self._edge_bvh.memset_flags()
                self._edge_bvh.calc_internal_aabb()

        with qd.checkpoint(1, yield_on=pair_overflow):
            if qd.static(self.has_contact):
                self._csys.reset_collision_counts()
                self._tri_bvh.query_pt(
                    self._surf_mgr,
                    self._vtx_mgr,
                    self._csys.pairs_pt,
                    self._csys.n_pairs_pt,
                    self._csys.max_pairs_pt[0],
                    self._csys.d_hat[0],
                    pair_overflow,
                )
                self._edge_bvh.query_ee(
                    self._surf_mgr,
                    self._vtx_mgr,
                    self._csys.pairs_ee,
                    self._csys.n_pairs_ee,
                    self._csys.max_pairs_ee[0],
                    self._csys.d_hat[0],
                    pair_overflow,
                )
                if qd.static(self._csys.has_halfplanes):
                    halfplane_query_device(self._csys, self._surf_mgr, self._vtx_mgr)

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _step_contact_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
        assembly_overflow: qd.types.ndarray(qd.i32, ndim=0),
        triplet_overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        """One cgq-ordered ABD + contact timestep."""
        # CP0: frame initialization, prediction and E0 at the current state.
        with qd.checkpoint(0, yield_on=self.never_yield):
            if qd.static(self.has_contact):
                self._abd.predict(self._config.dt)
                self._vtx_mgr.compute_positions(self._abd)
                for _ in range(1):
                    self.energy[()] = qd.f64(0.0)
                    self._csys.n_active_pairs[()] = 0
                    self.ncond[()] = 1
                    self.newton_iter[()] = 0
                    self.converged[()] = 0
                self._abd.compute_total_energy(self.energy, self._config.dt)
                filter_energy_pt_device(
                    self._csys,
                    self._surf_mgr,
                    self._vtx_mgr,
                    self._csys.pairs_pt,
                    self._csys.n_pairs_pt,
                )
                filter_energy_ee_device(
                    self._csys,
                    self._surf_mgr,
                    self._vtx_mgr,
                    self._csys.pairs_ee,
                    self._csys.n_pairs_ee,
                )
                if qd.static(self._csys.has_halfplanes):
                    filter_energy_ph_device(self._csys, self._surf_mgr, self._vtx_mgr)
                reduce_contact_energy_device(self._csys, self.energy)

        while qd.graph.do_while(self.ncond):
            # CP1: exact contact assembly demand; overflow resumes at CP2.
            with qd.checkpoint(1, yield_on=assembly_overflow):
                if qd.static(self.has_contact):
                    self._csys.reset_counted_demand()
                    count_active_pt_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_pt,
                        self._csys.n_pairs_pt,
                    )
                    count_active_ee_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_ee,
                        self._csys.n_pairs_ee,
                    )
                    if qd.static(self._csys.has_halfplanes):
                        count_active_ph_device(self._csys, self._surf_mgr, self._vtx_mgr)
                    for _ in range(1):
                        if self._csys.n_counted_doublets[()] > self._csys.max_contact_doublets[0]:
                            assembly_overflow[()] = 1
                        if self._csys.n_counted_triplets[()] > self._csys.max_contact_triplets[0]:
                            assembly_overflow[()] = 1

            # CP2: initialize this Newton iteration and assemble contact.
            with qd.checkpoint(2, yield_on=self.never_yield):
                if qd.static(self.has_contact):
                    for _ in range(1):
                        self.newton_iter[()] = self.newton_iter[()] + 1
                        self.ncond[()] = 1
                    self._csys.reset_assembly_counts()
                    fused_filter_assemble_pt_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_pt,
                        self._csys.n_pairs_pt,
                    )
                    fused_filter_assemble_ee_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_ee,
                        self._csys.n_pairs_ee,
                    )
                    if qd.static(self._csys.has_halfplanes):
                        fused_filter_assemble_ph_device(self._csys, self._surf_mgr, self._vtx_mgr)

            # CP3: contact sort/unique and body-triplet capacity check.
            with qd.checkpoint(3, yield_on=triplet_overflow):
                if qd.static(self.has_contact):
                    self._csys.doublet_sort_seed()
                    self._csys.doublet_sort_radix()
                    self._csys.doublet_segment_flags()
                    self._csys.doublet_scan()
                    self._csys.doublet_zero_unique()
                    self._csys.doublet_fsr_merge()
                    self._csys.doublet_extract_unique()
                    self._csys.triplet_sort_seed()
                    self._csys.triplet_sort_radix()
                    self._csys.triplet_segment_flags()
                    self._csys.triplet_scan()
                    self._csys.triplet_zero_unique()
                    self._csys.triplet_fsr_merge()
                    self._csys.triplet_extract_unique()
                    self._lsys.set_live_from_contact(self._csys)

            # CP4: solve, submit dx, snapshot E0 state, build swept BVHs.
            # Contains the nested PCG graph loop.
            with qd.checkpoint(4, yield_on=self.never_yield):
                if qd.static(self.has_contact):
                    self._lsys.zero_rhs()
                    self._abd.assemble_kinetic(self._lsys, self._config.dt)
                    self._abd.assemble_shape(self._lsys, self._config.dt)
                    self._abd.distribute_contact_gradient(self._csys, self._lsys)
                    self._abd.distribute_contact_hessian(self._csys, self._lsys)
                    self._lsys.sort_seed()
                    self._lsys.sort_radix()
                    self._lsys.sort_segment_flags()
                    self._lsys.sort_scan()
                    self._lsys.sort_zero_bcoo()
                    self._lsys.sort_fsr_merge()
                    self._lsys.sort_extract_unique()
                    self._precond.build(self._lsys)
                    self._pcg.init(self._lsys, self._precond)
                    for _ in range(1):
                        self._pcg.pcond[()] = 1
                        self._pcg.iter_buf[0] = 0
                    while qd.graph.do_while(self._pcg.pcond):
                        self._pcg.iteration(
                            self._lsys,
                            self._precond,
                            self._config.pcg_tol,
                            self._config.pcg_max_iter,
                        )
                    self._abd.recover_dq(self._lsys)
                    self._abd.record_q_temp()
                    self._vtx_mgr.compute_displacements(self._abd)
                    self._vtx_mgr.record_safe_positions()
                    for _ in range(1):
                        self.max_disp[()] = qd.f64(0.0)
                        self.converged[()] = 0
                    self._abd.compute_max_displacement(self.max_disp, 1.0 / self._config.dt)
                    for _ in range(1):
                        if self.max_disp[()] < self._config.vel_tol:
                            if self.newton_iter[()] >= self._config.newton_min:
                                self.converged[()] = 1
                    self._tri_bvh.calc_leaf_aabb_tri(self._surf_mgr, self._vtx_mgr)
                    self._tri_bvh.reduce_scene_aabb()
                    self._tri_bvh.calc_morton()
                    self._tri_bvh.sort_morton()
                    self._tri_bvh.extract_indices()
                    self._tri_bvh.copy_leaf_aabb_to_temp()
                    self._tri_bvh.reorder_leaf_aabb()
                    self._tri_bvh.calc_leaf_nodes()
                    self._tri_bvh.calc_internal_nodes()
                    self._tri_bvh.memset_flags()
                    self._tri_bvh.calc_internal_aabb()
                    self._edge_bvh.calc_leaf_aabb_edge(self._surf_mgr, self._vtx_mgr)
                    self._edge_bvh.reduce_scene_aabb()
                    self._edge_bvh.calc_morton()
                    self._edge_bvh.sort_morton()
                    self._edge_bvh.extract_indices()
                    self._edge_bvh.copy_leaf_aabb_to_temp()
                    self._edge_bvh.reorder_leaf_aabb()
                    self._edge_bvh.calc_leaf_nodes()
                    self._edge_bvh.calc_internal_nodes()
                    self._edge_bvh.memset_flags()
                    self._edge_bvh.calc_internal_aabb()

            # CP5: query the swept BVHs. Pair overflow resumes this checkpoint.
            with qd.checkpoint(5, yield_on=pair_overflow):
                if qd.static(self.has_contact):
                    self._csys.reset_collision_counts()
                    self._tri_bvh.query_pt(
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_pt,
                        self._csys.n_pairs_pt,
                        self._csys.max_pairs_pt[0],
                        self._csys.d_hat[0],
                        pair_overflow,
                    )
                    self._edge_bvh.query_ee(
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_ee,
                        self._csys.n_pairs_ee,
                        self._csys.max_pairs_ee[0],
                        self._csys.d_hat[0],
                        pair_overflow,
                    )
                    if qd.static(self._csys.has_halfplanes):
                        halfplane_query_device(self._csys, self._surf_mgr, self._vtx_mgr)

            # CP6: directional CCD on the swept candidate set.
            with qd.checkpoint(6, yield_on=self.never_yield):
                if qd.static(self.has_contact):
                    self._csys.init_ccd()
                    ccd_alpha_pt_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_pt,
                        self._csys.n_pairs_pt,
                    )
                    ccd_alpha_ee_device(
                        self._csys,
                        self._surf_mgr,
                        self._vtx_mgr,
                        self._csys.pairs_ee,
                        self._csys.n_pairs_ee,
                    )
                    if qd.static(self._csys.has_halfplanes):
                        ccd_alpha_ph_device(self._csys, self._surf_mgr, self._vtx_mgr)

            # CP7: line search and Newton finalization. Contains nested LS loop.
            with qd.checkpoint(7, yield_on=self.never_yield):
                if qd.static(self.has_contact):
                    for _ in range(1):
                        self.alpha[()] = qd.f64(1.0)
                        ca = self._csys.ccd_alpha[()]
                        if ca > 0.0 and ca < 1.0:
                            self.alpha[()] = ca
                        self.ls_cond[()] = 1
                        self.ls_iter[()] = 0
                    while qd.graph.do_while(self.ls_cond):
                        self._abd.step_forward(self.alpha)
                        self._vtx_mgr.step_forward(self.alpha)
                        for _ in range(1):
                            self.energy_trial[()] = qd.f64(0.0)
                            self._csys.n_active_pairs[()] = 0
                        self._abd.compute_total_energy(self.energy_trial, self._config.dt)
                        filter_energy_pt_device(
                            self._csys,
                            self._surf_mgr,
                            self._vtx_mgr,
                            self._csys.pairs_pt,
                            self._csys.n_pairs_pt,
                        )
                        filter_energy_ee_device(
                            self._csys,
                            self._surf_mgr,
                            self._vtx_mgr,
                            self._csys.pairs_ee,
                            self._csys.n_pairs_ee,
                        )
                        if qd.static(self._csys.has_halfplanes):
                            filter_energy_ph_device(self._csys, self._surf_mgr, self._vtx_mgr)
                        reduce_contact_energy_device(self._csys, self.energy_trial)
                        for _ in range(1):
                            if self.converged[()] != 0:
                                self.ls_cond[()] = 0
                            elif self.energy_trial[()] <= self.energy[()]:
                                self.ls_cond[()] = 0
                            else:
                                self.alpha[()] = self.alpha[()] * 0.5
                                self.ls_iter[()] = self.ls_iter[()] + 1
                                if self.ls_iter[()] >= self._config.max_ls_iter:
                                    self.ls_cond[()] = 0
                    for _ in range(1):
                        self.energy[()] = self.energy_trial[()]
                        if self.converged[()] != 0:
                            self.ncond[()] = 0
                        elif self.newton_iter[()] >= self._config.max_newton:
                            self.ncond[()] = 0
                        else:
                            self.ncond[()] = 1

        # CP8: skipped on every yielded launch, run once after Newton exits.
        with qd.checkpoint(8, yield_on=self.never_yield):
            if qd.static(self.has_contact):
                self._abd.velocity_update(1.0 / self._config.dt)
                self._abd.copy_q_prev()

    def _handle_pair_overflow(self) -> None:
        n_pt = int(self._csys.n_pairs_pt.to_numpy())
        n_ee = int(self._csys.n_pairs_ee.to_numpy())
        n_ph = int(self._csys.n_pairs_ph.to_numpy())
        self._csys.realloc_pair_buffers(n_pt, n_ee)
        if n_ph > self._csys.pairs_ph.shape[0]:
            self._csys.realloc_ph_pairs(n_ph)
        self._csys.overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def _initialize_contact(self) -> None:
        pair_overflow = self._csys.overflow_flag
        status = self._init_contact_kernel(pair_overflow)
        while status.yielded:
            self._handle_pair_overflow()
            status = self._init_contact_kernel.resume(
                pair_overflow,
                from_checkpoint=1,
            )

    def step(self):
        """Run one timestep with overflow handling (yield/resume)."""
        csys = getattr(self, "_csys", None)
        triplet_overflow = self._lsys.triplet_overflow

        if csys is None:
            status = self._step_kernel(triplet_overflow)
            while status.yielded:
                need = int(self._lsys.n_live_rt.to_numpy()[0])
                new_cap = max(int(np.ceil(need * 1.2)), self._lsys.capacity + 1)
                self._lsys.realloc_triplet_buffers(new_cap)
                triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
                status = self._step_kernel.resume(
                    triplet_overflow,
                    from_checkpoint=status.checkpoint,
                )
            return

        pair_overflow = csys.overflow_flag
        assembly_overflow = csys.assembly_overflow_flag
        status = self._step_contact_kernel(
            pair_overflow,
            assembly_overflow,
            triplet_overflow,
        )
        while status.yielded:
            if status.checkpoint == 1:
                need_d = int(csys.n_counted_doublets.to_numpy())
                need_t = int(csys.n_counted_triplets.to_numpy())
                csys.realloc_assembly_buffers(need_d, need_t)
                assembly_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = 2
            elif status.checkpoint == 3:
                need = int(self._lsys.n_live_rt.to_numpy()[0])
                new_cap = max(int(np.ceil(need * 1.2)), self._lsys.capacity + 1)
                self._lsys.realloc_triplet_buffers(new_cap)
                triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = 4
            elif status.checkpoint == 5:
                self._handle_pair_overflow()
                resume_from = 5
            else:
                raise RuntimeError(f"unexpected contact checkpoint yield: {status.checkpoint}")

            status = self._step_contact_kernel.resume(
                pair_overflow,
                assembly_overflow,
                triplet_overflow,
                from_checkpoint=resume_from,
            )
