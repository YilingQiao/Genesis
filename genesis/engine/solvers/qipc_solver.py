import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.qipc_entity import QIPCEntity
from genesis.engine.materials import QIPC
from genesis.engine.states.solvers import QIPCSolverState
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from .base_solver import GravityMixin, Solver, TimeBasedMixin
from .qipc import (
    ABDPreconditioner,
    AffineBodyDynamics,
    ContactSystem,
    GlobalLinearSystem,
    GlobalSurfaceManager,
    GlobalVertexManager,
    LinearPCG,
    SimConfig,
    SimEngine,
)
from .qipc.affine_body_math import (
    compute_abd_gravity,
    compute_dyadic_mass,
    compute_mesh_volume,
    dyadic_mass_to_mat12,
    extract_tet_surface,
    invert_mass_12x12,
    transform_to_q,
)


class QIPCSolver(GravityMixin, TimeBasedMixin, Solver):
    """
    An affine-body Incremental Potential Contact (IPC) solver.

    Each entity is one 12-DOF affine body, near-rigid through a stiffness penalty, with penetration-free frictionless
    contact against every other QIPC entity (projected-Newton on the incremental potential, with a barrier, continuous
    collision detection and a line search). The solver steps by itself inside the shared substep loop and exchanges no
    state with any other solver.

    The step kernel is compiled at build time with `dt`, gravity and every solver option baked in, in float64
    regardless of the Genesis precision. It requires the CUDA backend, a single environment, and no differentiability.
    """

    material_cls = QIPC.Base

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._engine = None
        self._abd = None
        self._vtx_mgr = None
        self._n_bodies = 0
        self._n_verts = 0

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None):
        if visualize_contact:
            gs.raise_exception("'visualize_contact' is not supported for QIPC entities.")
        is_plane = isinstance(morph, gs.morphs.Plane)
        entity = QIPCEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            body_idx=-1 if is_plane else self._n_bodies,
            v_start=self._n_verts,
            name=name,
        )
        if not is_plane:
            self._n_bodies += 1
            self._n_verts += entity.n_vertices
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        if not self.is_active:
            return

        if self._sim.n_envs > 0:
            gs.raise_exception("QIPCSolver does not support parallel environments (n_envs > 0) for now.")
        if gs.backend != gs.cuda:
            gs.raise_exception("QIPCSolver requires the CUDA backend.")
        if self._sim.requires_grad:
            gs.raise_exception("QIPCSolver does not support differentiable mode.")

        bodies = [entity for entity in self._entities if not entity.is_plane]
        planes = [entity for entity in self._entities if entity.is_plane]
        if not bodies:
            gs.raise_exception("The QIPC scene holds only planes. Add at least one affine body.")

        # Host-side assembly: per-body mass and rest data, then one flat global vertex space.
        gravity = np.array(self._options.gravity, dtype=np.float64)
        bodies_q = np.empty((self._n_bodies, 12), dtype=np.float64)
        bodies_mass_m = np.empty((self._n_bodies,), dtype=np.float64)
        bodies_mass_m_xbar = np.empty((self._n_bodies, 3), dtype=np.float64)
        bodies_mass_m_xx = np.empty((self._n_bodies, 3, 3), dtype=np.float64)
        bodies_mass_inv = np.empty((self._n_bodies, 12, 12), dtype=np.float64)
        bodies_gravity_acc = np.empty((self._n_bodies, 12), dtype=np.float64)
        bodies_kappa_vol = np.empty((self._n_bodies,), dtype=np.float64)
        bodies_is_fixed = np.empty((self._n_bodies,), dtype=np.int32)
        verts_x_bar = np.concatenate([entity.init_verts for entity in bodies])
        verts_body_idx = np.concatenate(
            [np.full((entity.n_vertices,), entity.body_idx, dtype=np.int32) for entity in bodies]
        )
        surf_tris = []
        surf_edges = []
        surf_verts = []
        for entity in bodies:
            i_b = entity.body_idx
            m, m_xbar, m_xx = compute_dyadic_mass(entity.init_verts, entity.tets, entity.material.rho)
            mass_12 = dyadic_mass_to_mat12(m, m_xbar, m_xx)
            mass_inv = invert_mass_12x12(mass_12)
            bodies_q[i_b] = transform_to_q(entity.init_transform)
            bodies_mass_m[i_b] = m
            bodies_mass_m_xbar[i_b] = m_xbar
            bodies_mass_m_xx[i_b] = m_xx
            bodies_mass_inv[i_b] = mass_inv
            bodies_gravity_acc[i_b] = mass_inv @ compute_abd_gravity(mass_12, gravity)
            bodies_kappa_vol[i_b] = entity.material.kappa * compute_mesh_volume(entity.init_verts, entity.tets)
            bodies_is_fixed[i_b] = entity.is_fixed
            entity_tris, entity_edges, entity_verts = extract_tet_surface(entity.tets)
            surf_tris.append(entity_tris + entity.v_start)
            surf_edges.append(entity_edges + entity.v_start)
            surf_verts.append(entity_verts + entity.v_start)
        surf_tris = np.concatenate(surf_tris)
        surf_edges = np.concatenate(surf_edges)
        surf_verts = np.concatenate(surf_verts)

        abd = AffineBodyDynamics(self._n_bodies, self._n_verts)
        abd.n_bodies_rt.from_numpy(np.array([self._n_bodies], dtype=np.int32))
        abd.q.from_numpy(bodies_q)
        abd.q_prev.from_numpy(bodies_q)
        abd.q_v.from_numpy(np.zeros((self._n_bodies, 12), dtype=np.float64))
        abd.mass_m.from_numpy(bodies_mass_m)
        abd.mass_m_xbar.from_numpy(bodies_mass_m_xbar)
        abd.mass_m_xx.from_numpy(bodies_mass_m_xx)
        abd.mass_inv.from_numpy(bodies_mass_inv)
        abd.gravity_acc.from_numpy(bodies_gravity_acc)
        abd.kappa_vol.from_numpy(bodies_kappa_vol)
        abd.is_fixed.from_numpy(bodies_is_fixed)
        abd.x_bar.from_numpy(verts_x_bar)
        abd.body_id.from_numpy(verts_body_idx)

        vtx_mgr = GlobalVertexManager(self._n_verts)
        vtx_mgr.n_verts_rt.from_numpy(np.array([self._n_verts], dtype=np.int32))
        vtx_mgr.x_bar.from_numpy(verts_x_bar)
        vtx_mgr.body_id.from_numpy(verts_body_idx)

        surf_mgr = GlobalSurfaceManager(len(surf_tris), len(surf_edges), len(surf_verts))
        surf_mgr.surf_triangles.from_numpy(surf_tris)
        surf_mgr.surf_edges.from_numpy(surf_edges)
        surf_mgr.surf_verts.from_numpy(surf_verts)
        surf_mgr.wire_area_weights(verts_x_bar)

        # Half-planes take pseudo-vertex ids on the tail of the vertex index space (see ContactSystem.hp_off), so the
        # contact system is sized for one extra slot per plane.
        csys = ContactSystem(
            max_pairs_pt=self._options.n_contact_pairs_init,
            max_pairs_ee=self._options.n_contact_pairs_init,
            d_hat=self._options.d_hat,
            kappa=self._options.contact_kappa,
            n_verts=self._n_verts + len(planes),
            dt=self.substep_dt,
            n_halfplanes=len(planes),
            max_pairs_ph=self._options.n_contact_pairs_init,
        )
        if planes:
            csys.wire_halfplanes(
                np.stack([entity.plane_pos for entity in planes]),
                np.stack([entity.plane_normal for entity in planes]),
            )

        config = SimConfig(
            dt=self.substep_dt,
            gravity=tuple(self._options.gravity),
            max_newton=self._options.n_newton_iterations,
            vel_tol=self._options.newton_dv_threshold,
            max_ls_iter=self._options.n_linesearch_iterations,
            pcg_tol=self._options.pcg_threshold,
            pcg_max_iter=self._options.n_pcg_iterations,
        )
        lsys = GlobalLinearSystem(4 * self._n_bodies, 20 * self._n_bodies, csys.triplet_row.shape[0] * 16)
        pcg = LinearPCG(12 * self._n_bodies)
        precond = ABDPreconditioner(self._n_bodies)

        engine = SimEngine()
        engine.add_system(config)
        engine.add_system(abd)
        engine.add_system(lsys)
        engine.add_system(pcg)
        engine.add_system(precond)
        engine.add_system(vtx_mgr)
        engine.add_system(surf_mgr)
        engine.add_system(csys)
        engine.init()

        self._engine = engine
        self._abd = abd
        self._vtx_mgr = vtx_mgr

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        return None

    def substep_pre_coupling(self, f):
        self._engine.step()
        # Halt on divergence rather than integrating garbage: a non-finite potential poisons every later state. The
        # readback is cheap because the step's checkpoint protocol already synchronizes with the host.
        energy = float(qd_to_numpy(self._engine.energy))
        if not np.isfinite(energy):
            gs.raise_exception("QIPC Newton solve diverged (non-finite incremental potential).")

    def substep_post_coupling(self, f):
        return

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def reset_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        if not self.is_active:
            return None
        state = QIPCSolverState(self._scene)
        state.q = qd_to_torch(self._abd.q, copy=True)
        state.q_v = qd_to_torch(self._abd.q_v, copy=True)
        return state

    def set_state(self, f, state, envs_idx=None):
        q = np.ascontiguousarray(tensor_to_array(state.q), dtype=np.float64)
        q_v = np.ascontiguousarray(tensor_to_array(state.q_v), dtype=np.float64)
        self._abd.q.from_numpy(q)
        self._abd.q_prev.from_numpy(q)
        self._abd.q_v.from_numpy(q_v)
        # The step pipeline reuses the previous step's collision candidates, so an arbitrary state jump must rebuild
        # them from the restored positions, exactly as the engine bootstraps them at init.
        self._engine._initialize_contact()

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # ---------------------------------- entity access -----------------------------------
    # ------------------------------------------------------------------------------------

    def get_entity_verts(self, entity):
        """World-frame positions of one entity's vertices, as a float64 torch tensor of shape (n_vertices, 3)."""
        return qd_to_torch(self._vtx_mgr.positions, slice(entity.v_start, entity.v_start + entity.n_vertices))

    def get_entity_transform(self, entity):
        """Affine placement of one entity's body, as a float64 torch tensor of shape (4, 4)."""
        q = qd_to_torch(self._abd.q, slice(entity.body_idx, entity.body_idx + 1))[0]
        transform = torch.eye(4, dtype=q.dtype, device=q.device)
        transform[:3, 3] = q[:3]
        transform[:3, :3] = q[3:].reshape(3, 3)
        return transform

    # ------------------------------------------------------------------------------------
    # ------------------------------------- gravity --------------------------------------
    # ------------------------------------------------------------------------------------

    def set_gravity(self, gravity, envs_idx=None):
        """Gravity is baked into the compiled step kernel at build time, so runtime changes are rejected."""
        gs.raise_exception("QIPCSolver bakes gravity into its step kernel at build time. Set it through QIPCOptions.")

    def get_gravity(self, envs_idx=None):
        """The gravity this solver was built with, in m/s^2, as a tensor of shape (3,)."""
        return torch.tensor(self._options.gravity, dtype=gs.tc_float, device=gs.device)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return bool(self._entities)

    @property
    def n_bodies(self):
        """Number of affine bodies in the solver (planes are analytic and carry none)."""
        return self._n_bodies

    @property
    def n_vertices(self):
        """Total number of simulated vertices across all affine bodies."""
        return self._n_verts

    @property
    def engine(self):
        """The vendored QIPC engine, exposing solve diagnostics (newton_iter, converged, energy, max_disp)."""
        return self._engine
