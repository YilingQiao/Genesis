from pathlib import Path

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu

from .base_entity import Entity


class QIPCEntity(Entity):
    """
    A near-rigid affine body simulated by the QIPC solver.

    The entity is tetrahedralized from its morph and simulated as one 12-DOF affine body with penetration-free
    Incremental Potential Contact (IPC) against every other QIPC entity. A `gs.morphs.Plane` morph becomes an analytic
    half-plane: a static ground with no degrees of freedom.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : QIPCSolver
        The solver simulating this entity.
    material : gs.materials.QIPC.Base
        The material defining density and rigidity stiffness.
    morph : Morph
        The morph specification that defines the entity's shape and initial pose.
    surface : Surface
        The surface associated with the entity.
    idx : int
        Unique identifier of the entity within the scene.
    body_idx : int
        Index of this entity's affine body in the solver's body arrays. Planes carry no body and receive -1.
    v_start : int
        Starting index of this entity's vertices in the solver's global vertex arrays.
    name : str | None, optional
        Entity name, auto-generated from the morph when not given.
    """

    def __init__(self, scene, solver, material, morph, surface, idx, body_idx, v_start, name=None):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._body_idx = body_idx
        self._v_start = v_start

        if isinstance(morph, gs.morphs.Plane):
            # The QIPC contact system takes the plane analytically: one reference point and one outward unit normal,
            # both in the world frame.
            normal = gu.transform_by_quat(np.array(morph.normal, dtype=np.float64), np.array(morph.quat))
            self._plane_normal = normal / np.linalg.norm(normal)
            self._plane_pos = np.array(morph.pos, dtype=np.float64)
            self._init_verts = np.zeros((0, 3), dtype=np.float64)
            self._tets = np.zeros((0, 4), dtype=np.int32)
        else:
            self._plane_normal = None
            self._plane_pos = None
            meshes = gs.Mesh.from_morph_surface(morph, surface)
            surface_verts, surface_faces, _ = mu.merge_submeshes(
                [mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes]
            )
            # Tetrahedralized untranslated: the affine state carries the morph pose exactly, so the rest shape stays
            # entity-local (a well-conditioned affine parameterization) and the on-disk tet cache is shared across
            # placements of the same shape.
            surface_trimesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
            verts, tets = eu.mesh_to_elements(surface_trimesh, tet_cfg=mu.generate_tetgen_config_from_morph(morph))
            if not len(verts) > 0:
                gs.raise_exception("Entity has zero vertices.")
            self._init_verts = np.ascontiguousarray(verts, dtype=np.float64)
            self._tets = np.ascontiguousarray(tets, dtype=np.int32)

        # The morph pose, offsets composed in the body frame as documented on `Morph.offset_pos`.
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(self._morph.offset_pos, dtype=np.float64),
            np.array(self._morph.offset_quat, dtype=np.float64),
            np.array(self._morph.pos, dtype=np.float64),
            np.array(self._morph.quat, dtype=np.float64),
        )
        self._init_transform = gu.trans_quat_to_T(pos, quat)

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "qipc_box"
        if isinstance(morph, gs.morphs.Sphere):
            return "qipc_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "qipc_cylinder"
        if isinstance(morph, gs.morphs.Plane):
            return "qipc_plane"
        if isinstance(morph, gs.morphs.Mesh):
            return f"qipc_{Path(morph.file).stem}"
        return "qipc_entity"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- runtime access ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_verts(self) -> torch.Tensor:
        """
        Get the world-frame positions of this entity's simulated vertices.

        Returns
        -------
        verts : torch.Tensor, shape (n_vertices, 3)
            The vertex positions, in the solver's native float64 precision.
        """
        if self.is_plane:
            gs.raise_exception("A plane is analytic and carries no vertices.")
        return self._solver.get_entity_verts(self)

    @gs.assert_built
    def get_transform(self) -> torch.Tensor:
        """
        Get the affine placement of this entity's body.

        Returns
        -------
        transform : torch.Tensor, shape (4, 4)
            The homogeneous transform mapping the entity's rest shape to the world frame. Its linear part is only
            near-orthogonal, since rigidity is enforced by a stiffness penalty.
        """
        if self.is_plane:
            gs.raise_exception("A plane is analytic and carries no body.")
        return self._solver.get_entity_transform(self)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_plane(self):
        """Whether this entity is an analytic half-plane rather than an affine body."""
        return self._plane_normal is not None

    @property
    def is_fixed(self):
        """Whether this entity is fixed in the world."""
        return self._morph.fixed

    @property
    def body_idx(self):
        """Index of this entity's affine body in the solver's body arrays, -1 for a plane."""
        return self._body_idx

    @property
    def v_start(self):
        """Starting index of this entity's vertices in the solver's global vertex arrays."""
        return self._v_start

    @property
    def n_vertices(self):
        """Number of simulated vertices of this entity."""
        return len(self._init_verts)

    @property
    def init_verts(self):
        """Rest-shape vertex positions in the entity frame, shape (n_vertices, 3)."""
        return self._init_verts

    @property
    def tets(self):
        """Tetrahedra indexing into `init_verts`, shape (n_tets, 4)."""
        return self._tets

    @property
    def init_transform(self):
        """The initial world placement of the entity's rest shape, shape (4, 4)."""
        return self._init_transform

    @property
    def plane_pos(self):
        """World-frame reference point of the half-plane, None for an affine body."""
        return self._plane_pos

    @property
    def plane_normal(self):
        """World-frame outward unit normal of the half-plane, None for an affine body."""
        return self._plane_normal
