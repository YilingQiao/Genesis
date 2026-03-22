"""Scene-level control surface for visualization toggles and entity rendering.

Provides a unified API that all frontends (pyglet viewer, ImGui overlay, web server)
program against instead of reaching through scene.visualizer._rasterizer._context.
"""

import numpy as np

import genesis as gs


class SceneController:
    """Wraps the RasterizerContext behind a clean public API.

    Created by Visualizer.build() and exposed as scene.controller.
    All methods are no-ops if ctx is None (headless scene).
    """

    def __init__(self, scene, ctx=None):
        self._scene = scene
        self._ctx = ctx
        self._wireframe = False
        self._face_normals = False
        self._vertex_normals = False
        self._entity_wireframe = {}  # entity_idx -> bool

    # -- Visualization toggles -------------------------------------------------

    def set_shadows(self, enable: bool) -> None:
        if self._ctx is None:
            return
        self._ctx.shadow = enable

    def get_shadows(self) -> bool:
        if self._ctx is None:
            return False
        return bool(self._ctx.shadow)

    def set_world_frame(self, enable: bool) -> None:
        if self._ctx is None:
            return
        if enable:
            self._ctx.on_world_frame()
        else:
            self._ctx.off_world_frame()

    def get_world_frame(self) -> bool:
        if self._ctx is None:
            return False
        return bool(self._ctx.world_frame_shown)

    def set_link_frame(self, enable: bool) -> None:
        if self._ctx is None:
            return
        if enable:
            self._ctx.on_link_frame()
            self._ctx.update_link_frame(self._ctx.buffer)
        else:
            self._ctx.off_link_frame()

    def get_link_frame(self) -> bool:
        if self._ctx is None:
            return False
        return bool(self._ctx.link_frame_shown)

    def set_link_frame_size(self, size: float) -> None:
        if self._ctx is None:
            return
        current = self._ctx.link_frame_size
        if current <= 0:
            return
        scale = size / current
        self._ctx.link_frame_mesh.vertices *= scale
        self._ctx.link_frame_size = size
        if self._ctx.link_frame_shown:
            self._ctx.off_link_frame()
            self._ctx.on_link_frame()

    def get_link_frame_size(self) -> float:
        if self._ctx is None:
            return 0.0
        return float(self._ctx.link_frame_size)

    def set_camera_frustum(self, enable: bool) -> None:
        if self._ctx is None:
            return
        if enable:
            self._ctx.on_camera_frustum()
        else:
            self._ctx.off_camera_frustum()

    def get_camera_frustum(self) -> bool:
        if self._ctx is None:
            return False
        return bool(self._ctx.camera_frustum_shown)

    def set_wireframe(self, enable: bool) -> None:
        """Toggle global material-level wireframe on all mesh primitives.

        Clears per-entity wireframe state (global overrides per-entity).
        """
        if self._ctx is None:
            return
        self._wireframe = enable
        self._entity_wireframe.clear()
        for node in self._ctx._scene.mesh_nodes:
            for primitive in node.mesh.primitives:
                if primitive.material is not None:
                    primitive.material.wireframe = enable
        self._ctx._scene._meshes_updated = True

    def get_wireframe(self) -> bool:
        return self._wireframe

    def set_face_normals(self, enable: bool) -> None:
        if self._ctx is None:
            return
        from genesis.ext.pyrender.constants import RenderFlags

        self._face_normals = enable
        current = getattr(self._ctx, "_extra_render_flags", RenderFlags.NONE)
        if enable:
            self._ctx._extra_render_flags = current | RenderFlags.FACE_NORMALS
        else:
            self._ctx._extra_render_flags = current & ~RenderFlags.FACE_NORMALS

    def get_face_normals(self) -> bool:
        return self._face_normals

    def set_vertex_normals(self, enable: bool) -> None:
        if self._ctx is None:
            return
        from genesis.ext.pyrender.constants import RenderFlags

        self._vertex_normals = enable
        current = getattr(self._ctx, "_extra_render_flags", RenderFlags.NONE)
        if enable:
            self._ctx._extra_render_flags = current | RenderFlags.VERTEX_NORMALS
        else:
            self._ctx._extra_render_flags = current & ~RenderFlags.VERTEX_NORMALS

    def get_vertex_normals(self) -> bool:
        return self._vertex_normals

    # -- Entity rendering ------------------------------------------------------

    def switch_entity_vis_mode(self, entity, mode: str) -> None:
        """Switch entity between 'visual' and 'collision' rendering.

        Reapplies per-entity wireframe after the switch.
        """
        if self._ctx is None:
            return
        from genesis.ext import pyrender

        if not hasattr(entity, "surface"):
            return
        old_mode = entity.surface.vis_mode
        if old_mode == mode:
            return

        rigid_solver = self._scene.rigid_solver

        # Remove old geom nodes
        old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
        for geom in old_geoms:
            if geom.uid in self._ctx.rigid_nodes:
                self._ctx.remove_node(self._ctx.rigid_nodes[geom.uid])
                del self._ctx.rigid_nodes[geom.uid]

        entity.surface.vis_mode = mode

        rigid_solver.update_geoms_render_T()
        rigid_solver.update_vgeoms()
        rigid_solver.update_vgeoms_render_T()

        if mode == "visual":
            geoms = entity.vgeoms
            geoms_T = rigid_solver._vgeoms_render_T
        else:
            geoms = entity.geoms
            geoms_T = rigid_solver._geoms_render_T

        for geom in geoms:
            geom_envs_idx = self._ctx._get_geom_active_envs_idx(geom, self._ctx.rendered_envs_idx)
            if len(geom_envs_idx) == 0:
                continue
            mesh = geom.get_trimesh()
            geom_T = geoms_T[geom.idx][geom_envs_idx]
            is_collision = mode == "collision"
            self._ctx.add_rigid_node(
                geom,
                pyrender.Mesh.from_trimesh(
                    mesh=mesh,
                    poses=geom_T,
                    smooth=geom.surface.smooth if not is_collision else False,
                    double_sided=geom.surface.double_sided if not is_collision else False,
                    is_floor=isinstance(entity._morph, gs.morphs.Plane),
                    env_shared=not self._ctx.env_separate_rigid,
                ),
            )

        # Reapply per-entity wireframe if it was set
        if self._entity_wireframe.get(entity.idx, False):
            self._apply_entity_wireframe(entity, True)

    def set_entity_wireframe(self, entity, enable: bool) -> None:
        """Toggle wireframe rendering for a specific entity's mesh primitives."""
        if self._ctx is None:
            return
        self._entity_wireframe[entity.idx] = enable
        self._apply_entity_wireframe(entity, enable)

    def _apply_entity_wireframe(self, entity, enable: bool) -> None:
        """Internal: apply wireframe state to an entity's geom nodes."""
        geoms = (
            entity.vgeoms
            if hasattr(entity, "surface") and entity.surface.vis_mode == "visual"
            else entity.geoms
            if hasattr(entity, "geoms")
            else []
        )
        for geom in geoms:
            if geom.uid in self._ctx.rigid_nodes:
                node = self._ctx.rigid_nodes[geom.uid]
                for primitive in node.mesh.primitives:
                    if primitive.material is not None:
                        primitive.material.wireframe = enable
        self._ctx._scene._meshes_updated = True

    def set_entity_contact_viz(self, entity, enable: bool) -> None:
        """Toggle contact-force arrow rendering for an entity and its links."""
        entity._visualize_contact = enable
        if hasattr(entity, "links"):
            for link in entity.links:
                link._visualize_contact = enable

    def refresh_visual_transforms(self) -> None:
        """Refresh render transforms so visuals reflect the latest qpos.

        Call after entity.set_qpos() or entity.set_dofs_position().
        """
        if self._ctx is None:
            return
        rigid_solver = self._scene.rigid_solver
        if not rigid_solver.is_active:
            return
        rigid_solver.update_geoms_render_T()
        rigid_solver.update_vgeoms()
        rigid_solver.update_vgeoms_render_T()
        self._ctx.update_link_frame(self._ctx.buffer)
        self._ctx.update_rigid(self._ctx.buffer)

    # -- State snapshots -------------------------------------------------------

    def get_vis_state(self) -> dict:
        """Return current visualization state as a plain dict."""
        if self._ctx is None:
            return {}
        return {
            "shadows": bool(self._ctx.shadow),
            "world_frame": bool(self._ctx.world_frame_shown),
            "link_frame": bool(self._ctx.link_frame_shown),
            "link_frame_size": float(getattr(self._ctx, "link_frame_size", 0.1)),
            "camera_frustum": bool(self._ctx.camera_frustum_shown),
            "face_normals": self._face_normals,
            "vertex_normals": self._vertex_normals,
            "wireframe": self._wireframe,
            "orthographic": False,
        }

    def get_scene_camera_state(self) -> dict:
        """Return scene camera (cameras[0]) state as a plain dict.

        This is the scene's primary render camera, NOT the pyglet viewer's
        interactive camera managed by Trackball.
        """
        try:
            camera = self._scene.visualizer.cameras[0]
            pos = camera.pos
            lookat = camera.lookat
            fov = camera.fov if hasattr(camera, "fov") else 30.0
            state = {
                "pos": pos.tolist(),
                "lookat": lookat.tolist(),
                "fov": float(fov),
            }
            # Include view/projection matrices for gizmo accuracy
            try:
                view_mat = np.linalg.inv(camera.transform)
                state["view_matrix"] = view_mat.T.flatten().tolist()
            except Exception:
                pass
            try:
                rasterizer = self._scene.visualizer._rasterizer
                cam_node = rasterizer._camera_nodes[camera.uid]
                proj = cam_node.camera.get_projection_matrix(width=camera.res[0], height=camera.res[1])
                state["proj_matrix"] = proj.T.flatten().tolist()
            except Exception:
                pass
            return state
        except Exception:
            return {}

    # -- Scene camera FOV ------------------------------------------------------

    def set_scene_camera_fov(self, fov: float) -> None:
        """Set FOV on the scene's primary render camera (cameras[0])."""
        try:
            camera = self._scene.visualizer.cameras[0]
            camera._fov = float(fov)
        except Exception:
            pass

    def get_scene_camera_fov(self) -> float:
        """Get FOV from the scene's primary render camera (cameras[0])."""
        try:
            camera = self._scene.visualizer.cameras[0]
            return float(camera.fov)
        except Exception:
            return 30.0
