"""
ImGui overlay plugin for joint control and simulation controls.

Requires: pip install imgui-bundle
"""

import os
import time
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
from genesis.vis.viewer_plugins import ViewerPlugin, EVENT_HANDLED, EVENT_HANDLE_STATE

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.viewer import Viewer

FREE_JOINT_POS_LIMIT = 10.0
QUATERNION_COMPONENT_LIMIT = 1.0

_FPS_HISTORY_SIZE = 30


class ImGuiOverlayPlugin(ViewerPlugin):
    """
    ViewerPlugin that adds an ImGui control panel for simulation and joint control.

    Features:
    - Simulation controls: play/pause, step, reset
    - Joint sliders for each entity (editable only when paused)
    - FPS display with rolling average
    - Multi-step support
    - Custom panel registration API

    Limitations:
    - Only controls environment 0 in batched simulations

    Usage:
        scene.build()
        plugin = ImGuiOverlayPlugin()
        scene.viewer._pyrender_viewer.register_plugin(plugin)

        while scene.viewer.is_alive():
            if plugin.should_step():
                scene.step()
    """

    def __init__(
        self,
        show_sim_controls=True,
        show_entity_browser=True,
        show_visualization=True,
        show_camera_controls=True,
    ):
        super().__init__()
        self._imgui = None
        self._impl = None
        self._io = None
        self._available = False
        self._init_attempted = False
        self._last_time = None
        self.paused = False
        self._step_requested = False
        self._steps_remaining = 0
        self._step_count = 1
        self._entity_cache = {}
        self._user_panels = []
        self._fps_history = []

        # Section visibility flags
        self.show_sim_controls = show_sim_controls
        self.show_entity_browser = show_entity_browser
        self.show_visualization = show_visualization
        self.show_camera_controls = show_camera_controls

    def register_panel(self, callback, section="side"):
        """Register custom UI panel. callback(imgui) called each frame.

        Thread-safe: uses copy-on-write list.

        Args:
            callback: Function taking imgui module as argument, called each frame.
            section: "side" adds to main panel, "overlay" creates floating window.
        """
        new_list = list(self._user_panels) + [(callback, section)]
        self._user_panels = new_list  # Atomic reference swap

    def build(self, viewer: "Viewer", camera, scene: "Scene"):
        """Store references; ImGui initialization is deferred to on_draw (viewer thread)."""
        super().build(viewer, camera, scene)
        # Cache entity data now (doesn't require OpenGL)
        self._cache_entity_data()

    def _init_imgui(self):
        """Initialize ImGui. Must be called from the viewer thread (e.g., in on_draw)."""
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend

            self._imgui = imgui
            imgui.create_context()
            # Load default font at larger size before renderer builds the atlas
            io = imgui.get_io()
            io.fonts.clear()
            font_cfg = imgui.ImFontConfig()
            font_cfg.size_pixels = 18.0
            io.fonts.add_font_default(font_cfg)
            self._impl = pyglet_backend.create_renderer(self.viewer, attach_callbacks=False)
            # Fix: Set window reference for cursor handling (not set when attach_callbacks=False)
            self._impl._window = self.viewer
            self._io = imgui.get_io()
            self._io.set_ini_filename("")  # Don't persist window positions
            self._setup_style()
            self._available = True
        except ImportError:
            print("ImGuiOverlayPlugin: imgui-bundle not found. Install with: pip install imgui-bundle")
        except Exception as e:
            print(f"ImGuiOverlayPlugin: Failed to initialize ImGui: {e}")

    def _setup_style(self):
        """Apply dark theme styling (IsaacGym-inspired)."""
        imgui = self._imgui
        # Start from ImGui's built-in dark theme
        imgui.style_colors_dark()
        style = imgui.get_style()
        Col_ = imgui.Col_
        sc = style.set_color_

        # Geometry - clean, slightly rounded
        style.window_rounding = 4.0
        style.frame_rounding = 2.0
        style.child_rounding = 2.0
        style.popup_rounding = 2.0
        style.scrollbar_rounding = 2.0
        style.grab_rounding = 2.0
        style.tab_rounding = 2.0

        # Spacing
        style.window_padding = (8.0, 8.0)
        style.frame_padding = (6.0, 4.0)
        style.item_spacing = (8.0, 4.0)
        style.item_inner_spacing = (4.0, 4.0)
        style.scrollbar_size = 14.0
        style.grab_min_size = 12.0
        style.window_border_size = 1.0
        style.frame_border_size = 0.0

        # Solid dark backgrounds
        sc(Col_.window_bg, (0.10, 0.10, 0.10, 0.95))
        sc(Col_.child_bg, (0.12, 0.12, 0.12, 1.0))
        sc(Col_.popup_bg, (0.10, 0.10, 0.10, 0.98))

        # Text
        sc(Col_.text, (1.0, 1.0, 1.0, 1.0))
        sc(Col_.text_disabled, (0.50, 0.50, 0.50, 1.0))

        # Borders
        sc(Col_.border, (0.28, 0.28, 0.28, 1.0))

        # Frames (sliders, input fields) - dark gray
        sc(Col_.frame_bg, (0.20, 0.20, 0.20, 1.0))
        sc(Col_.frame_bg_hovered, (0.28, 0.28, 0.28, 1.0))
        sc(Col_.frame_bg_active, (0.32, 0.32, 0.32, 1.0))

        # Title bar
        sc(Col_.title_bg, (0.08, 0.08, 0.08, 1.0))
        sc(Col_.title_bg_active, (0.14, 0.14, 0.14, 1.0))
        sc(Col_.title_bg_collapsed, (0.08, 0.08, 0.08, 0.75))

        # Buttons - blue accent
        sc(Col_.button, (0.24, 0.40, 0.60, 1.0))
        sc(Col_.button_hovered, (0.30, 0.50, 0.72, 1.0))
        sc(Col_.button_active, (0.20, 0.36, 0.55, 1.0))

        # Headers (collapsing headers)
        sc(Col_.header, (0.22, 0.22, 0.22, 1.0))
        sc(Col_.header_hovered, (0.30, 0.50, 0.72, 1.0))
        sc(Col_.header_active, (0.26, 0.44, 0.65, 1.0))

        # Interactive accents - blue
        sc(Col_.check_mark, (0.40, 0.65, 0.90, 1.0))
        sc(Col_.slider_grab, (0.35, 0.55, 0.80, 1.0))
        sc(Col_.slider_grab_active, (0.40, 0.65, 0.90, 1.0))

        # Scrollbar
        sc(Col_.scrollbar_bg, (0.08, 0.08, 0.08, 1.0))
        sc(Col_.scrollbar_grab, (0.30, 0.30, 0.30, 1.0))
        sc(Col_.scrollbar_grab_hovered, (0.40, 0.40, 0.40, 1.0))
        sc(Col_.scrollbar_grab_active, (0.50, 0.50, 0.50, 1.0))

        # Tabs
        sc(Col_.tab, (0.14, 0.14, 0.14, 1.0))
        sc(Col_.tab_hovered, (0.30, 0.50, 0.72, 1.0))
        sc(Col_.tab_selected, (0.24, 0.40, 0.60, 1.0))

        # Separators
        sc(Col_.separator, (0.28, 0.28, 0.28, 1.0))
        sc(Col_.separator_hovered, (0.35, 0.55, 0.80, 1.0))
        sc(Col_.separator_active, (0.40, 0.65, 0.90, 1.0))

        # Resize grip
        sc(Col_.resize_grip, (0.28, 0.28, 0.28, 0.25))
        sc(Col_.resize_grip_hovered, (0.35, 0.55, 0.80, 0.65))
        sc(Col_.resize_grip_active, (0.40, 0.65, 0.90, 0.90))

    def _get_entity_name(self, entity, idx: int) -> str:
        """Extract a human-readable name for an entity, with index for disambiguation."""
        morph_file = getattr(getattr(entity, "morph", None), "file", None)
        if morph_file:
            base_name = os.path.splitext(os.path.basename(morph_file))[0]
            return f"{base_name} [{idx}]"
        return f"Entity_{entity.idx}"

    def _cache_entity_data(self):
        """Cache static joint metadata from all rigid entities."""
        self._entity_cache.clear()

        if not hasattr(self.scene, "rigid_solver") or self.scene.rigid_solver is None:
            return

        for entity in self.scene.rigid_solver.entities:
            q_names, q_limits_lower, q_limits_upper, q_is_quaternion = [], [], [], []
            quat_groups = []  # list of (start_idx, end_idx) for each quaternion group

            if entity.n_dofs == 0:
                # Still include for vis_mode toggle, but no joint data
                self._entity_cache[entity.idx] = {
                    "entity": entity,
                    "name": self._get_entity_name(entity, entity.idx),
                    "q_names": [],
                    "q_limits": ([], []),
                    "q_is_quaternion": [],
                    "quat_groups": [],
                    "n_qs": 0,
                    "n_dofs": 0,
                }
                continue

            for joint in entity.joints:
                if joint.n_qs == 0 or joint.type == gs.JOINT_TYPE.FIXED:
                    continue

                if joint.type == gs.JOINT_TYPE.FREE:
                    # Free joint: 3 position DOFs + 4 quaternion components
                    q_names.extend(
                        [
                            f"{joint.name}_x",
                            f"{joint.name}_y",
                            f"{joint.name}_z",
                            f"{joint.name}_qw",
                            f"{joint.name}_qx",
                            f"{joint.name}_qy",
                            f"{joint.name}_qz",
                        ]
                    )
                    q_limits_lower.extend([-FREE_JOINT_POS_LIMIT] * 3 + [-QUATERNION_COMPONENT_LIMIT] * 4)
                    q_limits_upper.extend([FREE_JOINT_POS_LIMIT] * 3 + [QUATERNION_COMPONENT_LIMIT] * 4)
                    q_is_quaternion.extend([False, False, False, True, True, True, True])
                    quat_groups.append((len(q_names) - 4, len(q_names)))
                elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                    # Spherical joint: n_qs=4 (quaternion), n_dofs=3
                    quat_start = len(q_names)
                    q_names.extend(
                        [
                            f"{joint.name}_qw",
                            f"{joint.name}_qx",
                            f"{joint.name}_qy",
                            f"{joint.name}_qz",
                        ]
                    )
                    q_limits_lower.extend([-QUATERNION_COMPONENT_LIMIT] * 4)
                    q_limits_upper.extend([QUATERNION_COMPONENT_LIMIT] * 4)
                    q_is_quaternion.extend([True, True, True, True])
                    quat_groups.append((quat_start, quat_start + 4))
                else:
                    # Revolute, prismatic, or other joints: n_qs == n_dofs
                    for i in range(joint.n_qs):
                        name = joint.name if joint.n_qs == 1 else f"{joint.name}[{i}]"
                        q_names.append(name)
                        q_limits_lower.append(float(joint.dofs_limit[i, 0]))
                        q_limits_upper.append(float(joint.dofs_limit[i, 1]))
                        q_is_quaternion.append(False)

            if q_names:
                self._entity_cache[entity.idx] = {
                    "entity": entity,
                    "name": self._get_entity_name(entity, entity.idx),
                    "q_names": q_names,
                    "q_limits": (q_limits_lower, q_limits_upper),
                    "q_is_quaternion": q_is_quaternion,
                    "quat_groups": quat_groups,
                    "n_qs": len(q_names),
                    "n_dofs": entity.n_dofs,
                }

    def _apply_qpos_update(self, entity, new_qpos, is_multi_env: bool) -> None:
        """Apply qpos update to entity, handling single-env vs multi-env correctly.

        Args:
            entity: The RigidEntity to update.
            new_qpos: Array-like of new joint positions.
            is_multi_env: If True, pass envs_idx=0 to set_qpos. If False, omit envs_idx.
        """
        qpos_array = np.asarray(new_qpos)
        # Single-env scenes don't accept envs_idx parameter
        if is_multi_env:
            entity.set_qpos(qpos_array, envs_idx=0)
        else:
            entity.set_qpos(qpos_array)

        # Update visual transforms after qpos change
        rigid_solver = self.scene.rigid_solver
        if rigid_solver.is_active:
            rigid_solver.update_geoms_render_T()
            rigid_solver.update_vgeoms()
            rigid_solver.update_vgeoms_render_T()

            # Force context to update render buffer with new transforms
            gs_context = self.viewer.gs_context
            gs_context.update_link_frame(gs_context.buffer)
            gs_context.update_rigid(gs_context.buffer)

    def _switch_entity_vis_mode(self, entity, new_mode):
        """Switch entity visualization between 'visual' and 'collision' at runtime."""
        from genesis.ext import pyrender

        old_mode = entity.surface.vis_mode
        if old_mode == new_mode:
            return

        gs_context = self.viewer.gs_context
        rigid_solver = self.scene.rigid_solver

        # Remove old geom nodes
        old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
        for geom in old_geoms:
            if geom.uid in gs_context.rigid_nodes:
                gs_context.remove_node(gs_context.rigid_nodes[geom.uid])
                del gs_context.rigid_nodes[geom.uid]

        # Set new mode
        entity.surface.vis_mode = new_mode

        # Update transforms so they're fresh
        rigid_solver.update_geoms_render_T()
        rigid_solver.update_vgeoms()
        rigid_solver.update_vgeoms_render_T()

        # Add new geom nodes
        if new_mode == "visual":
            geoms = entity.vgeoms
            geoms_T = rigid_solver._vgeoms_render_T
        else:
            geoms = entity.geoms
            geoms_T = rigid_solver._geoms_render_T

        for geom in geoms:
            geom_envs_idx = gs_context._get_geom_active_envs_idx(geom, gs_context.rendered_envs_idx)
            if len(geom_envs_idx) == 0:
                continue

            mesh = geom.get_trimesh()
            geom_T = geoms_T[geom.idx][geom_envs_idx]
            is_collision = "collision" in new_mode
            gs_context.add_rigid_node(
                geom,
                pyrender.Mesh.from_trimesh(
                    mesh=mesh,
                    poses=geom_T,
                    smooth=geom.surface.smooth if not is_collision else False,
                    double_sided=geom.surface.double_sided if not is_collision else False,
                    is_floor=isinstance(entity._morph, gs.morphs.Plane),
                    env_shared=not gs_context.env_separate_rigid,
                ),
            )

    def _is_capturing(self) -> bool:
        """Check if ImGui wants mouse/keyboard input."""
        if not self._available:
            return False
        return self._io.want_capture_mouse or self._io.want_capture_keyboard

    # Event handlers - forward input to ImGui and block when capturing
    def on_mouse_press(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_press(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_release(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_release(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            # imgui backend expects: on_mouse_scroll(x, y, mods, scroll)
            self._impl.on_mouse_scroll(x, y, 0, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_motion(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_motion(x, y, dx, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_press(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_press(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_release(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_release(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_text(self, text) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_text(text)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_resize(self, width, height) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_resize(width, height)
        return None

    def on_draw(self) -> None:
        """Render ImGui overlay after scene is drawn."""
        # Lazy initialization: must happen in viewer thread (which owns OpenGL context)
        if not self._init_attempted:
            self._init_imgui()

        if not self._available:
            return

        # Update delta time manually (avoid calling pyglet.clock.tick() which conflicts with viewer loop)
        current_time = time.perf_counter()
        if self._last_time is not None:
            self._io.delta_time = current_time - self._last_time
        else:
            self._io.delta_time = 1.0 / 60.0
        if self._io.delta_time <= 0.0:
            self._io.delta_time = 1.0 / 1000.0
        self._last_time = current_time

        # Track FPS history
        if self._io.delta_time > 0:
            self._fps_history.append(1.0 / self._io.delta_time)
            if len(self._fps_history) > _FPS_HISTORY_SIZE:
                self._fps_history = self._fps_history[-_FPS_HISTORY_SIZE:]

        self._imgui.new_frame()

        self._render_control_panel()

        self._imgui.render()
        self._impl.render(self._imgui.get_draw_data())

    def _render_control_panel(self):
        """Render unified control panel with all sections."""
        imgui = self._imgui
        imgui.begin("Genesis Control Panel", flags=imgui.WindowFlags_.always_auto_resize)

        if self.show_sim_controls:
            self._render_sim_controls()

        if self.show_visualization:
            if imgui.collapsing_header("Visualization"):
                imgui.indent()
                self._render_visualization()
                imgui.unindent()

        if self.show_entity_browser:
            if imgui.collapsing_header("Entities"):
                imgui.indent()
                self._render_entity_browser()
                imgui.unindent()

        if self.show_camera_controls:
            if imgui.collapsing_header("Camera"):
                imgui.indent()
                self._render_camera_controls()
                imgui.unindent()

        # Render user callback panels (side panels)
        for callback, section in self._user_panels:
            if section == "side":
                callback(imgui)

        imgui.end()

        # Render overlay panels as separate windows
        for callback, section in self._user_panels:
            if section == "overlay":
                callback(imgui)

    def _render_sim_controls(self):
        """Render simulation control buttons, time display, and FPS."""
        imgui = self._imgui

        # Play/Pause, Step, Reset buttons
        if imgui.button("Pause" if not self.paused else "Play", size=(60, 0)):
            self.paused = not self.paused
        imgui.same_line()
        if imgui.button("Step", size=(50, 0)):
            self._steps_remaining = self._step_count
        imgui.same_line()
        if imgui.button("Reset", size=(50, 0)):
            with self.viewer.render_lock:
                self.scene.reset()

        # Time display (frame count * dt = simulation time)
        if hasattr(self.scene, "t"):
            sim_time = self.scene.t * self.scene.sim.dt
            imgui.text(f"Time: {sim_time:.3f}s  Step: {self.scene.t}")

        # FPS display
        if self._fps_history:
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            imgui.same_line()
            imgui.text(f"  FPS: {avg_fps:.0f}")

        if hasattr(self.scene, "n_envs") and self.scene.n_envs > 1:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), f"Note: Controlling env 0 of {self.scene.n_envs}")

        imgui.separator()

    def _render_visualization(self):
        """Render visualization toggle controls."""
        imgui = self._imgui
        render_flags = self.viewer.render_flags
        gs_context = self.viewer.gs_context

        # Shadows
        changed, new_val = imgui.checkbox("Shadows", render_flags["shadows"])
        if changed:
            render_flags["shadows"] = new_val

        # Wireframe combo (4-state cycle matching DefaultControlsPlugin._toggle_wireframe)
        wireframe_options = ["Default", "Flip Wireframe", "All Wireframe", "All Solid"]
        if render_flags["all_solid"]:
            current_wireframe = 3
        elif render_flags["all_wireframe"]:
            current_wireframe = 2
        elif render_flags["flip_wireframe"]:
            current_wireframe = 1
        else:
            current_wireframe = 0
        changed, new_idx = imgui.combo("Wireframe", current_wireframe, wireframe_options)
        if changed:
            render_flags["flip_wireframe"] = new_idx == 1
            render_flags["all_wireframe"] = new_idx == 2
            render_flags["all_solid"] = new_idx == 3

        # World Frame
        changed, new_val = imgui.checkbox("World Frame", gs_context.world_frame_shown)
        if changed:
            if new_val:
                gs_context.on_world_frame()
            else:
                gs_context.off_world_frame()

        # Link Frame
        changed, new_val = imgui.checkbox("Link Frame", gs_context.link_frame_shown)
        if changed:
            if new_val:
                gs_context.on_link_frame()
            else:
                gs_context.off_link_frame()

        # Link Frame Size slider
        link_size = gs_context.link_frame_size
        changed_size, new_size = imgui.slider_float("Frame Size##link_frame_size", link_size, 0.02, 0.5, "%.2f")
        if changed_size and gs_context.link_frame_size > 0:
            scale = new_size / gs_context.link_frame_size
            gs_context.link_frame_mesh.vertices *= scale
            gs_context.link_frame_size = new_size
            if gs_context.link_frame_shown:
                gs_context.off_link_frame()
                gs_context.on_link_frame()

        # Camera Frustum
        changed, new_val = imgui.checkbox("Camera Frustum", gs_context.camera_frustum_shown)
        if changed:
            if new_val:
                gs_context.on_camera_frustum()
            else:
                gs_context.off_camera_frustum()

        # Face Normals
        changed, new_val = imgui.checkbox("Face Normals", render_flags["face_normals"])
        if changed:
            render_flags["face_normals"] = new_val

        # Vertex Normals
        changed, new_val = imgui.checkbox("Vertex Normals", render_flags["vertex_normals"])
        if changed:
            render_flags["vertex_normals"] = new_val

    def _render_camera_controls(self):
        """Render camera position, lookat, FOV controls."""
        imgui = self._imgui
        trackball = self.viewer._trackball

        # Read current camera state from trackball
        pose = trackball._n_pose
        pos = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
        # Use trackball's actual orbit center as lookat (not derived from z-axis)
        target = trackball._n_target
        lookat = [float(target[0]), float(target[1]), float(target[2])]

        # Position drag
        changed_pos, new_pos = imgui.drag_float3("Position##cam_pos", pos, 0.05, -100.0, 100.0, "%.2f")

        # Lookat drag
        changed_lookat, new_lookat = imgui.drag_float3("Lookat##cam_lookat", lookat, 0.05, -100.0, 100.0, "%.2f")

        if changed_pos or changed_lookat:
            cam_pos = np.array(list(new_pos)) if changed_pos else np.array(pos)
            cam_lookat = np.array(list(new_lookat)) if changed_lookat else np.array(lookat)
            # Build pose with fixed world-up to prevent unintuitive roll
            from genesis.utils import geom as gu

            world_up = np.array([0.0, 0.0, 1.0])
            cam_pose = gu.pos_lookat_up_to_T(cam_pos, cam_lookat, world_up)
            self.scene.viewer._camera_up = cam_pose[:3, 1].copy()
            trackball.set_camera_pose(cam_pose)
            # Sync trackball orbit center so mouse orbiting works correctly after
            trackball._n_target = cam_lookat.copy()
            trackball._target = cam_lookat.copy()

        # FOV slider
        fov_deg = float(self.camera.camera.yfov * 180.0 / np.pi)
        changed_fov, new_fov = imgui.slider_float("FOV##cam_fov", fov_deg, 15.0, 120.0, "%.1f")
        if changed_fov:
            self.camera.camera.yfov = new_fov * np.pi / 180.0

        # Reset Camera button
        if imgui.button("Reset Camera", size=(120, 0)):
            self.viewer._reset_view()

    def _render_entity_browser(self):
        """Render entity list with joint sliders."""
        imgui = self._imgui

        if not self._entity_cache:
            imgui.text("No controllable entities")
            return

        for entity_idx, data in self._entity_cache.items():
            entity = data["entity"]
            expanded = imgui.collapsing_header(f"{data['name']}##entity_{entity_idx}")
            if not expanded:
                continue

            imgui.indent()

            # DOF count display
            imgui.text(f"DOFs: {data['n_dofs']}")

            # Vis mode combo
            vis_modes = ["visual", "collision"]
            current_mode = entity.surface.vis_mode
            current_mode_idx = vis_modes.index(current_mode) if current_mode in vis_modes else 0
            changed_mode, new_mode_idx = imgui.combo(f"Vis Mode##vis_{entity_idx}", current_mode_idx, vis_modes)
            if changed_mode:
                self._switch_entity_vis_mode(entity, vis_modes[new_mode_idx])

            # Visualize contact toggle
            show_contact = entity.visualize_contact
            changed_contact, new_contact = imgui.checkbox(f"Show Contacts##contact_{entity_idx}", show_contact)
            if changed_contact:
                entity._visualize_contact = new_contact
                for link in entity.links:
                    link._visualize_contact = new_contact

            # Joint sections only for entities with DOFs
            if data["n_dofs"] > 0:
                # Get qpos - handle multi-env case by using only env 0
                qpos_tensor = entity.get_qpos()
                qpos_np = qpos_tensor.cpu().numpy()

                # If multi-env (2D tensor with shape [n_envs, n_qs]), use only env 0
                is_multi_env = qpos_np.ndim == 2
                if is_multi_env:
                    qpos = qpos_np[0]
                else:
                    qpos = qpos_np.flatten()

                changed_any = False
                new_qpos = list(qpos)

                # Joint control section
                if imgui.collapsing_header(f"Joint Control##joints_{entity_idx}"):
                    imgui.indent()
                    lower, upper = data["q_limits"]
                    for i, (name, val, lo, hi, is_quat) in enumerate(
                        zip(data["q_names"], qpos, lower, upper, data["q_is_quaternion"])
                    ):
                        if not self.paused:
                            imgui.text(f"{name}: {val:.4f}")
                        elif is_quat:
                            changed, new_val = imgui.drag_float(
                                f"{name}##{entity_idx}_{i}", float(val), 0.01, float(lo), float(hi), "%.4f"
                            )
                            if changed:
                                new_qpos[i] = new_val
                                changed_any = True
                        else:
                            changed, new_val = imgui.slider_float(
                                f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                            )
                            if changed:
                                new_qpos[i] = new_val
                                changed_any = True
                    imgui.unindent()

                if changed_any:
                    # Normalize any edited quaternion groups
                    for qstart, qend in data["quat_groups"]:
                        q = np.array(new_qpos[qstart:qend])
                        norm = np.linalg.norm(q)
                        if norm > 1e-8:
                            q /= norm
                            new_qpos[qstart:qend] = q.tolist()
                    with self.viewer.render_lock:
                        self._apply_qpos_update(entity, new_qpos, is_multi_env)

            imgui.unindent()

    def should_step(self) -> bool:
        """Check if simulation should advance this frame."""
        if self._steps_remaining > 0:
            self._steps_remaining -= 1
            return True
        # Legacy single-step support
        if self._step_requested:
            self._step_requested = False
            return True
        return not self.paused

    def on_close(self) -> None:
        """Clean up ImGui resources."""
        if self._available and self._impl:
            self._impl.shutdown()
        if self._imgui:
            self._imgui.destroy_context()
