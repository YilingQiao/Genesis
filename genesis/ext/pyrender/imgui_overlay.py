"""
ImGui overlay plugin for joint control and simulation controls.

Requires: pip install imgui-bundle
"""

import os
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
from genesis.vis.viewer_plugins import ViewerPlugin, EVENT_HANDLED, EVENT_HANDLE_STATE

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.viewer import Viewer

FREE_JOINT_POS_LIMIT = 10.0
QUATERNION_COMPONENT_LIMIT = 1.0


class ImGuiOverlayPlugin(ViewerPlugin):
    """
    ViewerPlugin that adds ImGui panels for joint control and simulation.

    Features:
    - Joint Control panel: sliders for each joint (editable only when paused)
    - Simulation panel: play/pause/step/reset/speed controls

    Limitations:
    - Only controls environment 0 in batched simulations
    - Free joint quaternions are read-only (position editable)
    - Spherical joint quaternions are read-only

    Usage:
        scene.build()
        plugin = ImGuiOverlayPlugin()
        scene.viewer._pyrender_viewer.register_plugin(plugin)

        while scene.viewer.is_alive():
            if plugin.should_step():
                scene.step()
    """

    def __init__(self):
        super().__init__()
        self._imgui = None
        self._impl = None
        self._io = None
        self._available = False
        self.paused = False
        self.speed = 1.0
        self._step_requested = False
        self._entity_cache = {}

    def build(self, viewer: "Viewer", camera, scene: "Scene"):
        """Initialize ImGui and cache entity joint data."""
        super().build(viewer, camera, scene)

        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend

            self._imgui = imgui
            imgui.create_context()
            self._impl = pyglet_backend.create_renderer(viewer)
            self._io = imgui.get_io()
            self._io.ini_filename = None  # Don't persist window positions
            self._setup_style()
            self._available = True
        except ImportError:
            print("ImGuiOverlayPlugin: imgui-bundle not found. Install with: pip install imgui-bundle")
            return

        self._cache_entity_data()

    def _setup_style(self):
        """Apply dark theme styling."""
        style = self._imgui.get_style()
        style.window_rounding = 4.0
        style.frame_rounding = 2.0
        style.colors[self._imgui.Col_.window_bg.value] = (0.1, 0.1, 0.1, 0.9)

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
            if entity.n_dofs == 0:
                continue

            q_names, q_limits_lower, q_limits_upper, q_is_quaternion = [], [], [], []

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
                    # Position components are editable, quaternion components are read-only
                    q_is_quaternion.extend([False, False, False, True, True, True, True])
                elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                    # Spherical joint: n_qs=4 (quaternion), n_dofs=3
                    # All 4 quaternion components are read-only
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
                    "n_qs": len(q_names),
                }

    def _is_capturing(self) -> bool:
        """Check if ImGui wants mouse/keyboard input."""
        if not self._available:
            return False
        return self._io.want_capture_mouse or self._io.want_capture_keyboard

    # Event handlers - block input when ImGui is capturing
    def on_mouse_press(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_release(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_motion(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_press(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_release(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        return EVENT_HANDLED if self._is_capturing() else None

    def on_resize(self, width, height) -> EVENT_HANDLE_STATE:
        if not self._available:
            return None
        fb_width, fb_height = self.viewer.get_framebuffer_size()
        if width > 0 and height > 0:
            self._io.display_framebuffer_scale = (fb_width / width, fb_height / height)
        self._io.display_size = (fb_width, fb_height)
        return None

    def on_draw(self) -> None:
        """Render ImGui overlay after scene is drawn."""
        if not self._available:
            return

        # process_inputs may fail on some backends during window transitions
        try:
            self._impl.process_inputs()
        except AttributeError:
            pass
        self._imgui.new_frame()

        self._render_joint_panel()
        self._render_sim_controls()

        self._imgui.render()
        self._impl.render(self._imgui.get_draw_data())

    def _render_joint_panel(self):
        """Render joint control sliders."""
        imgui = self._imgui
        imgui.begin("Joint Control", flags=imgui.WindowFlags_.always_auto_resize)

        if not self._entity_cache:
            imgui.text("No controllable entities")
            imgui.end()
            return

        if not self.paused:
            imgui.text_colored((0.7, 0.7, 0.7, 1.0), "Pause simulation to edit joints")

        for entity_idx, data in self._entity_cache.items():
            entity = data["entity"]
            expanded, _ = imgui.collapsing_header(data["name"])
            if not expanded:
                continue

            # Get qpos - handle multi-env case by using only env 0
            qpos_tensor = entity.get_qpos()
            qpos_np = qpos_tensor.cpu().numpy()

            # If multi-env (2D tensor with shape [n_envs, n_qs]), use only env 0
            if qpos_np.ndim == 2:
                qpos = qpos_np[0]
            else:
                qpos = qpos_np.flatten()

            lower, upper = data["q_limits"]
            changed_any = False
            new_qpos = list(qpos)

            for i, (name, val, lo, hi, is_quat) in enumerate(
                zip(data["q_names"], qpos, lower, upper, data["q_is_quaternion"])
            ):
                if is_quat:
                    imgui.text(f"{name}: {val:.4f}")
                elif not self.paused:
                    imgui.text(f"{name}: {val:.3f}")
                else:
                    changed, new_val = imgui.slider_float(
                        f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                    )
                    if changed:
                        new_qpos[i] = new_val
                        changed_any = True

            if changed_any:
                with self.viewer.render_lock:
                    # Enforce env 0 only for multi-env scenes
                    entity.set_qpos(np.array(new_qpos), envs_idx=0)

        imgui.end()

    def _render_sim_controls(self):
        """Render simulation control panel."""
        imgui = self._imgui
        imgui.begin("Simulation", flags=imgui.WindowFlags_.always_auto_resize)

        if imgui.button("Pause" if not self.paused else "Play", size=(60, 0)):
            self.paused = not self.paused
        imgui.same_line()
        if imgui.button("Step", size=(50, 0)):
            self._step_requested = True
        imgui.same_line()
        if imgui.button("Reset", size=(50, 0)):
            with self.viewer.render_lock:
                self.scene.reset()

        _, self.speed = imgui.slider_float("Speed", self.speed, 0.1, 5.0, "%.1fx")

        if hasattr(self.scene, "t"):
            imgui.text(f"Time: {self.scene.t:.3f}s")

        if hasattr(self.scene, "n_envs") and self.scene.n_envs > 1:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), f"Note: Controlling env 0 of {self.scene.n_envs}")

        imgui.end()

    def should_step(self) -> bool:
        """Check if simulation should advance this frame."""
        if self._step_requested:
            self._step_requested = False
            return True
        return not self.paused

    def get_speed(self) -> float:
        """Get current simulation speed multiplier."""
        return self.speed

    def on_close(self) -> None:
        """Clean up ImGui resources."""
        if self._available and self._impl:
            self._impl.shutdown()
        if self._imgui:
            self._imgui.destroy_context()
