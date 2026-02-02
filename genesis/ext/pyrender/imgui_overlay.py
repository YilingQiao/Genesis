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


class ImGuiOverlayPlugin(ViewerPlugin):
    """
    ViewerPlugin that adds an ImGui control panel for simulation and joint control.

    Features:
    - Simulation controls: play/pause, step, reset
    - Joint sliders for each entity (editable only when paused)

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
        self._init_attempted = False
        self._last_time = None
        self.paused = False
        self._step_requested = False
        self._entity_cache = {}

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
        """Apply dark theme styling."""
        style = self._imgui.get_style()
        style.window_rounding = 4.0
        style.frame_rounding = 2.0
        style.set_color_(self._imgui.Col_.window_bg, (0.1, 0.1, 0.1, 0.9))

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
            gs_context.update_rigid(gs_context.buffer)

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

        self._imgui.new_frame()

        self._render_control_panel()

        self._imgui.render()
        self._impl.render(self._imgui.get_draw_data())

    def _render_control_panel(self):
        """Render unified control panel with simulation controls and joint sliders."""
        imgui = self._imgui
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.always_auto_resize)

        # Simulation controls section
        if imgui.button("Pause" if not self.paused else "Play", size=(60, 0)):
            self.paused = not self.paused
        imgui.same_line()
        if imgui.button("Step", size=(50, 0)):
            self._step_requested = True
        imgui.same_line()
        if imgui.button("Reset", size=(50, 0)):
            with self.viewer.render_lock:
                self.scene.reset()

        if hasattr(self.scene, "t"):
            imgui.text(f"Time: {self.scene.t:.3f}s")

        if hasattr(self.scene, "n_envs") and self.scene.n_envs > 1:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), f"Note: Controlling env 0 of {self.scene.n_envs}")

        imgui.separator()

        # Joint control section
        if not self._entity_cache:
            imgui.text("No controllable entities")
            imgui.end()
            return

        if not self.paused:
            imgui.text_colored((0.7, 0.7, 0.7, 1.0), "Pause to edit joints")

        for entity_idx, data in self._entity_cache.items():
            entity = data["entity"]
            expanded = imgui.collapsing_header(data["name"])
            if not expanded:
                continue

            # Get qpos - handle multi-env case by using only env 0
            qpos_tensor = entity.get_qpos()
            qpos_np = qpos_tensor.cpu().numpy()

            # If multi-env (2D tensor with shape [n_envs, n_qs]), use only env 0
            is_multi_env = qpos_np.ndim == 2
            if is_multi_env:
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
                    self._apply_qpos_update(entity, new_qpos, is_multi_env)

        imgui.end()

    def should_step(self) -> bool:
        """Check if simulation should advance this frame."""
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
