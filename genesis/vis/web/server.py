"""Genesis Web GUI server.

Runs a FastAPI + uvicorn server in a background daemon thread.
Bridges the Genesis scene to browser clients via WebSocket:
- Binary messages: JPEG frames (server -> client)
- JSON text messages: control commands and state updates (bidirectional)
"""

import asyncio
import json
import os
import threading
import time

import numpy as np

import genesis as gs

from genesis.vis.scene_ops import (
    refresh_visual_transforms,
    set_entity_contact_viz,
    set_entity_wireframe,
    switch_entity_vis_mode,
)

from .frame_producer import FrameProducer
from .protocol import MsgType, build_scene_info, build_state_update

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class GenesisWebServer:
    """Web server that bridges a Genesis scene to browser clients.

    Usage::

        scene = gs.Scene(show_viewer=False)
        # ... add entities, build ...

        web = GenesisWebServer(scene, port=8765)
        web.start()

        while True:
            web.process_commands()
            if web.should_step():
                scene.step()
            web.produce_frame()
    """

    def __init__(self, scene, host="0.0.0.0", port=8765, resolution=(1280, 720), jpeg_quality=80):
        self.scene = scene
        self.host = host
        self.port = port

        self.producer = FrameProducer(scene, resolution, jpeg_quality)

        self._clients = []
        self._clients_lock = threading.Lock()
        self._loop = None
        self._thread = None

        # Cooperative sim control (same pattern as ImGui overlay)
        self._paused = False
        self._step_requested = False
        self._reset_requested = False

        # Pending commands from web clients (consumed on main thread)
        self._pending_commands = []
        self._commands_lock = threading.Lock()

        # FPS tracking
        self._frame_count = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0

        self._app = None
        self._server = None

        # Initial camera state for reset support
        self._initial_camera_pos = None
        self._initial_camera_lookat = None
        self._initial_camera_fov = None

        # Per-entity wireframe tracking
        self._entity_wireframe = {}  # entity_idx -> bool

    def _build_app(self):
        """Create the FastAPI application. Called lazily to defer import."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        app = FastAPI()

        @app.websocket("/ws")
        async def ws_handler(ws: WebSocket):
            await ws.accept()
            with self._clients_lock:
                self._clients.append(ws)
            try:
                # Send scene info on connect
                try:
                    scene_info = build_scene_info(self.scene)
                    await ws.send_text(json.dumps(scene_info))
                except Exception as e:
                    gs.logger.warning(f"Failed to build scene_info: {e}")
                # Handle incoming control messages
                async for raw in ws.iter_text():
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self._enqueue_command(msg)
            except WebSocketDisconnect:
                pass
            finally:
                with self._clients_lock:
                    if ws in self._clients:
                        self._clients.remove(ws)

        # Serve index.html at root
        @app.get("/")
        async def index():
            return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

        # Serve other static files
        if os.path.isdir(_STATIC_DIR):
            app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        self._app = app
        return app

    def _enqueue_command(self, msg):
        """Thread-safe: web clients enqueue commands for main thread."""
        with self._commands_lock:
            self._pending_commands.append(msg)

    # ------------------------------------------------------------------
    # Main thread API
    # ------------------------------------------------------------------

    def should_step(self):
        """Poll from main loop. Returns True if the simulation should advance."""
        if self._reset_requested:
            self._reset_requested = False
            self.scene.reset()
        if self._step_requested:
            self._step_requested = False
            return True
        return not self._paused

    def process_commands(self):
        """Called from main thread to apply pending web client commands."""
        with self._commands_lock:
            commands = self._pending_commands[:]
            self._pending_commands.clear()
        for cmd in commands:
            self._handle_command(cmd)

    def produce_frame(self):
        """Called from main thread after scene.step().

        Captures a frame and schedules broadcast to all WebSocket clients.
        """
        self.producer.produce_frame()
        self._update_fps()

        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._broadcast_frame()),
            )

    def _update_fps(self):
        """Update FPS counter."""
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

    # ------------------------------------------------------------------
    # Async helpers (run on the uvicorn event loop)
    # ------------------------------------------------------------------

    async def _broadcast_frame(self):
        """Send latest frame to all connected clients."""
        frame = self.producer.consume_frame()
        if frame is None:
            return

        # Also build a state update
        sim_time = float(self.scene.cur_t) if self.scene.is_built else 0.0
        step = int(self.scene.t) if self.scene.is_built else 0

        camera_state = None
        try:
            camera = self.scene.visualizer.cameras[0]
            pos = camera.pos
            lookat = camera.lookat
            camera_state = {
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "lookat": [float(lookat[0]), float(lookat[1]), float(lookat[2])],
                "fov": float(getattr(camera, "fov", 30.0)),
            }
            # Send actual view/projection matrices for accurate gizmo rendering.
            # This avoids reconstruction errors from pos/lookat/up ambiguity.
            try:
                view_mat = np.linalg.inv(camera.transform)
                # Transpose to column-major for JavaScript (OpenGL convention)
                camera_state["view_matrix"] = view_mat.T.flatten().tolist()
            except Exception:
                gs.logger.debug("Failed to compute view matrix", exc_info=True)
            try:
                rasterizer = self.scene.visualizer._rasterizer
                cam_node = rasterizer._camera_nodes[camera.uid]
                proj = cam_node.camera.get_projection_matrix(width=camera.res[0], height=camera.res[1])
                camera_state["proj_matrix"] = proj.T.flatten().tolist()
            except Exception:
                gs.logger.debug("Failed to compute projection matrix", exc_info=True)
        except Exception:
            gs.logger.debug("Failed to build camera state", exc_info=True)

        state_json = json.dumps(
            build_state_update(
                sim_time=float(sim_time),
                step=int(step),
                fps=self._fps,
                paused=self._paused,
                camera_state=camera_state,
            )
        )

        with self._clients_lock:
            clients = self._clients[:]

        async def _send_to_client(ws):
            try:
                await ws.send_bytes(frame)
                await ws.send_text(state_json)
            except Exception:
                # Client disconnected; remove from active list
                with self._clients_lock:
                    if ws in self._clients:
                        self._clients.remove(ws)

        await asyncio.gather(*(_send_to_client(ws) for ws in clients), return_exceptions=True)

    # ------------------------------------------------------------------
    # Command handling (runs on main thread)
    # ------------------------------------------------------------------

    def _handle_command(self, cmd):
        """Apply a single command dict on the main thread."""
        msg_type = cmd.get("type")

        if msg_type == MsgType.SIM_CONTROL:
            self._handle_sim_control(cmd)
        elif msg_type == MsgType.CAMERA_UPDATE:
            self._handle_camera_update(cmd)
        elif msg_type == MsgType.ENTITY_UPDATE:
            self._handle_entity_update(cmd)
        elif msg_type == MsgType.VIS_TOGGLE:
            self._handle_vis_toggle(cmd)

    def _handle_sim_control(self, cmd):
        action = cmd.get("action")
        if action == "pause":
            self._paused = True
        elif action == "play":
            self._paused = False
        elif action == "step":
            self._step_requested = True
        elif action == "reset":
            self._reset_requested = True

    def _handle_camera_update(self, cmd):
        """Apply camera manipulation commands."""
        if not self.scene.visualizer.cameras:
            return
        camera = self.scene.visualizer.cameras[0]
        action = cmd.get("action")

        if action == "orbit":
            self._apply_orbit(camera, cmd)
        elif action == "pan":
            self._apply_pan(camera, cmd)
        elif action == "zoom":
            self._apply_zoom(camera, cmd)
        elif action == "set_pose":
            pos = cmd.get("pos")
            lookat = cmd.get("lookat")
            if pos is not None or lookat is not None:
                camera.set_pose(pos=pos, lookat=lookat)
        elif action == "set_fov":
            fov = cmd.get("fov")
            if fov is not None:
                try:
                    camera._fov = float(fov)
                except Exception:
                    gs.logger.debug("Failed to set camera FOV", exc_info=True)
        elif action == "reset":
            try:
                if self._initial_camera_pos is not None:
                    camera.set_pose(
                        pos=self._initial_camera_pos.copy(),
                        lookat=self._initial_camera_lookat.copy(),
                    )
                if self._initial_camera_fov is not None and hasattr(camera, "fov"):
                    camera.fov = self._initial_camera_fov
            except Exception:
                gs.logger.debug("Failed to reset camera", exc_info=True)

    def _apply_orbit(self, camera, cmd):
        """Orbit the camera around its lookat point."""
        d_azimuth = cmd.get("d_azimuth", 0.0)
        d_elevation = cmd.get("d_elevation", 0.0)
        if d_azimuth == 0.0 and d_elevation == 0.0:
            return

        pos = camera.pos.copy()
        lookat = camera.lookat.copy()

        # Vector from lookat to camera
        offset = pos - lookat
        radius = np.linalg.norm(offset)
        if radius < 1e-8:
            return

        # Current spherical angles
        theta = np.arctan2(offset[0], offset[1])  # azimuth
        phi = np.arcsin(np.clip(offset[2] / radius, -1.0, 1.0))  # elevation

        # Apply deltas
        theta += d_azimuth
        phi = np.clip(phi + d_elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

        # Convert back to Cartesian
        new_offset = np.array(
            [
                radius * np.cos(phi) * np.sin(theta),
                radius * np.cos(phi) * np.cos(theta),
                radius * np.sin(phi),
            ],
            dtype=np.float32,
        )

        camera.set_pose(pos=lookat + new_offset, lookat=lookat)

    def _apply_pan(self, camera, cmd):
        """Pan the camera (translate both pos and lookat)."""
        dx = cmd.get("dx", 0.0)
        dy = cmd.get("dy", 0.0)
        if dx == 0.0 and dy == 0.0:
            return

        pos = camera.pos.copy()
        lookat = camera.lookat.copy()
        up = camera.up.copy()

        forward = lookat - pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(right, forward)

        offset = right * dx + cam_up * dy
        camera.set_pose(pos=pos + offset, lookat=lookat + offset)

    def _apply_zoom(self, camera, cmd):
        """Zoom by moving the camera closer/farther from lookat."""
        factor = cmd.get("factor", 1.0)
        if factor == 1.0:
            return

        pos = camera.pos.copy()
        lookat = camera.lookat.copy()

        offset = pos - lookat
        camera.set_pose(pos=lookat + offset * factor, lookat=lookat)

    def _handle_entity_update(self, cmd):
        """Apply entity state changes (qpos, vis_mode, wireframe, contacts)."""
        entity_idx = cmd.get("entity_idx")
        # Use type() not isinstance() to reject bool (bool is a subclass of int).
        # None is also rejected here (type(None) is NoneType, not int).
        if type(entity_idx) is not int:
            gs.logger.warning(f"Invalid entity_idx type: {type(entity_idx).__name__}")
            return

        entities = self.scene.entities
        if entity_idx < 0 or entity_idx >= len(entities):
            return

        entity = entities[entity_idx]

        # Per-entity vis mode switch
        vis_mode = cmd.get("vis_mode")
        if vis_mode is not None:
            if vis_mode not in ("visual", "collision"):
                gs.logger.warning(f"Invalid vis_mode: {vis_mode!r}")
                return
            self._switch_entity_vis_mode(entity, vis_mode)
            return

        # Per-entity wireframe toggle
        wireframe = cmd.get("wireframe")
        if wireframe is not None:
            self._set_entity_wireframe(entity, entity_idx, bool(wireframe))
            return

        # Contact visualization toggle
        contact_viz = cmd.get("contact_viz")
        if contact_viz is not None:
            self._set_entity_contact_viz(entity, bool(contact_viz))
            return

        # DOF/qpos updates require set_qpos
        if not hasattr(entity, "set_qpos"):
            return

        # Full qpos update
        qpos = cmd.get("qpos")
        if qpos is not None:
            if not isinstance(qpos, list):
                gs.logger.warning(f"Invalid qpos type: {type(qpos).__name__}")
                return
            # Quaternion normalization
            quat_groups = cmd.get("quat_groups")
            if quat_groups and cmd.get("normalize_quats"):
                qpos = list(qpos)
                for start, end in quat_groups:
                    q = np.array(qpos[start:end])
                    norm = np.linalg.norm(q)
                    if norm > 1e-8:
                        q /= norm
                        qpos[start:end] = q.tolist()
            self._paused = True  # Auto-pause on manual qpos change
            is_multi_env = self.scene.n_envs > 1
            if is_multi_env:
                entity.set_qpos(qpos, envs_idx=0)
            else:
                entity.set_qpos(qpos)
            self._update_visual_transforms()
            return

        # Single DOF slider update
        dof_idx = cmd.get("dof_idx")
        value = cmd.get("value")
        if dof_idx is not None and value is not None:
            if type(dof_idx) is not int:
                gs.logger.warning(f"Invalid dof_idx type: {type(dof_idx).__name__}")
                return
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                gs.logger.warning(f"Invalid value type: {type(value).__name__}")
                return

            self._paused = True  # Auto-pause on manual DOF change
            is_multi_env = self.scene.n_envs > 1
            current = entity.get_qpos()
            if is_multi_env:
                # Extract env-0 from batched [n_envs, n_qs] result
                current = current[0]
            new_qpos = current.tolist() if hasattr(current, "tolist") else list(current)
            if 0 <= dof_idx < len(new_qpos):
                new_qpos[dof_idx] = value
                if is_multi_env:
                    entity.set_qpos(new_qpos, envs_idx=0)
                else:
                    entity.set_qpos(new_qpos)
                self._update_visual_transforms()

    def _get_ctx(self):
        """Get the RasterizerContext, or None if unavailable."""
        try:
            return self.scene.visualizer._rasterizer._context
        except Exception:
            return None

    def _update_visual_transforms(self):
        """Update render transforms so visuals reflect the latest qpos immediately."""
        try:
            ctx = self._get_ctx()
            if ctx is not None:
                refresh_visual_transforms(self.scene, ctx)
        except Exception:
            gs.logger.debug("Failed to update visual transforms", exc_info=True)

    def _switch_entity_vis_mode(self, entity, new_mode):
        """Switch entity between 'visual' and 'collision' rendering."""
        try:
            ctx = self._get_ctx()
            if ctx is not None:
                switch_entity_vis_mode(self.scene, ctx, entity, new_mode)
        except Exception:
            gs.logger.debug("Failed to switch entity vis mode", exc_info=True)

    def _set_entity_wireframe(self, entity, entity_idx, enable):
        """Toggle wireframe rendering for all geom nodes of an entity."""
        try:
            ctx = self._get_ctx()
            if ctx is not None:
                self._entity_wireframe[entity_idx] = enable
                set_entity_wireframe(ctx, entity, enable)
        except Exception:
            gs.logger.debug("Failed to set entity wireframe", exc_info=True)

    def _set_entity_contact_viz(self, entity, enable):
        """Toggle contact visualization for an entity and its links."""
        try:
            set_entity_contact_viz(entity, enable)
        except Exception:
            gs.logger.debug("Failed to set contact visualization", exc_info=True)

    def _handle_vis_toggle(self, cmd):
        """Toggle visualization options via the rasterizer context."""
        prop = cmd.get("property")
        value = cmd.get("value")
        if prop is None or value is None:
            return

        ctx = self.scene.visualizer._rasterizer._context

        if prop == "shadows":
            ctx.shadow = bool(value)
        elif prop == "wireframe":
            self._toggle_wireframe(bool(value))
        elif prop == "world_frame":
            if value:
                ctx.on_world_frame()
            else:
                ctx.off_world_frame()
        elif prop == "link_frame":
            if value:
                ctx.on_link_frame()
                # Also update positions to current state
                ctx.update_link_frame(ctx.buffer)
            else:
                ctx.off_link_frame()
        elif prop == "camera_frustum":
            if value:
                ctx.on_camera_frustum()
            else:
                ctx.off_camera_frustum()
        elif prop == "face_normals":
            self._toggle_render_flag("face_normals", bool(value))
        elif prop == "vertex_normals":
            self._toggle_render_flag("vertex_normals", bool(value))
        elif prop == "orthographic":
            self._toggle_orthographic(bool(value))
        elif prop == "link_frame_size":
            try:
                new_size = float(value)
                if ctx.link_frame_size > 0:
                    scale = new_size / ctx.link_frame_size
                    ctx.link_frame_mesh.vertices *= scale
                    ctx.link_frame_size = new_size
                    if ctx.link_frame_shown:
                        ctx.off_link_frame()
                        ctx.on_link_frame()
            except Exception:
                gs.logger.debug("Failed to resize link frame", exc_info=True)

    def _toggle_wireframe(self, enable):
        """Toggle wireframe rendering for all mesh primitives."""
        ctx = self.scene.visualizer._rasterizer._context
        for node in ctx._scene.mesh_nodes:
            for primitive in node.mesh.primitives:
                if primitive.material is not None:
                    primitive.material.wireframe = enable
        # Signal JIT renderer to rebuild cached render_flags
        ctx._scene._meshes_updated = True
        # Global toggle overrides per-entity state
        self._entity_wireframe.clear()

    def _toggle_render_flag(self, flag_name, enable):
        """Toggle face_normals or vertex_normals render flags on the rasterizer context."""
        from genesis.ext.pyrender.constants import RenderFlags

        flag_map = {
            "face_normals": RenderFlags.FACE_NORMALS,
            "vertex_normals": RenderFlags.VERTEX_NORMALS,
        }
        flag = flag_map.get(flag_name)
        if flag is None:
            return
        try:
            ctx = self.scene.visualizer._rasterizer._context
            current = getattr(ctx, "_extra_render_flags", RenderFlags.NONE)
            if enable:
                ctx._extra_render_flags = current | flag
            else:
                ctx._extra_render_flags = current & ~flag
        except Exception:
            gs.logger.debug("Failed to toggle render flag", exc_info=True)

    def _toggle_orthographic(self, enable):
        """Switch between perspective and orthographic projection."""
        try:
            camera = self.scene.visualizer.cameras[0]
            rasterizer = self.scene.visualizer._rasterizer
            camera_node = rasterizer._camera_nodes[camera.uid]

            if enable:
                # Compute orthographic magnification from current perspective view
                pos = camera.pos
                lookat = camera.lookat
                distance = float(np.linalg.norm(pos - lookat))
                half_height = distance * np.tan(np.deg2rad(camera.fov / 2.0))
                half_width = half_height * camera.aspect_ratio

                # Store the original perspective camera for later restoration
                self._perspective_camera = camera_node.camera

                from genesis.ext.pyrender import OrthographicCamera

                camera_node.camera = OrthographicCamera(
                    xmag=half_width,
                    ymag=half_height,
                    znear=camera.near,
                    zfar=camera.far,
                )
            else:
                # Restore perspective camera
                if hasattr(self, "_perspective_camera") and self._perspective_camera is not None:
                    camera_node.camera = self._perspective_camera
                    self._perspective_camera = None
        except Exception:
            gs.logger.debug("Failed to toggle orthographic projection", exc_info=True)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _capture_initial_camera(self):
        """Snapshot the camera pose and FOV so we can restore on reset."""
        try:
            camera = self.scene.visualizer.cameras[0]
            self._initial_camera_pos = camera.pos.copy()
            self._initial_camera_lookat = camera.lookat.copy()
            self._initial_camera_fov = getattr(camera, "fov", 30.0)
        except Exception:
            gs.logger.debug("Failed to capture initial camera state", exc_info=True)

    def start(self):
        """Start the web server in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._capture_initial_camera()
        app = self._build_app()

        def _run():
            import uvicorn

            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                loop="asyncio",
                log_level="warning",
            )
            self._server = uvicorn.Server(config)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="genesis-web")
        self._thread.start()

        gs.logger.info(f"Genesis Web GUI started at http://{self.host}:{self.port}")

    def stop(self):
        """Signal the server to shut down."""
        if self._server is not None:
            self._server.should_exit = True
