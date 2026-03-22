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

from .frame_producer import FrameProducer
from .protocol import (
    build_scene_info,
    build_state_update,
    parse_client_message,
    sanitize_floats,
    CameraUpdateMsg,
    EntityUpdateMsg,
    SetResolutionMsg,
    SetTargetFpsMsg,
    SimControlMsg,
    VisToggleMsg,
)

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

        # Target FPS limiter (0 = unlimited)
        self._target_fps = 0
        self._last_frame_time = 0.0

        self._app = None
        self._server = None

        # Initial camera state for reset support
        self._initial_camera_pos = None
        self._initial_camera_lookat = None
        self._initial_camera_fov = None

        # Orthographic toggle state (frontend-specific, not on controller)
        self._perspective_camera = None

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
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    parsed = parse_client_message(data)
                    if parsed is not None:
                        self._enqueue_command(parsed)
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
        Respects target FPS by sleeping if producing frames too fast.
        """
        # Target FPS limiter
        if self._target_fps > 0:
            now = time.monotonic()
            min_interval = 1.0 / self._target_fps
            elapsed = now - self._last_frame_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_frame_time = time.monotonic()

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

        # Use controller for camera state
        camera_state = self.scene.controller.get_scene_camera_state() or None

        # Gather current entity qpos for gizmo tracking
        entity_positions = None
        try:
            ep = []
            for entity in self.scene.entities:
                if hasattr(entity, "n_dofs") and entity.n_dofs > 0:
                    qpos = entity.get_qpos()
                    if hasattr(qpos, "cpu"):
                        qpos = qpos.cpu().numpy()
                    ep.append(
                        {
                            "idx": entity.idx,
                            "qpos": sanitize_floats(qpos).tolist(),
                        }
                    )
            if ep:
                entity_positions = ep
        except Exception:
            pass

        state_json = json.dumps(
            build_state_update(
                sim_time=float(sim_time),
                step=int(step),
                fps=self._fps,
                paused=self._paused,
                camera_state=camera_state,
                entity_positions=entity_positions,
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
        """Apply a single validated command on the main thread."""
        if isinstance(cmd, SimControlMsg):
            self._handle_sim_control(cmd)
        elif isinstance(cmd, CameraUpdateMsg):
            self._handle_camera_update(cmd)
        elif isinstance(cmd, EntityUpdateMsg):
            self._handle_entity_update(cmd)
        elif isinstance(cmd, VisToggleMsg):
            self._handle_vis_toggle(cmd)
        elif isinstance(cmd, SetResolutionMsg):
            self._handle_set_resolution(cmd)
        elif isinstance(cmd, SetTargetFpsMsg):
            self._handle_set_target_fps(cmd)

    def _handle_sim_control(self, cmd: SimControlMsg):
        if cmd.action == "pause":
            self._paused = True
        elif cmd.action == "play":
            self._paused = False
        elif cmd.action == "step":
            self._step_requested = True
        elif cmd.action == "reset":
            self._reset_requested = True

    def _handle_set_resolution(self, cmd: SetResolutionMsg):
        """Update the render resolution from the client viewport size."""
        # Clamp to reasonable range
        width = max(320, min(3840, cmd.width))
        height = max(240, min(2160, cmd.height))
        try:
            camera = self.scene.visualizer.cameras[0]
            camera.set_resolution((width, height))
        except Exception:
            gs.logger.debug("Failed to set resolution")

    def _handle_set_target_fps(self, cmd: SetTargetFpsMsg):
        """Set the target FPS limiter."""
        self._target_fps = max(0, min(240, cmd.fps))

    def _handle_camera_update(self, cmd: CameraUpdateMsg):
        """Apply camera manipulation commands (frontend-specific camera math)."""
        if not self.scene.visualizer.cameras:
            return
        camera = self.scene.visualizer.cameras[0]

        if cmd.action == "orbit":
            self._apply_orbit(camera, cmd)
        elif cmd.action == "pan":
            self._apply_pan(camera, cmd)
        elif cmd.action == "zoom":
            self._apply_zoom(camera, cmd)
        elif cmd.action == "set_pose":
            if cmd.pos is not None or cmd.lookat is not None:
                camera.set_pose(pos=cmd.pos, lookat=cmd.lookat)
        elif cmd.action == "set_fov":
            if cmd.fov is not None:
                self.scene.controller.set_scene_camera_fov(cmd.fov)
        elif cmd.action == "reset":
            try:
                if self._initial_camera_pos is not None:
                    camera.set_pose(
                        pos=self._initial_camera_pos.copy(),
                        lookat=self._initial_camera_lookat.copy(),
                    )
                if self._initial_camera_fov is not None:
                    self.scene.controller.set_scene_camera_fov(self._initial_camera_fov)
            except Exception:
                gs.logger.debug("Failed to reset camera")

    def _apply_orbit(self, camera, cmd: CameraUpdateMsg):
        """Orbit the camera around its lookat point."""
        d_azimuth = cmd.d_azimuth if cmd.d_azimuth is not None else 0.0
        d_elevation = cmd.d_elevation if cmd.d_elevation is not None else 0.0
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

        camera.set_pose(pos=lookat + new_offset, lookat=lookat, up=np.array([0, 0, 1], dtype=np.float32))

    def _apply_pan(self, camera, cmd: CameraUpdateMsg):
        """Pan the camera (translate both pos and lookat)."""
        dx = cmd.dx if cmd.dx is not None else 0.0
        dy = cmd.dy if cmd.dy is not None else 0.0
        if dx == 0.0 and dy == 0.0:
            return

        pos = camera.pos.copy()
        lookat = camera.lookat.copy()
        up = np.array([0, 0, 1], dtype=np.float32)

        forward = lookat - pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        cam_up = np.cross(right, forward)

        offset = right * dx + cam_up * dy
        camera.set_pose(pos=pos + offset, lookat=lookat + offset, up=up)

    def _apply_zoom(self, camera, cmd: CameraUpdateMsg):
        """Zoom by moving the camera closer/farther from lookat."""
        factor = cmd.factor if cmd.factor is not None else 1.0
        if factor == 1.0:
            return

        pos = camera.pos.copy()
        lookat = camera.lookat.copy()

        offset = pos - lookat
        camera.set_pose(pos=lookat + offset * factor, lookat=lookat)

    def _handle_entity_update(self, cmd: EntityUpdateMsg):
        """Apply entity state changes (qpos, vis_mode, wireframe, contacts)."""
        entity_idx = cmd.entity_idx
        entities = self.scene.entities
        if entity_idx < 0 or entity_idx >= len(entities):
            return

        entity = entities[entity_idx]
        ctrl = self.scene.controller

        # Per-entity vis mode switch
        if cmd.vis_mode is not None:
            ctrl.switch_entity_vis_mode(entity, cmd.vis_mode)
            return

        # Per-entity wireframe toggle
        if cmd.wireframe is not None:
            ctrl.set_entity_wireframe(entity, cmd.wireframe)
            return

        # Contact visualization toggle
        if cmd.contact_viz is not None:
            ctrl.set_entity_contact_viz(entity, cmd.contact_viz)
            return

        # DOF/qpos updates require set_qpos
        if not hasattr(entity, "set_qpos"):
            return

        # Full qpos update
        if cmd.qpos is not None:
            qpos = cmd.qpos
            # Quaternion normalization
            if cmd.quat_groups and cmd.normalize_quats:
                qpos = list(qpos)
                for start, end in cmd.quat_groups:
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
            ctrl.refresh_visual_transforms()
            return

        # Single DOF slider update
        if cmd.dof_idx is not None and cmd.value is not None:
            self._paused = True  # Auto-pause on manual DOF change
            is_multi_env = self.scene.n_envs > 1
            current = entity.get_qpos()
            if is_multi_env:
                # Extract env-0 from batched [n_envs, n_qs] result
                current = current[0]
            new_qpos = current.tolist() if hasattr(current, "tolist") else list(current)
            if 0 <= cmd.dof_idx < len(new_qpos):
                new_qpos[cmd.dof_idx] = cmd.value
                if is_multi_env:
                    entity.set_qpos(new_qpos, envs_idx=0)
                else:
                    entity.set_qpos(new_qpos)
                ctrl.refresh_visual_transforms()

    def _handle_vis_toggle(self, cmd: VisToggleMsg):
        """Toggle visualization options via scene.controller."""
        prop = cmd.property
        value = cmd.value
        ctrl = self.scene.controller

        if prop == "shadows":
            ctrl.set_shadows(bool(value))
        elif prop == "wireframe":
            ctrl.set_wireframe(bool(value))
        elif prop == "world_frame":
            ctrl.set_world_frame(bool(value))
        elif prop == "link_frame":
            ctrl.set_link_frame(bool(value))
        elif prop == "camera_frustum":
            ctrl.set_camera_frustum(bool(value))
        elif prop == "face_normals":
            ctrl.set_face_normals(bool(value))
        elif prop == "vertex_normals":
            ctrl.set_vertex_normals(bool(value))
        elif prop == "orthographic":
            self._toggle_orthographic(bool(value))
        elif prop == "link_frame_size":
            try:
                ctrl.set_link_frame_size(float(value))
            except Exception:
                gs.logger.debug("Failed to resize link frame")

    def _toggle_orthographic(self, enable):
        """Switch between perspective and orthographic projection (frontend-specific)."""
        try:
            camera = self.scene.visualizer.cameras[0]
            rasterizer = self.scene.visualizer._rasterizer
            camera_node = rasterizer._camera_nodes[camera.uid]

            if enable:
                pos = camera.pos
                lookat = camera.lookat
                distance = float(np.linalg.norm(pos - lookat))
                half_height = distance * np.tan(np.deg2rad(camera.fov / 2.0))
                half_width = half_height * camera.aspect_ratio

                self._perspective_camera = camera_node.camera

                from genesis.ext.pyrender import OrthographicCamera

                camera_node.camera = OrthographicCamera(
                    xmag=half_width,
                    ymag=half_height,
                    znear=camera.near,
                    zfar=camera.far,
                )
            else:
                if self._perspective_camera is not None:
                    camera_node.camera = self._perspective_camera
                    self._perspective_camera = None
        except Exception:
            gs.logger.debug("Failed to toggle orthographic projection")

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
            gs.logger.debug("Failed to capture initial camera state")

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
