# Genesis Web GUI

A browser-based GUI for the Genesis physics simulator. Streams rendered frames via WebSocket and provides interactive controls for simulation, camera, entities, and gizmo manipulation.

## Running the Web GUI

```bash
# Basic usage (GPU backend)
python examples/gui/web_gui_demo.py

# CPU backend
python examples/gui/web_gui_demo.py --cpu

# Custom port
python examples/gui/web_gui_demo.py --port 9000
```

Then open `http://localhost:8765` (or your custom port) in a browser.

### Requirements

- Genesis installed (`pip install genesis-world`)
- A scene with `show_viewer=False` and at least one camera
- No Node.js needed for end users — pre-built frontend assets are shipped in the package

### Integration in Your Script

```python
import genesis as gs
from genesis.vis.web.server import GenesisWebServer

gs.init()
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.add_camera(res=(1280, 720), pos=(2, 2, 1.5), lookat=(0, 0, 0.5), fov=30)
scene.build()

web = GenesisWebServer(scene, port=8765)
web.start()

while True:
    web.process_commands()   # Apply client commands (camera, joints, sim control)
    if web.should_step():    # Respects pause/step/reset from the GUI
        scene.step()
    web.produce_frame()      # Capture and broadcast a JPEG frame
```

## Architecture

```
Browser (Vue 3)                    Python (FastAPI + uvicorn)
┌─────────────┐                    ┌──────────────────────┐
│  Frontend   │◄── WebSocket ──►   │  GenesisWebServer    │
│  (Vue SFCs) │    /ws             │                      │
│             │  binary: JPEG ◄──  │  FrameProducer       │
│             │  JSON: commands ──►│  (JPEG encoder)      │
│             │  JSON: state    ◄──│                      │
└─────────────┘                    │  protocol.py         │
      │                            │  (message types)     │
      │ served at /                └──────────┬───────────┘
      │ assets at /static/                    │
      │                                       │
      ▼                               Genesis Scene
  Static files                     (entities, camera,
  (Vite build output)              rasterizer, physics)
```

### Communication Protocol

All communication happens over a single WebSocket connection at `/ws`:

**Binary messages (server → client):** JPEG-encoded camera frames, decoded via `createImageBitmap()` and drawn on a `<canvas>` element using `requestAnimationFrame`.

**JSON messages (bidirectional):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `scene_info` | server → client | Initial scene state: entities, joints, vis settings, camera |
| `state_update` | server → client | Per-frame: sim time, step, FPS, paused, camera pose/matrices, entity qpos |
| `sim_control` | client → server | Play, pause, step, reset |
| `camera_update` | client → server | Orbit, pan, zoom, set pose, set FOV, reset |
| `entity_update` | client → server | DOF slider, full qpos, vis mode, wireframe, contacts |
| `vis_toggle` | client → server | Toggle shadows, world frame, link frame, wireframe, orthographic |

### Backend (`genesis/vis/web/`)

| File | Purpose |
|------|---------|
| `server.py` | `GenesisWebServer` — FastAPI app, WebSocket handler, command routing, camera/entity manipulation |
| `protocol.py` | Message types (`MsgType` enum), `build_scene_info()`, `build_state_update()` |
| `frame_producer.py` | Thread-safe JPEG frame capture from the Genesis rasterizer |

The server runs on a background daemon thread. Commands from WebSocket clients are enqueued and processed on the main thread via `process_commands()`. Frame capture and broadcasting happen via `produce_frame()`.

### Frontend (`genesis/vis/web/frontend/`)

Built with **Vue 3 + Vite**. Plain JavaScript (no TypeScript). Pre-built output committed to `genesis/vis/web/static/`.

#### Composables (state management via `provide/inject`)

| File | Purpose |
|------|---------|
| `useWebSocket.js` | Connection, exponential backoff reconnect (500ms → 30s), binary/JSON dispatch, frame generation token for stale decode invalidation |
| `useScene.js` | Reactive scene state: entities, selection, sim status, vis state, entity edit state, euler/quat synchronization |
| `useCamera.js` | Camera pos/lookat/fov with focus guard (skip server updates while user edits inputs) |
| `useGizmo.js` | 3D manipulation gizmo: translate/rotate modes, drawing, hit testing, drag interaction, background canvas compositing |

#### Components

| File | Purpose |
|------|---------|
| `App.vue` | Root layout: full-screen viewport with floating glass panels |
| `ViewportCanvas.vue` | Canvas with `requestAnimationFrame` render loop, camera mouse handlers (orbit/pan/zoom), gizmo interaction priority |
| `LeftPanel.vue` | Entity tree with inline joint controls: ScrubInput sliders for DOFs, quat/euler rotation toggle, free-joint position editing |
| `RightPanel.vue` | Camera properties (ScrubInput for pos/lookat/fov), global vis toggles (segmented pill buttons), per-entity vis mode/wireframe/contacts/gizmo |
| `FloatingSimBar.vue` | Bottom center bar: play/pause/step/reset controls + live status (connection, FPS, time, step) |
| `SegmentedToggle.vue` | Reusable pill-style toggle button (Yes/No, Visual/Collision, etc.) |
| `ScrubInput.vue` | Drag-to-adjust value control: drag horizontally to scrub, double-click to type. Supports bounded (fill shows position) and unbounded (center-reset) modes |

#### Utilities

| File | Purpose |
|------|---------|
| `gizmoMath.js` | Pure functions: vec3/mat4/quaternion operations, 3D→2D projection, euler↔quaternion conversion |
| `protocol.js` | Message type and action constants matching the server's `protocol.py` |

### Build Pipeline

```bash
cd genesis/vis/web/frontend
npm install        # Install dependencies (dev only)
npm run dev        # Dev server with HMR (proxies /ws to localhost:8765)
npm run build      # Production build → ../static/
npm test           # Run Vitest tests (62 tests)
```

- Vite `base: '/static/'` so built HTML references assets at `/static/assets/*`
- FastAPI serves `static/index.html` at `/` and mounts `static/` at `/static`
- `node_modules/` is gitignored; built assets in `static/` are committed
- End users need no Node.js — `pip install genesis-world` includes the pre-built frontend

### UI Design

The frontend uses a **frosted glass** design with semi-transparent panels floating over the full-screen rendered viewport:

- **Left panel**: Entity tree with expandable inline joint controls
- **Right panel**: Camera properties, global visualization settings, per-entity visual properties
- **Bottom bar**: Simulation controls (play/pause/step/reset) + live status
- **Theme**: Dark viewport, light semi-transparent panels (`backdrop-filter: blur`), Inter font, blue accent (#4b7cf3)
