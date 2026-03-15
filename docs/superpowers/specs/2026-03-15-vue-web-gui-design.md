# Genesis Web GUI — Vue 3 Frontend Redesign

## Problem Statement

The current Genesis Web GUI is a single monolithic `index.html` (~1200+ lines) with inline CSS and vanilla JavaScript. While functional, it lacks:

- **Maintainability**: All UI logic, styling, and state management in one file
- **Visual polish**: Basic dark theme with traditional toggle switches and flat tabs
- **Scalability**: Adding new features means appending to an already long file

The goal is to rebuild the frontend using Vue 3 with a modern, Spline-inspired design while keeping the Python backend (FastAPI + WebSocket) completely unchanged.

## Design Goals

1. **Modern, polished UI** inspired by Spline 3D editor — rounded shapes, dark theme, segmented pill toggles, floating controls
2. **Three-panel layout** — left entity tree, center viewport, right properties panel
3. **100% feature parity** with the current `index.html` — no features dropped
4. **Zero impact on end users** — pre-built assets committed to `static/`, no Node.js required to use Genesis
5. **Clean component architecture** — composables for state, single-file components for UI

## Architecture

### Project Structure

```
genesis/vis/web/
├── frontend/                    # Vue 3 + Vite project (dev only)
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html               # Vite entry point
│   ├── src/
│   │   ├── App.vue              # Root 3-panel layout
│   │   ├── main.js              # Vue app bootstrap + WebSocket init
│   │   ├── composables/
│   │   │   ├── useWebSocket.js  # Connection, reconnect, message dispatch
│   │   │   ├── useScene.js      # Reactive scene state
│   │   │   ├── useCamera.js     # Camera state + commands
│   │   │   └── useGizmo.js      # Gizmo drawing, hit testing, drag
│   │   ├── components/          # Phase 1: minimal set (see Decomposition Strategy)
│   │   │   ├── LeftPanel.vue    # Entity tree + scene hierarchy
│   │   │   ├── RightPanel.vue   # Properties for selected entity, camera, vis toggles
│   │   │   └── ViewportCanvas.vue # Canvas + floating sim bar + floating status
│   │   ├── utils/
│   │   │   └── gizmoMath.js     # vec3, mat4, quat pure functions
│   │   └── styles/
│   │       └── theme.css        # CSS custom properties and global resets only
│   └── jsconfig.json            # IDE support (path aliases, etc.)
├── static/                      # Built output (committed to git)
│   ├── index.html               # Vite-generated entry point
│   └── assets/                  # Vite-generated JS/CSS chunks
├── server.py                    # UNCHANGED
├── protocol.py                  # UNCHANGED
└── frame_producer.py            # UNCHANGED
```

### Build Pipeline

- **Tooling**: Vue 3 + Vite (plain JavaScript with JSDoc type annotations — no TypeScript compilation step)
- **Build output**: Vite outputs chunked assets (JS, CSS, `index.html`) to `../static/`. The server already mounts `static/` via `StaticFiles` at `/static`, so chunked output works without any special plugin. **Critical: `base` must be set to `/static/` in `vite.config.js`** so that the generated `index.html` references assets as `/static/assets/index-xxxxx.js` rather than `./assets/...`, since the HTML is served from `/` but assets live under the `/static` mount.
- **Dev workflow**: `cd frontend && npm run dev` with HMR. Vite `server.proxy` configured with `ws: true` to proxy `/ws` to a running Genesis server (developer runs both `npm run dev` and a Genesis script simultaneously).
- **Production**: `npm run build` → output committed to `static/`
- **End user experience**: `pip install genesis` ships pre-built `static/` assets — no Node.js needed
- **Git hygiene**: `node_modules/` added to `.gitignore` (general pattern, not path-specific). Built `static/` assets committed to git. Contributors must run `npm run build` after modifying Vue source before committing.

### Data Flow

```
                    useWebSocket()
                         │
            ┌────────────┼────────────┐
            │            │            │
        useScene()   useCamera()  useGizmo()
            │            │            │
    ┌───────┴───────┐    │            │
    │               │    │            │
LeftPanel      RightPanel │      ViewportCanvas
(EntityTree)   (EntityProperties,    │
               CameraProperties,    FloatingSimBar
               VisToggles)          FloatingStatus
```

### Composables

**`useWebSocket()`** — Owns the single WebSocket connection to `/ws`. Binary messages update a reactive `latestFrame` ref. JSON messages are dispatched to `useScene()` or `useCamera()` by message type. Exposes `send(msg)` for all outbound commands. Handles reconnect with exponential backoff (initial: 500ms, multiplier: 2x, cap: 30s). Exposes `connected` ref.

**`useScene()`** — Reactive scene state: `entities[]`, `selectedEntityIdx`, `paused`, `simTime`, `step`, `fps`, `visState` (shadows, wireframe, frames, normals, etc.). Populated from `SCENE_INFO` on connect. `STATE_UPDATE` messages update `simTime`, `step`, `fps`, `paused`, and `camera_state` each frame — but entity qpos is only sent in the initial `SCENE_INFO`, not per-frame. Local qpos mutations (from sliders or gizmo drag) are tracked client-side. Components read reactively; mutations go through `send()`.

**Qpos state ownership**: During gizmo drag or slider manipulation, local entity qpos is authoritative. A per-entity `isLocallyEditing` flag suppresses incoming server updates for that entity's qpos. The flag is set immediately on `mousedown`/`input` (before debounce fires) to prevent race conditions where `STATE_UPDATE` messages arrive between drag start and the first debounced send. Cleared on `mouseup`/blur. This prevents the server from overwriting mid-edit values.

**Note on existing dead code**: The current `index.html` has a handler for `msg.entities` within `state_update` messages, but the server's `build_state_update()` never sends entity data in `STATE_UPDATE` — only in the initial `SCENE_INFO`. The new design intentionally does not replicate this dead code path.

**Message type constants**: Use string literals directly (e.g., `'sim_control'`, `'entity_update'`) rather than a separate constants file. The server's `protocol.py` is the canonical source, and the six string constants are too few to justify a synchronization obligation between Python and JavaScript.

**`useCamera()`** — Reactive camera state: `pos`, `lookat`, `fov`. Updated from server's `camera_state` in each `STATE_UPDATE`. Exposes `orbit()`, `pan()`, `zoom()`, `setPose()`, `setFov()`, `resetCamera()` — each sends the corresponding `CAMERA_UPDATE` command.

**Camera input focus guard**: When the user is actively editing a camera input field (position, lookat, FOV), incoming `STATE_UPDATE` camera values must NOT overwrite the input. The composable checks `document.activeElement` against camera input refs and skips reactive updates for focused fields. This preserves the current `syncCameraInputs` behavior.

**`useGizmo(canvas)`** — Takes a canvas ref. Manages gizmo state: `entityIdx`, `mode` (translate/rotate), `activeAxis`, `dragging`. Draws on canvas 2D context using server-provided `view_matrix`/`proj_matrix` (with fallback reconstruction from pos/lookat/fov). Handles mouse events for hit testing and drag. Sends `ENTITY_UPDATE` commands with new qpos on drag completion.

**Canvas rendering model**: The canvas uses `requestAnimationFrame` for its render loop, NOT Vue's reactivity system. Vue cannot drive imperative canvas drawing. `ViewportCanvas.vue` owns the rAF loop in its `onMounted`/`onUnmounted` lifecycle hooks. Each frame: draw latest JPEG → if gizmo active, save to background canvas → draw gizmo overlay. The composable attaches mouse event listeners to the canvas ref in `onMounted` and cleans up in `onUnmounted`.

**No Pinia** — State lives in composables provided at App level via `provide/inject`. The scope does not justify a state management library.

### Canvas & Gizmo Integration

```html
<div class="viewport" @mousedown @mousemove @mouseup @wheel>
  <canvas ref="canvas" />
  <FloatingSimBar />       <!-- absolute, bottom-center -->
  <FloatingStatus />       <!-- absolute, top-right -->
</div>
```

**Frame rendering pipeline**:
1. WebSocket binary message → `createImageBitmap()` from blob → `ctx.drawImage()` on canvas
2. If gizmo active: save frame to offscreen canvas (`_bgCanvas`)
3. Draw gizmo axes/arrows/circles on top using Canvas 2D API
4. On hover/drag: restore from `_bgCanvas`, redraw gizmo to avoid overdraw artifacts

**Mouse interaction priority**:
1. Gizmo hit test (if gizmo enabled for selected entity)
2. If gizmo miss → camera controls (left-drag = orbit, right-drag = pan, wheel = zoom)

Note: The current implementation only handles right-drag for pan (button === 2). Middle-drag pan may be added later but is not part of initial parity.

**Matrix pipeline**: Server sends `view_matrix` and `proj_matrix` as flat Float64Arrays with each `STATE_UPDATE`. `useGizmo()` stores these and uses them for 3D→2D projection. All gizmo math extracted to `utils/gizmoMath.js` as pure functions.

**Slider debouncing**: DOF sliders and other frequently-changing inputs are debounced at 50ms before sending WebSocket commands. This prevents flooding the server with hundreds of `entity_update` messages per second during rapid slider movement. Implemented via a shared `debounce()` utility or `watch` with `{ flush: 'post' }` and manual `setTimeout`.

## UI Design

### Layout

Three-panel layout:
- **Left panel** (~240px): Entity tree with search/filter, collapsible
- **Center**: Viewport canvas filling remaining space, with floating controls overlaid
- **Right panel** (~280px): Properties for selected entity, camera, visualization toggles

### Theme

Genesis-branded dark theme, Spline-inspired shapes:

```css
--bg-base:      #12131a    /* deepest background */
--bg-panel:     #1a1c28    /* panel backgrounds */
--bg-surface:   #222433    /* cards, inputs, elevated surfaces */
--bg-hover:     #2a2d40    /* hover states */
--accent:       #66b2c4    /* Genesis teal — primary accent */
--accent-hover: #7fc8d8    /* lighter teal for hover */
--accent-dim:   #3d7a8a    /* muted teal for active segments */
--text:         #e8e8f0    /* primary text */
--text-muted:   #7a7c92    /* secondary/label text */
--border:       #2a2d40    /* subtle borders */
--radius-sm:    6px        /* inputs, small elements */
--radius-md:    12px       /* panels, cards */
--radius-lg:    20px       /* pill buttons, floating bars */
```

No glassmorphism/blur. Clean flat surfaces with subtle 1px borders and rounded corners.

### Components

**Segmented toggles** — Rounded pill container with segments that highlight with `--accent-dim` when active. Contextual labels: "Visual | Collision", "Perspective | Orthographic", "On | Off".

**Entity tree** — Indented list with dot icons (static entities) and link icons (articulated). Selected entity gets `--accent` left border highlight.

**DOF sliders** — Custom-styled range inputs with `--accent` thumb and track fill. Value displayed right-aligned in `--accent` color. Compact vertical stacking.

**Floating sim bar** — Centered bottom of viewport. `--bg-panel` background, `--radius-lg` corners, subtle box-shadow. Icon buttons for Play/Pause/Step/Reset. Segmented Perspective/Ortho toggle.

**Floating status badge** — Top-right of viewport. Small pill with connection dot + FPS + sim time. Semi-transparent (`opacity: 0.85`).

**Section headers** — Uppercase, small font, `--text-muted`, thin divider below. Matches Spline's "Global Settings" / "Post-Processing" section style.

**Scrollbars** — Thin, rounded, `--border` track, `--accent-dim` thumb.

### Responsive

- Panels collapse on narrow screens (<700px)
- Left panel: hamburger toggle, slides over viewport as an overlay
- Right panel: hidden by default on narrow screens, accessible via a toggle button
- Viewport always fills remaining space
- Both panels use `transform: translateX()` transitions for smooth slide in/out

## Feature Parity Checklist

Every feature in the current `index.html` is preserved:

### Simulation Controls
- [x] Play / Pause toggle
- [x] Step (single frame advance)
- [x] Reset simulation

### Visualization Toggles
- [x] Shadows (On/Off)
- [x] World Frame (On/Off)
- [x] Link Frame (On/Off)
- [x] Link Frame Size (slider)
- [x] Orthographic / Perspective
- [x] Wireframe (On/Off)
- [x] Face Normals (On/Off)
- [x] Vertex Normals (On/Off)
- [x] Camera Frustum (On/Off)

### Camera Controls
- [x] Position X/Y/Z number inputs
- [x] Look At X/Y/Z number inputs
- [x] FOV slider
- [x] Reset Camera button
- [x] Mouse orbit (left drag)
- [x] Mouse pan (middle/right drag)
- [x] Mouse zoom (scroll wheel)

### Entity Browser
- [x] Entity list with names and DOF counts
- [x] Click-to-select (new: drives right panel)
- [x] Per-entity vis mode (Visual/Collision)
- [x] Per-entity wireframe toggle
- [x] Per-entity contact visualization toggle
- [x] DOF sliders with joint names, limits, live values
- [x] Free joint position (XYZ inputs)
- [x] Free joint orientation (Quaternion/Euler mode toggle)
- [x] Euler ↔ Quaternion conversion
- [x] Gizmo toggle per entity
- [x] Gizmo mode (Translate/Rotate)

### Status Display
- [x] Connection indicator (dot + text)
- [x] FPS counter (server-reported)
- [x] Simulation time
- [x] Step count
- [x] Paused indicator

### Other
- [x] Multi-environment note display
- [x] Responsive sidebar toggle for mobile
- [x] WebSocket auto-reconnect

## Backend Changes

**None.** The Python backend (`server.py`, `protocol.py`, `frame_producer.py`) remains completely unchanged. The WebSocket protocol (binary JPEG frames + JSON text messages) is preserved exactly. The server continues to serve `static/index.html` via `FileResponse`.

## Component Decomposition Strategy

Start with the minimum viable component set, extract sub-components only when justified:

**Phase 1 (initial build)**: `App.vue`, `LeftPanel.vue`, `ViewportCanvas.vue`, `RightPanel.vue`, plus the four composables and `gizmoMath.js`. Floating sim bar and floating status start as template sections within `ViewportCanvas.vue`. However, since `ViewportCanvas.vue` also owns the rAF loop, gizmo mouse events, and binary frame rendering, it may exceed the 200-line threshold quickly — in which case `FloatingSimBar.vue` and `FloatingStatus.vue` should be extracted immediately as they are purely declarative template content with no canvas interaction.

**Extract when**: A component exceeds ~200 lines, or the same UI pattern appears 3+ times.

**Likely early extractions**:
- `SegmentedToggle.vue` — appears ~6 times (vis toggles, entity vis mode, gizmo mode, ortho/persp, euler/quat mode, wireframe)
- `SliderControl.vue` — appears for every DOF slider, FOV, and link frame size
- `EntityTree.vue` from `LeftPanel.vue` — once tree logic (expand/collapse, selection highlighting) grows
- `EntityProperties.vue` from `RightPanel.vue` — once DOF sliders + free joint controls grow
- `CameraProperties.vue` from `RightPanel.vue` — camera inputs with focus guard logic
- `VisToggles.vue` from `RightPanel.vue` — visualization toggle section

## What's New (Visual/UX Only)

- Three-panel layout with entity selection model
- Spline-inspired visual styling with Genesis branding
- Segmented pill toggle buttons (replacing traditional switches)
- Floating viewport controls (sim bar at bottom, status badge at top-right)
- Component-based architecture for maintainability
