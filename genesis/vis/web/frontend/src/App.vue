<script setup>
import { provide, ref, onMounted } from 'vue'
import { useWebSocket } from './composables/useWebSocket.js'
import { useScene } from './composables/useScene.js'
import { useCamera } from './composables/useCamera.js'
import { useGizmo } from './composables/useGizmo.js'
import ViewportCanvas from './components/ViewportCanvas.vue'
import FloatingSimBar from './components/FloatingSimBar.vue'
import LeftPanel from './components/LeftPanel.vue'
import RightPanel from './components/RightPanel.vue'

const ws = useWebSocket()
const scene = useScene(ws.send)
const camera = useCamera(ws.send)
const gizmo = useGizmo(
  ws.send,
  scene.findEntity,
  camera.pos,
  camera.lookat,
  camera.fov,
  scene.autoPause,
  scene.updateEntityQpos,
)

// Responsive state
const leftOpen = ref(true)
const rightOpen = ref(true)
const isNarrow = ref(false)

function checkNarrow() {
  const narrow = window.innerWidth < 700
  if (narrow !== isNarrow.value) {
    isNarrow.value = narrow
    if (narrow) {
      leftOpen.value = false
      rightOpen.value = false
    } else {
      leftOpen.value = true
      rightOpen.value = true
    }
  }
}

ws.setHandlers({
  sceneInfo: (msg) => {
    scene.handleSceneInfo(msg)
    if (msg.camera_state) camera.setCameraState(msg.camera_state)
    if (msg.camera_state?.view_matrix) {
      gizmo.setServerMatrices(msg.camera_state.view_matrix, msg.camera_state.proj_matrix)
    }
  },
  stateUpdate: (msg) => {
    scene.handleStateUpdate(msg)
    if (msg.camera_state) {
      camera.handleCameraState(msg.camera_state)
      if (msg.camera_state.view_matrix) {
        gizmo.setServerMatrices(msg.camera_state.view_matrix, msg.camera_state.proj_matrix)
      }
    }
  },
  reconnect: () => {
    scene.resetState()
    camera.resetState()
    gizmo.resetState()
    ws.resetFrame()
  },
})

provide('ws', ws)
provide('scene', scene)
provide('camera', camera)
provide('gizmo', gizmo)

onMounted(() => {
  ws.connect()
  checkNarrow()
  window.addEventListener('resize', checkNarrow)
})
</script>

<template>
  <div class="app-root">
    <!-- Full-screen viewport behind everything -->
    <div class="viewport-full">
      <ViewportCanvas />
    </div>

    <!-- Floating panels on top of viewport -->
    <aside
      class="panel panel--left"
      :class="{ 'panel--hidden': !leftOpen }"
    >
      <LeftPanel />
    </aside>

    <aside
      class="panel panel--right"
      :class="{ 'panel--hidden': !rightOpen }"
    >
      <RightPanel />
    </aside>

    <!-- Floating control bar (bottom center) -->
    <FloatingSimBar />

    <!-- Mobile toggles -->
    <button
      v-if="isNarrow"
      class="mobile-toggle mobile-toggle--left"
      @click="leftOpen = !leftOpen"
    >&#9776;</button>
    <button
      v-if="isNarrow"
      class="mobile-toggle mobile-toggle--right"
      @click="rightOpen = !rightOpen"
    >&#9881;</button>
  </div>
</template>

<style scoped>
.app-root {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

/* Viewport fills entire screen */
.viewport-full {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-base);
}

/* Floating glass panels */
.panel {
  position: absolute;
  top: 16px;
  bottom: 16px;
  z-index: 10;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.panel--left {
  left: 16px;
  width: 250px;
}

.panel--right {
  right: 16px;
  width: 280px;
}

.panel--hidden {
  opacity: 0;
  pointer-events: none;
}

.panel--hidden.panel--left {
  transform: translateX(-20px);
}

.panel--hidden.panel--right {
  transform: translateX(20px);
}

.mobile-toggle {
  position: absolute;
  z-index: 25;
  top: 24px;
  width: 34px;
  height: 34px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}

.mobile-toggle:hover {
  background: rgba(255, 255, 255, 0.18);
}

.mobile-toggle--left {
  left: 24px;
}

.mobile-toggle--right {
  right: 24px;
}
</style>
