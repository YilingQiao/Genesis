<script setup>
import { ref, inject, onMounted, onUnmounted } from 'vue'
import { SET_RESOLUTION } from '../utils/protocol.js'

const ws = inject('ws')
const camera = inject('camera')
const gizmo = inject('gizmo')

const canvasRef = ref(null)
let ctx = null
let rafId = null
let resizeObserver = null
let resizeTimer = null

// Camera mouse state
let isDragging = false
let isRightDrag = false
let lastMouseX = 0
let lastMouseY = 0

function getCanvasCoords(e) {
  const canvas = canvasRef.value
  const r = canvas.getBoundingClientRect()
  return [
    (e.clientX - r.left) * (canvas.width / r.width),
    (e.clientY - r.top) * (canvas.height / r.height),
  ]
}

function sendResolution(w, h) {
  const width = Math.round(w * devicePixelRatio)
  const height = Math.round(h * devicePixelRatio)
  if (width > 0 && height > 0) {
    ws.send({ type: SET_RESOLUTION, width, height })
  }
}

function onResize(entries) {
  clearTimeout(resizeTimer)
  resizeTimer = setTimeout(() => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) {
        sendResolution(width, height)
      }
    }
  }, 200)
}

function renderLoop() {
  const canvas = canvasRef.value
  if (!canvas || !ctx) {
    rafId = requestAnimationFrame(renderLoop)
    return
  }

  const frame = ws.latestFrame.value
  if (frame) {
    if (canvas.width !== frame.width || canvas.height !== frame.height) {
      canvas.width = frame.width
      canvas.height = frame.height
    }
    ctx.drawImage(frame, 0, 0)
    frame.close()
    ws.latestFrame.value = null

    // Save clean frame for gizmo overlay redraws
    gizmo.saveBgFrame(canvas)
  } else if (gizmo.entityIdx.value !== null) {
    // Restore clean frame before redrawing gizmo
    gizmo.restoreBgFrame(ctx, canvas)
  }

  // Draw gizmo overlay
  gizmo.draw(ctx, canvas)

  rafId = requestAnimationFrame(renderLoop)
}

function onMouseDown(e) {
  e.preventDefault()
  const canvas = canvasRef.value

  // Check gizmo interaction first (left click only)
  if (e.button === 0 && gizmo.entityIdx.value !== null) {
    const coords = getCanvasCoords(e)
    const hitAxis = gizmo.hitTest(coords[0], coords[1], canvas.width, canvas.height)
    if (hitAxis) {
      gizmo.startDrag(hitAxis, coords)
      canvas.style.cursor = 'crosshair'
      return
    }
  }

  isDragging = true
  isRightDrag = e.button === 2
  lastMouseX = e.clientX
  lastMouseY = e.clientY
}

function onMouseUp() {
  if (gizmo.dragging.value) {
    gizmo.endDrag()
    if (canvasRef.value) canvasRef.value.style.cursor = 'grab'
    return
  }
  isDragging = false
  isRightDrag = false
}

function onMouseMove(e) {
  const canvas = canvasRef.value

  // Gizmo drag takes priority
  if (gizmo.dragging.value) {
    const coords = getCanvasCoords(e)
    gizmo.moveDrag(coords, canvas.width, canvas.height)
    return
  }

  // Gizmo hover highlighting
  if (gizmo.entityIdx.value !== null && !isDragging) {
    const coords = getCanvasCoords(e)
    const hit = gizmo.hitTest(coords[0], coords[1], canvas.width, canvas.height)
    gizmo.activeAxis.value = hit
    canvas.style.cursor = hit ? 'pointer' : 'grab'
  }

  if (!isDragging) return

  const dx = e.clientX - lastMouseX
  const dy = e.clientY - lastMouseY
  lastMouseX = e.clientX
  lastMouseY = e.clientY

  if (dx === 0 && dy === 0) return

  const rect = canvas.getBoundingClientRect()
  const sensitivity = 2.0

  if (isRightDrag) {
    camera.pan((dx / rect.width) * sensitivity, (dy / rect.height) * sensitivity)
  } else {
    camera.orbit((dx / rect.width) * sensitivity, (dy / rect.height) * sensitivity)
  }
}

function onWheel(e) {
  e.preventDefault()
  // Block zoom while gizmo is being dragged
  if (gizmo.dragging.value) return
  const factor = e.deltaY > 0 ? 1.1 : 0.9
  camera.zoom(factor)
}

function onContextMenu(e) {
  e.preventDefault()
}

onMounted(() => {
  const canvas = canvasRef.value
  ctx = canvas.getContext('2d')
  canvas.width = 1280
  canvas.height = 720
  rafId = requestAnimationFrame(renderLoop)

  // Watch for viewport resize and tell server
  resizeObserver = new ResizeObserver(onResize)
  resizeObserver.observe(canvas.parentElement)

  // Send initial resolution
  const rect = canvas.parentElement.getBoundingClientRect()
  sendResolution(rect.width, rect.height)

  window.addEventListener('mouseup', onMouseUp)
  window.addEventListener('mousemove', onMouseMove)
})

onUnmounted(() => {
  if (rafId) cancelAnimationFrame(rafId)
  if (resizeObserver) resizeObserver.disconnect()
  clearTimeout(resizeTimer)
  window.removeEventListener('mouseup', onMouseUp)
  window.removeEventListener('mousemove', onMouseMove)
})
</script>

<template>
  <canvas
    ref="canvasRef"
    class="viewport-canvas"
    @mousedown="onMouseDown"
    @wheel="onWheel"
    @contextmenu="onContextMenu"
  />
</template>

<style scoped>
.viewport-canvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
  object-fit: contain;
}

.viewport-canvas:active {
  cursor: grabbing;
}
</style>
