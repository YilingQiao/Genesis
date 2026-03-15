import { ref } from 'vue'
import {
  vec3Sub, vec3Add, vec3Scale, vec3Dot, vec3Len, vec3Norm, vec3Cross,
  buildViewMatrix, buildProjMatrix, project3D, ptSegDist,
  axisAngleToQuat, quatMul, quatNormalize,
  GIZMO_SIZE_FACTOR, GIZMO_HIT_RADIUS, GIZMO_AXES,
} from '../utils/gizmoMath.js'

/**
 * Gizmo composable for 3D manipulation handles.
 * Manages gizmo state, drawing, hit testing, and drag interaction.
 */
export function useGizmo(send, findEntity, cameraPos, cameraLookat, cameraFov, autoPause, updateQpos) {
  const entityIdx = ref(null)
  const mode = ref('translate')
  const activeAxis = ref(null)
  const dragging = ref(false)

  let lastMouseCanvas = null
  let bgCanvas = null
  let bgCtx = null

  // Server-provided matrices (exact match with renderer)
  let serverViewMat = null
  let serverProjMat = null

  function setServerMatrices(viewMatrix, projMatrix) {
    if (viewMatrix) serverViewMat = new Float64Array(viewMatrix)
    if (projMatrix) serverProjMat = new Float64Array(projMatrix)
  }

  function getMatrices(canvasWidth, canvasHeight) {
    if (serverViewMat && serverProjMat) {
      return { vm: serverViewMat, pm: serverProjMat }
    }
    // Fallback: reconstruct from pos/lookat/fov
    const cp = cameraPos.value
    const cl = cameraLookat.value
    const f = cameraFov.value
    const aspect = canvasWidth / Math.max(canvasHeight, 1)
    return {
      vm: buildViewMatrix(cp, cl, [0, 0, 1]),
      pm: buildProjMatrix((f * Math.PI) / 180, aspect, 0.05, 100),
    }
  }

  function getEntityPos() {
    if (entityIdx.value === null) return null
    const e = findEntity(entityIdx.value)
    if (!e || !e.has_free_joint) return null
    const q = e.qpos || []
    const s = e.free_joint_q_start || 0
    return [q[s] || 0, q[s + 1] || 0, q[s + 2] || 0]
  }

  function ensureBgCanvas(canvas) {
    if (!bgCanvas) {
      bgCanvas = document.createElement('canvas')
      bgCtx = bgCanvas.getContext('2d')
    }
    if (bgCanvas.width !== canvas.width || bgCanvas.height !== canvas.height) {
      bgCanvas.width = canvas.width
      bgCanvas.height = canvas.height
    }
  }

  function saveBgFrame(canvas) {
    if (entityIdx.value !== null) {
      ensureBgCanvas(canvas)
      bgCtx.drawImage(canvas, 0, 0)
    }
  }

  function restoreBgFrame(ctx, canvas) {
    if (bgCanvas && bgCanvas.width > 0) {
      ctx.drawImage(bgCanvas, 0, 0)
    }
  }

  function draw(ctx, canvas) {
    if (entityIdx.value === null) return
    const ePos = getEntityPos()
    if (!ePos) return

    const w = canvas.width
    const h = canvas.height
    const { vm, pm } = getMatrices(w, h)
    const center = project3D(ePos, vm, pm, w, h)
    if (!center) return

    const cp = cameraPos.value
    const dist = vec3Len(vec3Sub(cp, ePos))
    const axLen = dist * GIZMO_SIZE_FACTOR

    if (mode.value === 'translate') {
      drawTranslate(ctx, ePos, center, axLen, w, h, vm, pm)
    } else {
      drawRotate(ctx, ePos, center, axLen, w, h, vm, pm)
    }
  }

  function drawTranslate(ctx, ePos, center, axLen, w, h, vm, pm) {
    GIZMO_AXES.forEach((ax) => {
      const endW = vec3Add(ePos, vec3Scale(ax.dir, axLen))
      const end = project3D(endW, vm, pm, w, h)
      if (!end) return

      const active = activeAxis.value === ax.key
      const col = active ? '#ffffff' : ax.color
      const lw = active ? 3.5 : 2.5

      // Axis line
      ctx.strokeStyle = col
      ctx.lineWidth = lw
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(center[0], center[1])
      ctx.lineTo(end[0], end[1])
      ctx.stroke()

      // Arrowhead
      const dx = end[0] - center[0]
      const dy = end[1] - center[1]
      const a = Math.atan2(dy, dx)
      const hl = active ? 14 : 10
      ctx.fillStyle = col
      ctx.beginPath()
      ctx.moveTo(end[0], end[1])
      ctx.lineTo(end[0] - hl * Math.cos(a - 0.4), end[1] - hl * Math.sin(a - 0.4))
      ctx.lineTo(end[0] - hl * Math.cos(a + 0.4), end[1] - hl * Math.sin(a + 0.4))
      ctx.closePath()
      ctx.fill()

      // Label
      ctx.fillStyle = col
      ctx.font = 'bold 11px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(ax.key.toUpperCase(), end[0] + 14 * Math.cos(a), end[1] + 14 * Math.sin(a))
    })

    // Center dot
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(center[0], center[1], 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    ctx.stroke()
  }

  function drawRotate(ctx, ePos, center, radius, w, h, vm, pm) {
    GIZMO_AXES.forEach((ax) => {
      const n = ax.dir
      let u
      if (Math.abs(n[2]) < 0.9) {
        u = vec3Norm(vec3Cross(n, [0, 0, 1]))
      } else {
        u = vec3Norm(vec3Cross(n, [0, 1, 0]))
      }
      const v = vec3Cross(n, u)

      const active = activeAxis.value === ax.key
      const col = active ? '#ffffff' : ax.color
      ctx.strokeStyle = col
      ctx.lineWidth = active ? 3 : 2
      ctx.beginPath()

      const segs = 48
      let first = true
      for (let i = 0; i <= segs; i++) {
        const angle = (i / segs) * Math.PI * 2
        const pt = vec3Add(
          ePos,
          vec3Add(vec3Scale(u, Math.cos(angle) * radius), vec3Scale(v, Math.sin(angle) * radius)),
        )
        const sp = project3D(pt, vm, pm, w, h)
        if (!sp) { first = true; continue }
        if (first) { ctx.moveTo(sp[0], sp[1]); first = false }
        else ctx.lineTo(sp[0], sp[1])
      }
      ctx.stroke()

      // Label
      const labelPt = vec3Add(ePos, vec3Scale(u, radius * 1.15))
      const lsp = project3D(labelPt, vm, pm, w, h)
      if (lsp) {
        ctx.fillStyle = col
        ctx.font = 'bold 11px sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(ax.key.toUpperCase(), lsp[0], lsp[1])
      }
    })

    // Center dot
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(center[0], center[1], 4, 0, Math.PI * 2)
    ctx.fill()
  }

  function hitTest(cx, cy, canvasWidth, canvasHeight) {
    if (entityIdx.value === null) return null
    const ePos = getEntityPos()
    if (!ePos) return null

    const { vm, pm } = getMatrices(canvasWidth, canvasHeight)
    const center = project3D(ePos, vm, pm, canvasWidth, canvasHeight)
    if (!center) return null

    const cp = cameraPos.value
    const dist = vec3Len(vec3Sub(cp, ePos))
    const axLen = dist * GIZMO_SIZE_FACTOR

    let best = null
    let bestD = GIZMO_HIT_RADIUS

    if (mode.value === 'translate') {
      GIZMO_AXES.forEach((ax) => {
        const endW = vec3Add(ePos, vec3Scale(ax.dir, axLen))
        const end = project3D(endW, vm, pm, canvasWidth, canvasHeight)
        if (!end) return
        const d = ptSegDist(cx, cy, center[0], center[1], end[0], end[1])
        if (d < bestD) { bestD = d; best = ax.key }
      })
    } else {
      GIZMO_AXES.forEach((ax) => {
        const n = ax.dir
        let u
        if (Math.abs(n[2]) < 0.9) u = vec3Norm(vec3Cross(n, [0, 0, 1]))
        else u = vec3Norm(vec3Cross(n, [0, 1, 0]))
        const v = vec3Cross(n, u)
        const segs = 48
        for (let i = 0; i < segs; i++) {
          const a1 = (i / segs) * Math.PI * 2
          const a2 = ((i + 1) / segs) * Math.PI * 2
          const p1 = vec3Add(ePos, vec3Add(vec3Scale(u, Math.cos(a1) * axLen), vec3Scale(v, Math.sin(a1) * axLen)))
          const p2 = vec3Add(ePos, vec3Add(vec3Scale(u, Math.cos(a2) * axLen), vec3Scale(v, Math.sin(a2) * axLen)))
          const s1 = project3D(p1, vm, pm, canvasWidth, canvasHeight)
          const s2 = project3D(p2, vm, pm, canvasWidth, canvasHeight)
          if (!s1 || !s2) continue
          const d = ptSegDist(cx, cy, s1[0], s1[1], s2[0], s2[1])
          if (d < bestD) { bestD = d; best = ax.key }
        }
      })
    }
    return best
  }

  function startDrag(axis, coords) {
    dragging.value = true
    activeAxis.value = axis
    lastMouseCanvas = coords
  }

  function moveDrag(coords, canvasWidth, canvasHeight) {
    if (!dragging.value || !activeAxis.value) return
    const entity = findEntity(entityIdx.value)
    if (!entity) return

    if (mode.value === 'translate') {
      moveTranslate(entity, coords, canvasWidth, canvasHeight)
    } else {
      moveRotate(entity, coords, canvasWidth, canvasHeight)
    }
    lastMouseCanvas = coords
  }

  function moveTranslate(entity, coords, w, h) {
    const ePos = getEntityPos()
    if (!ePos) return
    const { vm, pm } = getMatrices(w, h)

    const axDirMap = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }
    const axDir = axDirMap[activeAxis.value]

    const sc = project3D(ePos, vm, pm, w, h)
    const se = project3D(vec3Add(ePos, axDir), vm, pm, w, h)
    if (!sc || !se) return

    const sd = [se[0] - sc[0], se[1] - sc[1]]
    const sdl2 = sd[0] * sd[0] + sd[1] * sd[1]
    if (sdl2 < 1e-4) return

    const dx = coords[0] - lastMouseCanvas[0]
    const dy = coords[1] - lastMouseCanvas[1]
    const worldDelta = (dx * sd[0] + dy * sd[1]) / sdl2

    const qpos = (entity.qpos || []).slice()
    const qs = entity.free_joint_q_start || 0
    const axIdx = { x: 0, y: 1, z: 2 }
    qpos[qs + axIdx[activeAxis.value]] += worldDelta

    entity.qpos = qpos
    autoPause()
    updateQpos(entityIdx.value, qpos, entity.quat_groups || [])
  }

  function moveRotate(entity, coords, w, h) {
    const ePos = getEntityPos()
    if (!ePos) return
    const { vm, pm } = getMatrices(w, h)
    const center = project3D(ePos, vm, pm, w, h)
    if (!center) return

    const prevA = Math.atan2(lastMouseCanvas[1] - center[1], lastMouseCanvas[0] - center[0])
    const currA = Math.atan2(coords[1] - center[1], coords[0] - center[0])
    let delta = currA - prevA
    if (delta > Math.PI) delta -= 2 * Math.PI
    if (delta < -Math.PI) delta += 2 * Math.PI

    const axDirMap = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }
    const axDir = axDirMap[activeAxis.value]

    const cp = cameraPos.value
    const cl = cameraLookat.value
    const viewDir = vec3Norm(vec3Sub(cl, cp))
    const sign = vec3Dot(viewDir, axDir) > 0 ? 1 : -1

    const rotQ = axisAngleToQuat(axDir, delta * sign)
    const qpos = (entity.qpos || []).slice()
    const qs = entity.free_joint_q_start || 0
    const curQ = {
      w: qpos[qs + 3] || 1,
      x: qpos[qs + 4] || 0,
      y: qpos[qs + 5] || 0,
      z: qpos[qs + 6] || 0,
    }
    const newQ = quatNormalize(quatMul(rotQ, curQ))
    qpos[qs + 3] = newQ.w
    qpos[qs + 4] = newQ.x
    qpos[qs + 5] = newQ.y
    qpos[qs + 6] = newQ.z

    entity.qpos = qpos
    autoPause()
    updateQpos(entityIdx.value, qpos, entity.quat_groups || [])
  }

  function endDrag() {
    dragging.value = false
    activeAxis.value = null
    lastMouseCanvas = null
  }

  function toggle(idx) {
    entityIdx.value = entityIdx.value === idx ? null : idx
  }

  function setMode(m) {
    mode.value = m
  }

  function resetState() {
    entityIdx.value = null
    mode.value = 'translate'
    activeAxis.value = null
    dragging.value = false
    lastMouseCanvas = null
    serverViewMat = null
    serverProjMat = null
  }

  return {
    entityIdx,
    mode,
    activeAxis,
    dragging,
    setServerMatrices,
    getEntityPos,
    ensureBgCanvas,
    saveBgFrame,
    restoreBgFrame,
    draw,
    hitTest,
    startDrag,
    moveDrag,
    endDrag,
    toggle,
    setMode,
    resetState,
  }
}
