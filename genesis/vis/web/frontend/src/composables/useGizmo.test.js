import { describe, it, expect } from 'vitest'
import { ref } from 'vue'
import { useGizmo } from './useGizmo.js'
import { buildViewMatrix, buildProjMatrix } from '../utils/gizmoMath.js'

function createGizmo(opts = {}) {
  const sent = []
  const send = (msg) => sent.push(msg)
  const qposUpdates = []
  const updateQpos = (entityIdx, qpos, quatGroups) => {
    qposUpdates.push({ entityIdx, qpos: [...qpos], quatGroups })
  }
  const autoPauseCalls = []
  const autoPause = () => autoPauseCalls.push(true)

  const entity = opts.entity || {
    idx: 0,
    has_free_joint: true,
    free_joint_q_start: 0,
    quat_groups: [[3, 7]],
    qpos: [0, 0, 0.5, 1, 0, 0, 0],
  }

  const findEntity = (idx) => idx === entity.idx ? entity : null
  const cameraPos = ref([0, -5, 0])
  const cameraLookat = ref([0, 0, 0])
  const cameraFov = ref(60)

  const gizmo = useGizmo(send, findEntity, cameraPos, cameraLookat, cameraFov, autoPause, updateQpos)
  return { gizmo, sent, qposUpdates, autoPauseCalls, entity, cameraPos, cameraLookat, cameraFov }
}

// Build valid projection matrices for testing
function setupMatrices(gizmo) {
  const vm = buildViewMatrix([0, -5, 0], [0, 0, 0], [0, 0, 1])
  const pm = buildProjMatrix((60 * Math.PI) / 180, 800 / 600, 0.05, 100)
  gizmo.setServerMatrices(Array.from(vm), Array.from(pm))
}

describe('useGizmo: toggle and mode', () => {
  it('toggle activates and deactivates correctly', () => {
    const { gizmo } = createGizmo()
    expect(gizmo.entityIdx.value).toBeNull()
    gizmo.toggle(0)
    expect(gizmo.entityIdx.value).toBe(0)
    gizmo.toggle(0)
    expect(gizmo.entityIdx.value).toBeNull()
    gizmo.toggle(1)
    expect(gizmo.entityIdx.value).toBe(1)
  })

  it('setMode changes gizmo mode', () => {
    const { gizmo } = createGizmo()
    expect(gizmo.mode.value).toBe('translate')
    gizmo.setMode('rotate')
    expect(gizmo.mode.value).toBe('rotate')
  })
})

describe('useGizmo: reconnect cleanup', () => {
  it('resetState clears all state including mode', () => {
    const { gizmo } = createGizmo()
    gizmo.entityIdx.value = 0
    gizmo.mode.value = 'rotate'
    gizmo.activeAxis.value = 'z'
    gizmo.dragging.value = true

    gizmo.resetState()

    expect(gizmo.entityIdx.value).toBeNull()
    expect(gizmo.mode.value).toBe('translate')
    expect(gizmo.activeAxis.value).toBeNull()
    expect(gizmo.dragging.value).toBe(false)
  })
})

describe('useGizmo: qpos routing through updateQpos', () => {
  it('moveTranslate calls updateQpos and does NOT use raw send', () => {
    const { gizmo, sent, qposUpdates, autoPauseCalls } = createGizmo()
    gizmo.entityIdx.value = 0
    gizmo.mode.value = 'translate'
    setupMatrices(gizmo)

    gizmo.startDrag('x', [400, 300])
    gizmo.moveDrag([420, 300], 800, 600)

    // Raw send must NOT have entity_update messages
    const rawEntityUpdates = sent.filter(m => m.type === 'entity_update')
    expect(rawEntityUpdates.length).toBe(0)

    // updateQpos MUST have been called (positive assertion)
    expect(qposUpdates.length).toBeGreaterThan(0)
    expect(qposUpdates[0].entityIdx).toBe(0)
    expect(qposUpdates[0].qpos.length).toBe(7)

    // autoPause MUST have been called
    expect(autoPauseCalls.length).toBeGreaterThan(0)
  })

  it('moveRotate calls updateQpos and does NOT use raw send', () => {
    const { gizmo, sent, qposUpdates, autoPauseCalls } = createGizmo()
    gizmo.entityIdx.value = 0
    gizmo.mode.value = 'rotate'
    setupMatrices(gizmo)

    gizmo.startDrag('z', [400, 300])
    gizmo.moveDrag([420, 320], 800, 600)

    const rawEntityUpdates = sent.filter(m => m.type === 'entity_update')
    expect(rawEntityUpdates.length).toBe(0)

    expect(qposUpdates.length).toBeGreaterThan(0)
    expect(qposUpdates[0].entityIdx).toBe(0)
    // Quaternion should have changed from identity
    const q = qposUpdates[0].qpos.slice(3, 7)
    const isIdentity = q[0] === 1 && q[1] === 0 && q[2] === 0 && q[3] === 0
    expect(isIdentity).toBe(false)

    expect(autoPauseCalls.length).toBeGreaterThan(0)
  })
})
