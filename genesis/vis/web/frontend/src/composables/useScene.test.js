import { describe, it, expect, vi } from 'vitest'
import { useScene } from './useScene.js'

function createScene() {
  const sent = []
  const send = (msg) => sent.push(msg)
  const scene = useScene(send)
  return { scene, sent }
}

function makeSceneInfoMsg(opts = {}) {
  return {
    type: 'scene_info',
    entities: opts.entities || [{
      idx: 0,
      name: 'robot',
      n_dofs: 7,
      n_qs: 7,
      has_free_joint: true,
      free_joint_q_start: 0,
      quat_groups: [[3, 7]],
      qpos: [0, 0, 0.5, 1, 0, 0, 0], // xyz + wxyz quat
      q_names: ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'],
      q_limits_lower: [-10, -10, -10, -1, -1, -1, -1],
      q_limits_upper: [10, 10, 10, 1, 1, 1, 1],
      q_is_quaternion: [false, false, false, true, true, true, true],
      vis_mode: 'visual',
      visualize_contact: false,
    }],
    vis_state: {
      shadows: true,
      world_frame: false,
      link_frame: false,
      link_frame_size: 0.1,
      orthographic: false,
    },
    n_envs: 1,
  }
}

describe('useScene: Euler/Quaternion synchronization', () => {
  it('initializes euler from qpos on scene info', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    const state = scene.entityEditState[0]
    expect(state).toBeDefined()
    // Identity quaternion → euler angles should be ~0
    expect(Math.abs(state.euler.roll)).toBeLessThan(1)
    expect(Math.abs(state.euler.pitch)).toBeLessThan(1)
    expect(Math.abs(state.euler.yaw)).toBeLessThan(1)
  })

  it('resyncs euler after updateEntityQpos', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    // Simulate a quaternion edit: 90-degree rotation around Z
    const entity = scene.findEntity(0)
    const cos45 = Math.cos(Math.PI / 4)
    const sin45 = Math.sin(Math.PI / 4)
    const newQpos = [0, 0, 0.5, cos45, 0, 0, sin45]
    entity.qpos = newQpos

    scene.updateEntityQpos(0, newQpos, [[3, 7]])

    const state = scene.entityEditState[0]
    // After 90-deg Z rotation, yaw should be ~90
    expect(state.euler.yaw).toBeCloseTo(90, 0)
  })

  it('resyncs euler when gizmo routes through updateEntityQpos', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    // Simulate what gizmo does after our fix: it calls updateEntityQpos
    const entity = scene.findEntity(0)
    const cos22 = Math.cos(Math.PI / 8)
    const sin22 = Math.sin(Math.PI / 8)
    const rotatedQpos = [1, 0, 0.5, cos22, 0, 0, sin22]
    entity.qpos = rotatedQpos

    scene.updateEntityQpos(0, rotatedQpos, [[3, 7]])

    const state = scene.entityEditState[0]
    // After 45-deg Z rotation, yaw should be ~45
    expect(state.euler.yaw).toBeCloseTo(45, 0)
  })
})

describe('useScene: finite-number guards', () => {
  it('updateEntityDof suppresses NaN values', () => {
    const { scene, sent } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityDof(0, 0, NaN)
    // Should not have sent anything
    const entityUpdates = sent.filter((m) => m.type === 'entity_update')
    expect(entityUpdates.length).toBe(0)
  })

  it('updateEntityDof sends finite values', () => {
    const { scene, sent } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityDof(0, 0, 1.5)
    const entityUpdates = sent.filter((m) => m.type === 'entity_update')
    expect(entityUpdates.length).toBe(1)
    expect(entityUpdates[0].value).toBe(1.5)
  })

  it('updateEntityQpos suppresses array with NaN', () => {
    const { scene, sent } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityQpos(0, [NaN, 0, 0, 1, 0, 0, 0], [[3, 7]])
    const entityUpdates = sent.filter((m) => m.type === 'entity_update')
    expect(entityUpdates.length).toBe(0)
  })

  it('updateEntityQpos sends valid arrays', () => {
    const { scene, sent } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityQpos(0, [0, 0, 0.5, 1, 0, 0, 0], [[3, 7]])
    const entityUpdates = sent.filter((m) => m.type === 'entity_update')
    expect(entityUpdates.length).toBe(1)
  })
})

describe('useScene: vis capability gating', () => {
  it('tracks capabilities from vis_state keys', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    expect(scene.visCapabilities.value.has('shadows')).toBe(true)
    expect(scene.visCapabilities.value.has('orthographic')).toBe(true)
    // Keys not in vis_state should not be present
    expect(scene.visCapabilities.value.has('wireframe')).toBe(false)
    expect(scene.visCapabilities.value.has('face_normals')).toBe(false)
  })

  it('clears capabilities on empty vis_state', () => {
    const { scene } = createScene()
    const msg = makeSceneInfoMsg()
    delete msg.vis_state
    scene.handleSceneInfo(msg)

    expect(scene.visCapabilities.value.size).toBe(0)
  })

  it('clears capabilities on resetState', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())
    expect(scene.visCapabilities.value.size).toBeGreaterThan(0)

    scene.resetState()
    expect(scene.visCapabilities.value.size).toBe(0)
  })
})

describe('useScene: entity state tracking', () => {
  it('updateEntityVisMode updates local vis_mode', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityVisMode(0, 'collision')
    const entity = scene.findEntity(0)
    expect(entity.vis_mode).toBe('collision')
  })

  it('updateEntityWireframe updates local _wireframe', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityWireframe(0, true)
    const entity = scene.findEntity(0)
    expect(entity._wireframe).toBe(true)
  })

  it('updateEntityContactViz updates local visualize_contact', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())

    scene.updateEntityContactViz(0, true)
    const entity = scene.findEntity(0)
    expect(entity.visualize_contact).toBe(true)
  })
})

describe('useScene: reconnect cleanup', () => {
  it('resetState clears paused, simTime, step, fps, nEnvs', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())
    scene.handleStateUpdate({ time: 5.0, step: 100, fps: 60, paused: true })

    expect(scene.paused.value).toBe(true)
    expect(scene.simTime.value).toBe(5.0)
    expect(scene.step.value).toBe(100)
    expect(scene.fps.value).toBe(60)

    scene.resetState()

    expect(scene.paused.value).toBe(false)
    expect(scene.simTime.value).toBe(0)
    expect(scene.step.value).toBe(0)
    expect(scene.fps.value).toBe(0)
    expect(scene.nEnvs.value).toBe(1)
  })

  it('resetState resets visState to defaults', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())
    scene.toggleVis('shadows', false)
    scene.toggleVis('wireframe', true)

    expect(scene.visState.shadows).toBe(false)
    expect(scene.visState.wireframe).toBe(true)

    scene.resetState()

    expect(scene.visState.shadows).toBe(true)
    expect(scene.visState.wireframe).toBe(false)
    expect(scene.visState.orthographic).toBe(false)
    expect(scene.visState.link_frame_size).toBe(0.1)
  })

  it('resetState clears entities and edit state', () => {
    const { scene } = createScene()
    scene.handleSceneInfo(makeSceneInfoMsg())
    scene.selectedEntityIdx.value = 0

    expect(scene.entities.value.length).toBe(1)
    expect(scene.selectedEntityIdx.value).toBe(0)

    scene.resetState()

    expect(scene.entities.value.length).toBe(0)
    expect(scene.selectedEntityIdx.value).toBeNull()
    expect(Object.keys(scene.entityEditState).length).toBe(0)
  })
})
