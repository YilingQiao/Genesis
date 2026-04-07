import { ref, reactive } from 'vue'
import { SIM_CONTROL, ENTITY_UPDATE, VIS_TOGGLE } from '../utils/protocol.js'
import { quatToEuler } from '../utils/gizmoMath.js'

/**
 * Scene state composable.
 * Manages entities, simulation status, visualization state, and entity editing.
 */
export function useScene(send) {
  const entities = ref([])
  const selectedEntityIdx = ref(null)
  const paused = ref(false)
  const simTime = ref(0)
  const step = ref(0)
  const fps = ref(0)
  const nEnvs = ref(1)
  const simDt = ref(0.01)
  const visState = reactive({
    shadows: true,
    world_frame: false,
    link_frame: false,
    link_frame_size: 0.1,
    camera_frustum: false,
    wireframe: false,
    face_normals: false,
    vertex_normals: false,
    orthographic: false,
  })

  // Track which vis keys the server actually supports
  const visCapabilities = ref(new Set())

  // Per-entity editing state
  const entityEditState = reactive({})

  function handleSceneInfo(msg) {
    if (msg.entities) {
      entities.value = msg.entities.map((e) => ({
        ...e,
        qpos: e.qpos ? [...e.qpos] : [],
      }))
    }

    if (msg.vis_state) {
      // Track supported capabilities
      visCapabilities.value = new Set(Object.keys(msg.vis_state))
      Object.keys(msg.vis_state).forEach((key) => {
        if (key in visState) {
          visState[key] = msg.vis_state[key]
        }
      })
    } else {
      visCapabilities.value = new Set()
    }

    nEnvs.value = msg.n_envs || 1
    if (msg.sim_dt) simDt.value = msg.sim_dt

    // Initialize edit state for all entities
    entities.value.forEach((e) => {
      const idx = e.idx != null ? e.idx : 0
      if (!entityEditState[idx]) {
        entityEditState[idx] = {
          mode: 'quat',
          euler: { roll: 0, pitch: 0, yaw: 0 },
        }
      }
      // Compute initial euler from qpos if free joint
      resyncEuler(idx)
    })
  }

  /**
   * Recompute Euler angles from the current quaternion in entity qpos.
   * Called whenever qpos changes from any source (quat input, gizmo, mode switch).
   */
  function resyncEuler(entityIdx) {
    const e = findEntity(entityIdx)
    if (!e || !e.has_free_joint || !e.quat_groups || e.quat_groups.length === 0) return
    const state = entityEditState[entityIdx]
    if (!state) return
    const qg = e.quat_groups[0]
    const qpos = e.qpos || []
    const qw = qpos[qg[0]] || 1
    const qx = qpos[qg[0] + 1] || 0
    const qy = qpos[qg[0] + 2] || 0
    const qz = qpos[qg[0] + 3] || 0
    state.euler = quatToEuler(qw, qx, qy, qz)
  }

  function handleStateUpdate(msg) {
    if (msg.time != null) simTime.value = msg.time
    if (msg.step != null) step.value = msg.step
    if (msg.fps != null) fps.value = msg.fps
    if (msg.paused != null) paused.value = msg.paused
    // Update entity qpos from server so gizmo tracks moving objects
    if (msg.entity_positions) {
      for (const ep of msg.entity_positions) {
        const e = findEntity(ep.idx)
        if (e && ep.qpos) {
          e.qpos = [...ep.qpos]
        }
      }
    }
  }

  function autoPause() {
    if (!paused.value) {
      send({ type: SIM_CONTROL, action: 'pause' })
      paused.value = true
    }
  }

  function simControl(action) {
    send({ type: SIM_CONTROL, action })
    if (action === 'play') paused.value = false
    else if (action === 'pause') paused.value = true
  }

  function toggleVis(property, value) {
    send({ type: VIS_TOGGLE, property, value })
    if (property in visState) {
      visState[property] = value
    }
  }

  function updateEntityDof(entityIdx, dofIdx, value) {
    if (!Number.isFinite(value)) return
    send({ type: ENTITY_UPDATE, entity_idx: entityIdx, dof_idx: dofIdx, value })
    // Keep local qpos in sync so subsequent full-qpos sends use fresh data
    const e = findEntity(entityIdx)
    if (e && e.qpos && dofIdx < e.qpos.length) {
      e.qpos[dofIdx] = value
    }
  }

  function updateEntityQpos(entityIdx, qpos, quatGroups) {
    // Validate all values are finite
    if (qpos.some((v) => !Number.isFinite(v))) return
    send({
      type: ENTITY_UPDATE,
      entity_idx: entityIdx,
      qpos,
      normalize_quats: true,
      quat_groups: quatGroups || [],
    })
    // Resync Euler after qpos changes
    resyncEuler(entityIdx)
  }

  function updateEntityVisMode(entityIdx, visMode) {
    send({ type: ENTITY_UPDATE, entity_idx: entityIdx, vis_mode: visMode })
    const e = findEntity(entityIdx)
    if (e) e.vis_mode = visMode
  }

  function updateEntityWireframe(entityIdx, wireframe) {
    send({ type: ENTITY_UPDATE, entity_idx: entityIdx, wireframe })
    const e = findEntity(entityIdx)
    if (e) e._wireframe = wireframe
  }

  function updateEntityContactViz(entityIdx, contactViz) {
    send({ type: ENTITY_UPDATE, entity_idx: entityIdx, contact_viz: contactViz })
    const e = findEntity(entityIdx)
    if (e) e.visualize_contact = contactViz
  }

  function findEntity(entityIdx) {
    return entities.value.find((e) => (e.idx != null ? e.idx : 0) === entityIdx) || null
  }

  function resetState() {
    entities.value = []
    selectedEntityIdx.value = null
    paused.value = false
    simTime.value = 0
    step.value = 0
    fps.value = 0
    nEnvs.value = 1
    visState.shadows = true
    visState.world_frame = false
    visState.link_frame = false
    visState.link_frame_size = 0.1
    visState.camera_frustum = false
    visState.wireframe = false
    visState.face_normals = false
    visState.vertex_normals = false
    visState.orthographic = false
    Object.keys(entityEditState).forEach((k) => delete entityEditState[k])
    visCapabilities.value = new Set()
  }

  return {
    entities,
    selectedEntityIdx,
    paused,
    simTime,
    step,
    fps,
    nEnvs,
    simDt,
    visState,
    visCapabilities,
    entityEditState,
    handleSceneInfo,
    handleStateUpdate,
    autoPause,
    simControl,
    toggleVis,
    updateEntityDof,
    updateEntityQpos,
    updateEntityVisMode,
    updateEntityWireframe,
    updateEntityContactViz,
    findEntity,
    resyncEuler,
    resetState,
  }
}
