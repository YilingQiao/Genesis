<script setup>
import { inject, ref } from 'vue'
import SegmentedToggle from './SegmentedToggle.vue'
import ScrubInput from './ScrubInput.vue'
import { eulerToQuat } from '../utils/gizmoMath.js'

const scene = inject('scene')

const expanded = ref({})

function toggleExpand(idx) {
  expanded.value[idx] = !expanded.value[idx]
}

function selectEntity(idx) {
  scene.selectedEntityIdx.value = idx
}

function entityDofLabel(entity) {
  const nDofs = entity.n_dofs || 0
  const nQs = entity.n_qs || nDofs
  let label = nDofs + ' DOF'
  if (nQs !== nDofs) label += ', ' + nQs + ' qs'
  return label
}

// ---- DOF Scrub ----
const scrubTimers = {}

function onDofScrub(entityIdx, qIdx, value) {
  if (!Number.isFinite(value)) return
  const entity = scene.findEntity(entityIdx)
  if (!entity) return
  scene.autoPause()
  if (entity.qpos && qIdx < entity.qpos.length) {
    entity.qpos[qIdx] = value
  }
  const key = entityIdx + '-' + qIdx
  clearTimeout(scrubTimers[key])
  scrubTimers[key] = setTimeout(() => {
    scene.updateEntityDof(entityIdx, qIdx, value)
    delete scrubTimers[key]
  }, 50)
}

function onDofScrubEnd(entityIdx, qIdx) {
  const entity = scene.findEntity(entityIdx)
  if (!entity) return
  const value = entity.qpos?.[qIdx]
  if (value != null && Number.isFinite(value)) {
    const key = entityIdx + '-' + qIdx
    clearTimeout(scrubTimers[key])
    delete scrubTimers[key]
    scene.updateEntityDof(entityIdx, qIdx, value)
  }
}

function onDofSet(entityIdx, qIdx, value) {
  if (!Number.isFinite(value)) return
  scene.autoPause()
  const entity = scene.findEntity(entityIdx)
  if (entity && entity.qpos) {
    entity.qpos[qIdx] = value
  }
  scene.updateEntityDof(entityIdx, qIdx, value)
}

// ---- Quaternion Input ----
function onQuatSet(entityIdx, qIdx, value) {
  if (!Number.isFinite(value)) return
  const entity = scene.findEntity(entityIdx)
  if (!entity) return
  scene.autoPause()
  const qpos = (entity.qpos || []).slice()
  qpos[qIdx] = value
  const quatGroups = entity.quat_groups || []
  for (const [start, end] of quatGroups) {
    let sum = 0
    for (let i = start; i < end; i++) sum += qpos[i] * qpos[i]
    const norm = Math.sqrt(sum)
    if (norm > 1e-8) {
      for (let i = start; i < end; i++) qpos[i] /= norm
    }
  }
  entity.qpos = qpos
  scene.updateEntityQpos(entityIdx, qpos, quatGroups)
}

// ---- Euler Input ----
function onEulerAngleChange(entityIdx, eulerKey, value) {
  if (!Number.isFinite(value)) return
  const entity = scene.findEntity(entityIdx)
  if (!entity) return
  scene.autoPause()
  const state = scene.entityEditState[entityIdx]
  if (!state) return
  state.euler[eulerKey] = value
  const q = eulerToQuat(state.euler.roll, state.euler.pitch, state.euler.yaw)
  const qpos = (entity.qpos || []).slice()
  const quatGroups = entity.quat_groups || []
  if (quatGroups.length > 0) {
    const [start] = quatGroups[0]
    qpos[start] = q.w
    qpos[start + 1] = q.x
    qpos[start + 2] = q.y
    qpos[start + 3] = q.z
  }
  entity.qpos = qpos
  scene.updateEntityQpos(entityIdx, qpos, quatGroups)
}

// ---- Mode toggle ----
const modeOptions = [
  { label: 'Quaternion', value: 'quat' },
  { label: 'Euler', value: 'euler' },
]

function getEditMode(entityIdx) {
  return scene.entityEditState[entityIdx]?.mode || 'quat'
}

function setEditMode(entityIdx, mode) {
  if (scene.entityEditState[entityIdx]) {
    scene.entityEditState[entityIdx].mode = mode
    if (mode === 'euler') {
      scene.resyncEuler(entityIdx)
    }
  }
}

// ---- Helpers ----
function getQSliders(entity) {
  const nQs = entity.n_qs || 0
  const qNames = entity.q_names || []
  const qLower = entity.q_limits_lower || []
  const qUpper = entity.q_limits_upper || []
  const qpos = entity.qpos || []
  const result = []
  for (let i = 0; i < nQs; i++) {
    result.push({
      idx: i,
      name: qNames[i] || ('q' + i),
      lower: qLower[i] != null ? qLower[i] : -3.14,
      upper: qUpper[i] != null ? qUpper[i] : 3.14,
      value: qpos[i] != null ? qpos[i] : 0,
    })
  }
  return result
}

function getPositionSliders(entity) {
  const all = getQSliders(entity)
  const freeStart = entity.free_joint_q_start || 0
  const labels = ['x', 'y', 'z']
  return all.slice(freeStart, freeStart + 3).map((s, i) => ({ ...s, name: labels[i] }))
}

function getQuatSliders(entity) {
  const all = getQSliders(entity)
  const freeStart = entity.free_joint_q_start || 0
  const labels = ['qw', 'qx', 'qy', 'qz']
  return all.slice(freeStart + 3, freeStart + 7).map((s, i) => ({ ...s, name: labels[i] }))
}

function getNonFreeSliders(entity) {
  const all = getQSliders(entity)
  const freeStart = entity.free_joint_q_start || 0
  const freeEnd = freeStart + 7
  return all.filter((s) => s.idx < freeStart || s.idx >= freeEnd)
}

function sliderRange(s) {
  return Math.min(Math.abs(s.upper - s.lower) / 2, 1)
}
</script>

<template>
  <div class="left-panel">
    <div class="left-panel__header">
      <div class="left-panel__logo">
        <span class="left-panel__logo-icon">G</span>
        <span class="left-panel__logo-text">Genesis</span>
      </div>
    </div>

    <div v-if="scene.nEnvs.value > 1" class="left-panel__env-note">
      Controlling env 0 of {{ scene.nEnvs.value }}
    </div>

    <div class="section-header section-header--bright">Entity State</div>

    <div class="left-panel__list">
      <div
        v-for="entity in scene.entities.value"
        :key="entity.idx"
        class="entity-item"
        :class="{ 'entity-item--selected': scene.selectedEntityIdx.value === entity.idx }"
      >
        <div
          class="entity-item__header"
          @click="selectEntity(entity.idx); toggleExpand(entity.idx)"
        >
          <div class="entity-item__left">
            <span
              class="entity-item__arrow"
              :class="{ 'entity-item__arrow--open': expanded[entity.idx] }"
              v-html="'&#9654;'"
            ></span>
            <span class="entity-item__icon">
              <svg v-if="(entity.n_dofs || 0) > 0" width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="2" y="2" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/>
                <circle cx="7" cy="7" r="2" fill="currentColor"/>
              </svg>
              <svg v-else width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="4" stroke="currentColor" stroke-width="1.5" fill="none"/>
              </svg>
            </span>
            <span class="entity-item__name">{{ entity.name || ('Entity ' + entity.idx) }}</span>
          </div>
          <span class="entity-item__badge">{{ entityDofLabel(entity) }}</span>
        </div>

        <!-- Expanded body -->
        <div v-if="expanded[entity.idx]" class="entity-item__body">
          <template v-if="(entity.n_qs || 0) > 0">

            <!-- Non-free-joint: all DOFs as bounded scrub controls -->
            <template v-if="!entity.has_free_joint">
              <div v-for="s in getQSliders(entity)" :key="s.idx" class="dof-row">
                <ScrubInput
                  :label="s.name"
                  :model-value="s.value"
                  :range="sliderRange(s)"
                  :min="s.lower"
                  :max="s.upper"
                  :bounded="true"
                  :step="0.001"
                  @update:model-value="(v) => onDofSet(entity.idx, s.idx, v)"
                  @scrub="(v) => onDofScrub(entity.idx, s.idx, v)"
                  @scrub-end="() => onDofScrubEnd(entity.idx, s.idx)"
                />
              </div>
            </template>

            <!-- Free-joint entity -->
            <template v-if="entity.has_free_joint">
              <!-- Position XYZ (unbounded, range +-1, always visible, vertical) -->
              <div class="body__section-label">Position</div>
              <div v-for="s in getPositionSliders(entity)" :key="'pos-' + s.idx" class="dof-row">
                <ScrubInput
                  :label="s.name"
                  :model-value="s.value"
                  :range="1"
                  :step="0.01"
                  :decimals="2"
                  @update:model-value="(v) => onDofSet(entity.idx, s.idx, v)"
                  @scrub="(v) => onDofScrub(entity.idx, s.idx, v)"
                  @scrub-end="() => onDofScrubEnd(entity.idx, s.idx)"
                />
              </div>

              <!-- Rotation mode toggle -->
              <div class="body__mode-toggle">
                <SegmentedToggle
                  :options="modeOptions"
                  :model-value="getEditMode(entity.idx)"
                  @update:model-value="(v) => setEditMode(entity.idx, v)"
                />
              </div>

              <!-- Quaternion rotation -->
              <template v-if="getEditMode(entity.idx) === 'quat'">
                <div v-for="s in getQuatSliders(entity)" :key="'q-' + s.idx" class="dof-row">
                  <ScrubInput
                    :label="s.name"
                    :model-value="s.value"
                    :range="0.5"
                    :step="0.01"
                    :decimals="4"
                    @update:model-value="(v) => onQuatSet(entity.idx, s.idx, v)"
                    @scrub-end="() => {}"
                  />
                </div>
              </template>

              <!-- Euler rotation (vertical) -->
              <template v-else>
                <div class="body__section-label">Rotation (deg)</div>
                <div v-for="info in [
                  { label: 'R', key: 'roll' },
                  { label: 'P', key: 'pitch' },
                  { label: 'Y', key: 'yaw' },
                ]" :key="info.key" class="dof-row">
                  <ScrubInput
                    :label="info.label"
                    :model-value="scene.entityEditState[entity.idx]?.euler?.[info.key] || 0"
                    :range="45"
                    :step="1"
                    :decimals="1"
                    @update:model-value="(v) => onEulerAngleChange(entity.idx, info.key, v)"
                    @scrub-end="() => {}"
                  />
                </div>
              </template>

              <!-- Remaining non-free joints (bounded) -->
              <div v-for="s in getNonFreeSliders(entity)" :key="'nf-' + s.idx" class="dof-row">
                <ScrubInput
                  :label="s.name"
                  :model-value="s.value"
                  :range="sliderRange(s)"
                  :min="s.lower"
                  :max="s.upper"
                  :bounded="true"
                  :step="0.001"
                  @update:model-value="(v) => onDofSet(entity.idx, s.idx, v)"
                  @scrub="(v) => onDofScrub(entity.idx, s.idx, v)"
                  @scrub-end="() => onDofScrubEnd(entity.idx, s.idx)"
                />
              </div>
            </template>

          </template>
          <div v-else class="body__no-joints">Static entity</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.left-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.left-panel__header {
  padding: 18px 18px 14px;
}

.left-panel__logo {
  display: flex;
  align-items: center;
  gap: 10px;
}

.left-panel__logo-icon {
  width: 26px;
  height: 26px;
  border-radius: 7px;
  background: var(--accent);
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

.left-panel__logo-text {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.3px;
}

.left-panel__env-note {
  color: var(--warning);
  font-size: 11px;
  padding: 10px 18px;
  background: rgba(240, 160, 48, 0.08);
}

.left-panel__list {
  flex: 1;
  overflow-y: auto;
  padding: 6px 0;
}

/* Entity items */
.entity-item {
  margin: 2px 8px;
  border-radius: 8px;
}

.entity-item--selected {
  background: var(--accent-soft);
}

.entity-item--selected .entity-item__name {
  color: var(--accent-hover);
}

.entity-item__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 12px;
  cursor: pointer;
  border-radius: 8px;
  transition: background var(--transition);
}

.entity-item__header:hover {
  background: var(--bg-hover);
}

.entity-item__left {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.entity-item__arrow {
  font-size: 7px;
  color: var(--text-dim);
  transition: transform var(--transition);
  flex-shrink: 0;
  width: 10px;
  display: inline-flex;
}

.entity-item__arrow--open {
  transform: rotate(90deg);
}

.entity-item__icon {
  flex-shrink: 0;
  color: var(--text-muted);
  display: flex;
  align-items: center;
}

.entity-item__name {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.entity-item__badge {
  font-size: 10px;
  color: var(--text-muted);
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.06);
  padding: 2px 8px;
  border-radius: 10px;
}

/* Expanded body */
.entity-item__body {
  padding: 6px 12px 14px 32px;
}

.body__section-label {
  font-size: 9px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
  margin-top: 8px;
}

.body__mode-toggle {
  margin: 8px 0 8px;
}

.body__no-joints {
  font-size: 11px;
  color: var(--text-dim);
  font-style: italic;
  padding: 4px 0;
}

.dof-row {
  margin-bottom: 5px;
}
</style>
