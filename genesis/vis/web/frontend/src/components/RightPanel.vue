<script setup>
import { inject, computed } from 'vue'
import SegmentedToggle from './SegmentedToggle.vue'
import ScrubInput from './ScrubInput.vue'

const scene = inject('scene')
const camera = inject('camera')
const gizmo = inject('gizmo')

// ---- Visualization Toggles (no face/vertex normals) ----
const boolToggles = [
  { key: 'shadows', label: 'Shadows' },
  { key: 'world_frame', label: 'World Frame' },
  { key: 'link_frame', label: 'Link Frame' },
  { key: 'camera_frustum', label: 'Camera Frustum' },
  { key: 'wireframe', label: 'Wireframe' },
]

const supportedBoolToggles = computed(() =>
  boolToggles.filter((t) => scene.visCapabilities.value.has(t.key))
)

const hasOrthographic = computed(() => scene.visCapabilities.value.has('orthographic'))
const onOffOptions = [
  { label: 'On', value: true },
  { label: 'Off', value: false },
]

// ---- Camera scrub handlers ----
function onCamPosChange(axis, value) {
  if (!Number.isFinite(value)) return
  const newPos = [...camera.pos.value]
  newPos[axis] = value
  camera.pos.value = newPos
  camera.setPose(newPos, null)
}

function onCamLookatChange(axis, value) {
  if (!Number.isFinite(value)) return
  const newLookat = [...camera.lookat.value]
  newLookat[axis] = value
  camera.lookat.value = newLookat
  camera.setPose(null, newLookat)
}

function onCamFovChange(value) {
  if (!Number.isFinite(value)) return
  camera.fov.value = value
  camera.setFov(value)
}

// ---- Selected Entity ----
const selectedEntity = computed(() => {
  if (scene.selectedEntityIdx.value === null) return null
  return scene.findEntity(scene.selectedEntityIdx.value)
})

// ---- Gizmo ----
function setGizmoActive(entityIdx, active) {
  if (active) {
    if (gizmo.entityIdx.value !== entityIdx) gizmo.toggle(entityIdx)
  } else {
    if (gizmo.entityIdx.value === entityIdx) gizmo.toggle(entityIdx)
  }
}

const gizmoModeOptions = [
  { label: 'Translate', value: 'translate' },
  { label: 'Rotate', value: 'rotate' },
]
</script>

<template>
  <div class="right-panel">
    <!-- Camera -->
    <div class="section-header">Camera</div>
    <div class="rp-section">
      <div class="cam-group">
        <div class="cam-group__label">Position</div>
        <div v-for="(label, i) in ['X', 'Y', 'Z']" :key="'pos-' + label" class="cam-row">
          <ScrubInput
            :label="label"
            :model-value="camera.pos.value[i]"
            :range="1"
            :step="0.01"
            :decimals="2"
            @update:model-value="(v) => onCamPosChange(i, v)"
            @scrub="(v) => onCamPosChange(i, v)"
            @scrub-end="() => {}"
          />
        </div>
      </div>
      <div class="cam-group">
        <div class="cam-group__label">Look At</div>
        <div v-for="(label, i) in ['X', 'Y', 'Z']" :key="'lookat-' + label" class="cam-row">
          <ScrubInput
            :label="label"
            :model-value="camera.lookat.value[i]"
            :range="1"
            :step="0.01"
            :decimals="2"
            @update:model-value="(v) => onCamLookatChange(i, v)"
            @scrub="(v) => onCamLookatChange(i, v)"
            @scrub-end="() => {}"
          />
        </div>
      </div>
      <div class="cam-group">
        <div class="cam-group__label">Field of View</div>
        <div class="cam-row">
          <ScrubInput
            label="FOV"
            :model-value="camera.fov.value"
            :range="52.5"
            :min="15"
            :max="120"
            :bounded="true"
            :step="0.5"
            :decimals="1"
            @update:model-value="onCamFovChange"
            @scrub="onCamFovChange"
            @scrub-end="() => {}"
          />
        </div>
      </div>
      <button class="btn-reset-cam" @click="camera.resetCamera()">Reset Camera</button>
    </div>

    <!-- Global Settings -->
    <div class="section-header">Global Settings</div>
    <div class="rp-section">
      <div v-for="toggle in supportedBoolToggles" :key="toggle.key" class="toggle-row">
        <span class="toggle-row__label">{{ toggle.label }}</span>
        <SegmentedToggle
          :options="onOffOptions"
          :model-value="scene.visState[toggle.key]"
          @update:model-value="(v) => scene.toggleVis(toggle.key, v)"
        />
      </div>
      <div v-if="hasOrthographic" class="toggle-row">
        <span class="toggle-row__label">Orthographic</span>
        <SegmentedToggle
          :options="onOffOptions"
          :model-value="scene.visState.orthographic"
          @update:model-value="(v) => scene.toggleVis('orthographic', v)"
        />
      </div>
    </div>

    <!-- Entity Visual Properties -->
    <div class="section-header">Entity Properties</div>
    <div class="rp-section">
      <template v-if="selectedEntity">
        <div class="entity-name-bar">
          {{ selectedEntity.name || ('Entity ' + selectedEntity.idx) }}
        </div>

        <div class="control-row">
          <span class="control-row__label">Vis Mode</span>
          <SegmentedToggle
            :options="[{ label: 'Visual', value: 'visual' }, { label: 'Collision', value: 'collision' }]"
            :model-value="selectedEntity.vis_mode || 'visual'"
            @update:model-value="(v) => scene.updateEntityVisMode(selectedEntity.idx, v)"
          />
        </div>
        <div class="control-row">
          <span class="control-row__label">Wireframe</span>
          <SegmentedToggle
            :options="onOffOptions"
            :model-value="selectedEntity._wireframe || false"
            @update:model-value="(v) => scene.updateEntityWireframe(selectedEntity.idx, v)"
          />
        </div>
        <div class="control-row">
          <span class="control-row__label">Contacts</span>
          <SegmentedToggle
            :options="onOffOptions"
            :model-value="selectedEntity.visualize_contact || false"
            @update:model-value="(v) => scene.updateEntityContactViz(selectedEntity.idx, v)"
          />
        </div>

        <template v-if="selectedEntity.has_free_joint">
          <div class="control-row">
            <span class="control-row__label">Gizmo</span>
            <SegmentedToggle
              :options="[{ label: 'Off', value: false }, { label: 'On', value: true }]"
              :model-value="gizmo.entityIdx.value === selectedEntity.idx"
              @update:model-value="(v) => setGizmoActive(selectedEntity.idx, v)"
            />
          </div>
          <div v-if="gizmo.entityIdx.value === selectedEntity.idx" class="control-row">
            <span class="control-row__label">Mode</span>
            <SegmentedToggle
              :options="gizmoModeOptions"
              :model-value="gizmo.mode.value"
              @update:model-value="(v) => gizmo.setMode(v)"
            />
          </div>
        </template>
      </template>
      <div v-else class="rp-empty">
        Select an entity
      </div>
    </div>
  </div>
</template>

<style scoped>
.right-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.rp-section {
  padding: 12px 18px 16px;
}

.rp-empty {
  color: var(--text-dim);
  font-size: 12px;
  text-align: center;
  padding: 24px 0;
}

.toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 0;
  gap: 10px;
}

.toggle-row__label {
  font-size: 12px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.cam-group {
  margin-bottom: 14px;
}

.cam-group__label {
  font-size: 10px;
  color: var(--text-muted);
  margin-bottom: 6px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.cam-row {
  margin-bottom: 4px;
}

.btn-reset-cam {
  width: 100%;
  padding: 8px 0;
  margin-top: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-muted);
  font-family: var(--font);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
}

.btn-reset-cam:hover {
  background: rgba(255, 255, 255, 0.12);
  color: var(--text);
}

.entity-name-bar {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  padding-bottom: 12px;
  margin-bottom: 8px;
}

.control-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 0;
  gap: 10px;
}

.control-row__label {
  font-size: 12px;
  color: var(--text-muted);
  flex-shrink: 0;
}
</style>
