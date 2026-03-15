<script setup>
import { inject, computed, ref, watch } from 'vue'
import { SET_TARGET_FPS } from '../utils/protocol.js'
import ScrubInput from './ScrubInput.vue'

const ws = inject('ws')
const scene = inject('scene')

const targetFpsInitialized = ref(false)
const targetFps = ref(100)

// Initialize target FPS to 1/dt when scene info arrives
watch(() => scene.simDt.value, (dt) => {
  if (!targetFpsInitialized.value && dt > 0) {
    const v = Math.round(1 / dt)
    targetFps.value = v
    ws.send({ type: SET_TARGET_FPS, fps: v })
    targetFpsInitialized.value = true
  }
})

const simDtFps = computed(() => {
  const dt = scene.simDt.value
  return dt > 0 ? Math.round(1 / dt) : 0
})

const actualFps = computed(() => scene.fps.value.toFixed(0))
const timeText = computed(() => scene.simTime.value.toFixed(3))

function onPlay() {
  scene.simControl(scene.paused.value ? 'play' : 'pause')
}

function onTargetFpsChange(val) {
  const v = Math.round(val)
  targetFps.value = v
  ws.send({ type: SET_TARGET_FPS, fps: v })
}
</script>

<template>
  <div class="bar">
    <!-- Controls -->
    <div class="bar__controls">
      <button
        class="bar__btn"
        :class="{ 'bar__btn--active': !scene.paused.value }"
        @click="onPlay"
        :title="scene.paused.value ? 'Play' : 'Pause'"
      >
        <span v-if="scene.paused.value">&#9654;</span>
        <span v-else>&#10074;&#10074;</span>
      </button>
      <button class="bar__btn" @click="scene.simControl('step')" title="Step"><span>&#9654;&#10074;</span></button>
      <button class="bar__btn" @click="scene.simControl('reset')" title="Reset"><span>&#8635;</span></button>
    </div>

    <div class="bar__sep" />

    <!-- FPS: (sim_rate, [target], actual) -->
    <div class="bar__info">
      <span class="bar__info-label">FPS</span>
      <div class="bar__fps-group">
        <span class="bar__fps-static" title="1/dt (physics rate)">{{ simDtFps }}</span>
        <span class="bar__fps-comma">,</span>
        <div class="bar__fps-target" title="Target FPS (editable)">
          <ScrubInput
            :model-value="targetFps"
            :range="120"
            :min="0"
            :max="240"
            :bounded="true"
            :step="1"
            :decimals="0"
            @update:model-value="onTargetFpsChange"
            @scrub="onTargetFpsChange"
            @scrub-end="() => {}"
          />
        </div>
        <span class="bar__fps-comma">,</span>
        <span class="bar__fps-actual" title="Actual FPS">{{ actualFps }}</span>
      </div>
    </div>

    <div class="bar__sep" />

    <!-- Time, step, paused -->
    <div class="bar__status">
      <span class="bar__dot" :class="{ 'bar__dot--on': ws.connected.value }" />
      <span class="bar__stat"><span class="bar__stat-label">Time</span> <span class="bar__stat-val">{{ timeText }}</span></span>
      <span class="bar__stat"><span class="bar__stat-label">Step</span> <span class="bar__stat-val">{{ scene.step.value }}</span></span>
      <span class="bar__paused" :class="{ 'bar__paused--visible': scene.paused.value }">PAUSED</span>
    </div>
  </div>
</template>

<style scoped>
.bar {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 15;
  display: flex;
  align-items: center;
  padding: 6px 10px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: var(--radius-xl);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  gap: 8px;
}

.bar__controls {
  display: flex;
  gap: 3px;
}

.bar__btn {
  width: 34px;
  height: 28px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: rgba(255, 255, 255, 0.5);
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}

.bar__btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: var(--text);
}

.bar__btn--active {
  background: var(--accent);
  color: #fff;
}

.bar__sep {
  width: 1px;
  height: 18px;
  background: rgba(255, 255, 255, 0.12);
  flex-shrink: 0;
}

.bar__info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bar__info-label {
  color: rgba(255, 255, 255, 0.35);
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
}

.bar__fps-group {
  display: flex;
  align-items: center;
  gap: 4px;
}

.bar__fps-static {
  color: var(--text);
  font-size: 12px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  min-width: 28px;
  text-align: center;
}

.bar__fps-comma {
  color: rgba(255, 255, 255, 0.3);
  font-size: 12px;
}

.bar__fps-target {
  width: 56px;
}

.bar__fps-actual {
  color: var(--accent);
  font-size: 12px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  min-width: 28px;
  text-align: center;
}

.bar__status {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 200px;
  padding: 0 4px;
  pointer-events: none;
  user-select: none;
}

.bar__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--danger);
  flex-shrink: 0;
}

.bar__dot--on {
  background: var(--success);
  box-shadow: 0 0 6px rgba(76, 175, 80, 0.5);
}

.bar__stat {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

.bar__stat-label {
  color: rgba(255, 255, 255, 0.35);
  font-size: 11px;
  font-weight: 500;
}

.bar__stat-val {
  color: var(--text);
  font-size: 12px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  min-width: 36px;
}

/* Always rendered to reserve space, hidden when not paused */
.bar__paused {
  color: var(--warning);
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.5px;
  visibility: hidden;
}

.bar__paused--visible {
  visibility: visible;
}
</style>
