<script setup>
import { inject, computed } from 'vue'

const ws = inject('ws')
const scene = inject('scene')

const fpsText = computed(() => scene.fps.value.toFixed(0))
const timeText = computed(() => scene.simTime.value.toFixed(3))

function onPlay() {
  scene.simControl(scene.paused.value ? 'play' : 'pause')
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

    <!-- Status (fixed width so controls don't shift) -->
    <div class="bar__status">
      <span class="bar__dot" :class="{ 'bar__dot--on': ws.connected.value }" />
      <span class="bar__stat"><span class="bar__stat-label">FPS</span> <span class="bar__stat-val">{{ fpsText }}</span></span>
      <span class="bar__stat"><span class="bar__stat-label">Time</span> <span class="bar__stat-val">{{ timeText }}</span></span>
      <span class="bar__stat"><span class="bar__stat-label">Step</span> <span class="bar__stat-val">{{ scene.step.value }}</span></span>
      <span v-if="scene.paused.value" class="bar__paused">PAUSED</span>
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
  padding: 6px 8px 6px 10px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: var(--radius-xl);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  gap: 6px;
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

.bar__status {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 320px;
  padding: 0 6px;
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

.bar__paused {
  color: var(--warning);
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.5px;
}
</style>
