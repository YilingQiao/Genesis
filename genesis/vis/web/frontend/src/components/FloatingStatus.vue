<script setup>
import { inject, computed } from 'vue'

const ws = inject('ws')
const scene = inject('scene')

const fpsText = computed(() => scene.fps.value.toFixed(0))
const timeText = computed(() => scene.simTime.value.toFixed(3) + 's')
</script>

<template>
  <div class="status-badge">
    <span class="status-badge__dot" :class="{ 'status-badge__dot--connected': ws.connected.value }" />
    <span class="status-badge__text">{{ ws.connected.value ? 'Connected' : 'Disconnected' }}</span>
    <span class="status-badge__sep" />
    <span class="status-badge__item">
      <span class="status-badge__label">FPS</span>
      <span class="status-badge__value">{{ fpsText }}</span>
    </span>
    <span class="status-badge__sep" />
    <span class="status-badge__item">
      <span class="status-badge__label">Time</span>
      <span class="status-badge__value">{{ timeText }}</span>
    </span>
    <span class="status-badge__sep" />
    <span class="status-badge__item">
      <span class="status-badge__label">Step</span>
      <span class="status-badge__value">{{ scene.step.value }}</span>
    </span>
    <span v-if="scene.paused.value" class="status-badge__paused">PAUSED</span>
  </div>
</template>

<style scoped>
.status-badge {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 18px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: var(--radius-xl);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  pointer-events: none;
  user-select: none;
}

.status-badge__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--danger);
  flex-shrink: 0;
}

.status-badge__dot--connected {
  background: var(--success);
  box-shadow: 0 0 8px rgba(76, 175, 80, 0.5);
}

.status-badge__text {
  color: var(--text);
  font-weight: 500;
  font-size: 12px;
  letter-spacing: 0.2px;
}

.status-badge__sep {
  width: 1px;
  height: 14px;
  background: rgba(255, 255, 255, 0.15);
}

.status-badge__item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.status-badge__label {
  color: rgba(255, 255, 255, 0.4);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

.status-badge__value {
  color: var(--text);
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.2px;
}

.status-badge__paused {
  color: var(--warning);
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.5px;
  margin-left: 2px;
}
</style>
