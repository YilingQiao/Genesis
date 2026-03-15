<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  modelValue: { type: Number, default: 0 },
  label: { type: String, default: '' },
  step: { type: Number, default: 0.01 },
  decimals: { type: Number, default: 3 },
  range: { type: Number, default: 1 },
  min: { type: Number, default: null },
  max: { type: Number, default: null },
  bounded: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'scrub', 'scrubEnd'])

const editing = ref(false)
const editValue = ref('')
const scrubOffset = ref(0)
const isDragging = ref(false)
let startX = 0
let startValue = 0

const displayValue = computed(() =>
  props.modelValue != null ? Number(props.modelValue).toFixed(props.decimals) : '0'
)

const barFill = computed(() => {
  if (props.bounded && props.min != null && props.max != null) {
    // Bounded: fill shows absolute position within [min, max]
    const span = props.max - props.min
    if (span <= 0) return 0
    const pct = ((props.modelValue - props.min) / span) * 100
    return Math.max(0, Math.min(100, pct))
  }
  // Unbounded: fill from center based on scrub offset
  if (!isDragging.value) return 50
  return 50 + (scrubOffset.value / props.range) * 50
})

function clampValue(val) {
  if (props.bounded && props.min != null && props.max != null) {
    return Math.max(props.min, Math.min(props.max, val))
  }
  return val
}

function onPointerDown(e) {
  if (editing.value) return
  e.preventDefault()
  isDragging.value = true
  startX = e.clientX
  startValue = props.modelValue || 0
  scrubOffset.value = 0
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
}

function onPointerMove(e) {
  const dx = e.clientX - startX
  const delta = (dx / 200) * props.range
  if (props.bounded) {
    // Bounded: accumulate from start, clamp to [min, max]
    const newVal = clampValue(startValue + delta)
    scrubOffset.value = delta
    emit('update:modelValue', Number(newVal.toFixed(props.decimals + 2)))
    emit('scrub', newVal)
  } else {
    // Unbounded: clamp delta to +-range, reset on release
    const clamped = Math.max(-props.range, Math.min(props.range, delta))
    scrubOffset.value = clamped
    const newVal = startValue + clamped
    emit('update:modelValue', Number(newVal.toFixed(props.decimals + 2)))
    emit('scrub', newVal)
  }
}

function onPointerUp() {
  isDragging.value = false
  if (!props.bounded) {
    scrubOffset.value = 0
  }
  window.removeEventListener('pointermove', onPointerMove)
  window.removeEventListener('pointerup', onPointerUp)
  emit('scrubEnd')
}

function onDoubleClick() {
  editing.value = true
  editValue.value = displayValue.value
}

function onEditBlur() {
  const val = parseFloat(editValue.value)
  if (Number.isFinite(val)) {
    emit('update:modelValue', clampValue(val))
    emit('scrubEnd')
  }
  editing.value = false
}

function onEditKeydown(e) {
  if (e.key === 'Enter') {
    e.target.blur()
  } else if (e.key === 'Escape') {
    editing.value = false
  }
}
</script>

<template>
  <div class="scrub" :class="{ 'scrub--dragging': isDragging }">
    <span v-if="label" class="scrub__label">{{ label }}</span>
    <div class="scrub__track" @pointerdown="onPointerDown" @dblclick="onDoubleClick">
      <div class="scrub__fill" :style="{ width: barFill + '%' }" />
      <input
        v-if="editing"
        class="scrub__input"
        type="number"
        v-model="editValue"
        :step="step"
        @blur="onEditBlur"
        @keydown="onEditKeydown"
        autofocus
      >
      <span v-else class="scrub__value">{{ displayValue }}</span>
    </div>
  </div>
</template>

<style scoped>
.scrub {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
}

.scrub__label {
  font-size: 9px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  min-width: 16px;
  flex-shrink: 0;
}

.scrub__track {
  flex: 1;
  position: relative;
  height: 24px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 5px;
  overflow: hidden;
  cursor: ew-resize;
  user-select: none;
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: border-color var(--transition);
}

.scrub--dragging .scrub__track {
  border-color: var(--accent);
}

.scrub__track:hover {
  border-color: rgba(255, 255, 255, 0.18);
}

.scrub__fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: var(--accent);
  opacity: 0.18;
  transition: width 0.05s linear;
  pointer-events: none;
}

.scrub__value {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  color: var(--text);
  pointer-events: none;
  z-index: 1;
}

.scrub__input {
  position: relative;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.06);
  border: none;
  color: var(--text);
  font-family: var(--font);
  font-size: 10px;
  text-align: center;
  outline: none;
  z-index: 1;
  padding: 0 4px;
}
</style>
