<script setup>
const props = defineProps({
  options: { type: Array, required: true },
  modelValue: { type: [String, Boolean, Number], required: true },
})
const emit = defineEmits(['update:modelValue'])
</script>

<template>
  <div class="seg">
    <button
      v-for="opt in options"
      :key="opt.value"
      class="seg__btn"
      :class="{ 'seg__btn--active': modelValue === opt.value }"
      @click="emit('update:modelValue', opt.value)"
    >{{ opt.label }}</button>
  </div>
</template>

<style scoped>
.seg {
  display: inline-flex;
  background: rgba(255, 255, 255, 0.06);
  border-radius: var(--radius-sm);
  padding: 2px;
  gap: 2px;
}

.seg__btn {
  padding: 5px 14px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--text-muted);
  font-family: var(--font);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
  white-space: nowrap;
}

.seg__btn:hover:not(.seg__btn--active) {
  color: var(--text);
  background: rgba(255, 255, 255, 0.06);
}

.seg__btn--active {
  background: var(--accent);
  color: #fff;
  font-weight: 600;
}
</style>
