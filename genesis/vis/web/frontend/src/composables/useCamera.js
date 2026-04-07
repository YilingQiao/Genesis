import { ref } from 'vue'
import { CAMERA_UPDATE } from '../utils/protocol.js'

/**
 * Camera state composable.
 * Manages camera position, lookat, fov, and provides command methods.
 */
export function useCamera(send) {
  const pos = ref([2, 2, 1.5])
  const lookat = ref([0, 0, 0.5])
  const fov = ref(30)

  // Track which camera input refs are focused to implement focus guard
  const focusedInputs = ref(new Set())

  function handleCameraState(cameraState) {
    if (!cameraState) return
    // Skip updates for focused fields (focus guard)
    if (focusedInputs.value.size > 0) return
    if (cameraState.pos) pos.value = [...cameraState.pos]
    if (cameraState.lookat) lookat.value = [...cameraState.lookat]
    if (cameraState.fov != null) fov.value = cameraState.fov
  }

  function setCameraState(cameraState) {
    // Force set without focus guard (used for initial SCENE_INFO)
    if (cameraState.pos) pos.value = [...cameraState.pos]
    if (cameraState.lookat) lookat.value = [...cameraState.lookat]
    if (cameraState.fov != null) fov.value = cameraState.fov
  }

  function orbit(dAzimuth, dElevation) {
    send({ type: CAMERA_UPDATE, action: 'orbit', d_azimuth: dAzimuth, d_elevation: dElevation })
  }

  function pan(dx, dy) {
    send({ type: CAMERA_UPDATE, action: 'pan', dx, dy })
  }

  function zoom(factor) {
    send({ type: CAMERA_UPDATE, action: 'zoom', factor })
  }

  function setPose(newPos, newLookat) {
    const msg = { type: CAMERA_UPDATE, action: 'set_pose' }
    if (newPos) {
      if (newPos.some((v) => !Number.isFinite(v))) return
      msg.pos = newPos
    }
    if (newLookat) {
      if (newLookat.some((v) => !Number.isFinite(v))) return
      msg.lookat = newLookat
    }
    send(msg)
  }

  function setFov(newFov) {
    if (!Number.isFinite(newFov)) return
    send({ type: CAMERA_UPDATE, action: 'set_fov', fov: newFov })
  }

  function resetCamera() {
    send({ type: CAMERA_UPDATE, action: 'reset' })
  }

  function markFocused(key) {
    focusedInputs.value.add(key)
  }

  function markBlurred(key) {
    focusedInputs.value.delete(key)
  }

  function resetState() {
    pos.value = [0, 0, 0]
    lookat.value = [0, 0, 0]
    fov.value = 30
    focusedInputs.value.clear()
  }

  return {
    pos,
    lookat,
    fov,
    focusedInputs,
    handleCameraState,
    setCameraState,
    orbit,
    pan,
    zoom,
    setPose,
    setFov,
    resetCamera,
    markFocused,
    markBlurred,
    resetState,
  }
}
