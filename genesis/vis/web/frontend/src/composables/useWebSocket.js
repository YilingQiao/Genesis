import { ref, onUnmounted } from 'vue'
import { SCENE_INFO, STATE_UPDATE } from '../utils/protocol.js'

/**
 * WebSocket connection composable.
 * Handles connect, reconnect with exponential backoff, message dispatch, and send.
 */
export function useWebSocket() {
  const connected = ref(false)
  const latestFrame = ref(null)

  let ws = null
  let reconnectDelay = 500
  let reconnectTimer = null
  let onSceneInfo = null
  let onStateUpdate = null
  let onReconnect = null

  // Generation token: incremented on every resetFrame/destroy to invalidate
  // any in-flight createImageBitmap promises from a previous connection.
  let frameGeneration = 0

  const MAX_DELAY = 30000
  const BACKOFF_MULTIPLIER = 2

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(protocol + '//' + location.host + '/ws')
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      connected.value = true
      reconnectDelay = 500
    }

    ws.onclose = () => {
      connected.value = false
      scheduleReconnect()
    }

    ws.onerror = () => {
      ws.close()
    }

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        handleBinaryFrame(event.data)
      } else {
        try {
          handleJsonMessage(JSON.parse(event.data))
        } catch (e) {
          console.warn('Malformed JSON from server:', e)
        }
      }
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) return
    if (onReconnect) onReconnect()
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, reconnectDelay)
    reconnectDelay = Math.min(reconnectDelay * BACKOFF_MULTIPLIER, MAX_DELAY)
  }

  function handleBinaryFrame(buffer) {
    const blob = new Blob([buffer], { type: 'image/jpeg' })
    // Capture generation at decode start so stale decodes are discarded
    const gen = frameGeneration
    createImageBitmap(blob)
      .then((bitmap) => {
        // If generation changed (reconnect/destroy happened), discard this bitmap
        if (gen !== frameGeneration) {
          bitmap.close()
          return
        }
        const old = latestFrame.value
        latestFrame.value = bitmap
        if (old) old.close()
      })
      .catch(() => {})
  }

  function handleJsonMessage(msg) {
    switch (msg.type) {
      case SCENE_INFO:
        if (onSceneInfo) onSceneInfo(msg)
        break
      case STATE_UPDATE:
        if (onStateUpdate) onStateUpdate(msg)
        break
    }
  }

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg))
    }
  }

  function setHandlers({ sceneInfo, stateUpdate, reconnect }) {
    onSceneInfo = sceneInfo || null
    onStateUpdate = stateUpdate || null
    onReconnect = reconnect || null
  }

  /**
   * Dispose the current frame and invalidate any in-flight bitmap decodes.
   * Called by App.vue during reconnect cleanup.
   */
  function resetFrame() {
    frameGeneration++
    const old = latestFrame.value
    if (old && typeof old.close === 'function') {
      old.close()
    }
    latestFrame.value = null
  }

  function destroy() {
    frameGeneration++
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.onclose = null
      ws.close()
      ws = null
    }
    const old = latestFrame.value
    if (old && typeof old.close === 'function') {
      old.close()
    }
    latestFrame.value = null
  }

  onUnmounted(destroy)

  return {
    connected,
    latestFrame,
    connect,
    send,
    setHandlers,
    resetFrame,
    destroy,
  }
}
