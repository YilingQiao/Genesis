import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// We need to test useWebSocket without a real DOM or WebSocket.
// The key behavior to test is the frame generation token logic.
// We'll extract and test the generation token pattern directly.

describe('useWebSocket: frame generation token pattern', () => {
  // Simulate the generation token logic from handleBinaryFrame
  it('stale bitmap decode is discarded when generation changes', async () => {
    let frameGeneration = 0
    let latestFrame = null
    const closedBitmaps = []

    // Simulate handleBinaryFrame with captured generation
    async function handleBinaryFrame(decodedBitmap) {
      const gen = frameGeneration
      // Simulate async createImageBitmap resolving
      await Promise.resolve()
      if (gen !== frameGeneration) {
        closedBitmaps.push(decodedBitmap)
        return
      }
      const old = latestFrame
      latestFrame = decodedBitmap
      if (old) closedBitmaps.push(old)
    }

    // Start decoding a frame
    const staleDecodePromise = handleBinaryFrame({ id: 'stale-bitmap' })

    // Simulate reconnect: bump generation (like resetFrame does)
    frameGeneration++
    latestFrame = null

    // Let the stale decode resolve
    await staleDecodePromise

    // The stale bitmap should have been closed, NOT assigned
    expect(latestFrame).toBeNull()
    expect(closedBitmaps).toContainEqual({ id: 'stale-bitmap' })
  })

  it('current-generation bitmap is assigned normally', async () => {
    let frameGeneration = 0
    let latestFrame = null
    const closedBitmaps = []

    async function handleBinaryFrame(decodedBitmap) {
      const gen = frameGeneration
      await Promise.resolve()
      if (gen !== frameGeneration) {
        closedBitmaps.push(decodedBitmap)
        return
      }
      const old = latestFrame
      latestFrame = decodedBitmap
      if (old) closedBitmaps.push(old)
    }

    await handleBinaryFrame({ id: 'frame-1' })
    expect(latestFrame).toEqual({ id: 'frame-1' })

    await handleBinaryFrame({ id: 'frame-2' })
    expect(latestFrame).toEqual({ id: 'frame-2' })
    // frame-1 should have been closed when replaced
    expect(closedBitmaps).toContainEqual({ id: 'frame-1' })
  })

  it('multiple stale decodes are all discarded', async () => {
    let frameGeneration = 0
    let latestFrame = null
    const closedBitmaps = []

    async function handleBinaryFrame(decodedBitmap) {
      const gen = frameGeneration
      await Promise.resolve()
      if (gen !== frameGeneration) {
        closedBitmaps.push(decodedBitmap)
        return
      }
      const old = latestFrame
      latestFrame = decodedBitmap
      if (old) closedBitmaps.push(old)
    }

    // Start 3 decodes
    const p1 = handleBinaryFrame({ id: 'stale-1' })
    const p2 = handleBinaryFrame({ id: 'stale-2' })
    const p3 = handleBinaryFrame({ id: 'stale-3' })

    // Reconnect
    frameGeneration++
    latestFrame = null

    // Let all resolve
    await Promise.all([p1, p2, p3])

    expect(latestFrame).toBeNull()
    expect(closedBitmaps.length).toBe(3)
  })

  it('resetFrame increments generation and disposes current frame', () => {
    // Simulate resetFrame behavior
    let frameGeneration = 0
    let latestFrame = { close: vi.fn() }

    function resetFrame() {
      frameGeneration++
      if (latestFrame && typeof latestFrame.close === 'function') {
        latestFrame.close()
      }
      latestFrame = null
    }

    const oldFrame = latestFrame
    resetFrame()

    expect(frameGeneration).toBe(1)
    expect(latestFrame).toBeNull()
    expect(oldFrame.close).toHaveBeenCalledOnce()
  })
})
