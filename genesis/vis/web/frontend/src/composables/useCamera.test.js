import { describe, it, expect } from 'vitest'
import { useCamera } from './useCamera.js'

function createCamera() {
  const sent = []
  const send = (msg) => sent.push(msg)
  const camera = useCamera(send)
  return { camera, sent }
}

describe('useCamera: finite-number guards', () => {
  it('setPose suppresses NaN in pos', () => {
    const { camera, sent } = createCamera()
    camera.setPose([NaN, 0, 0], null)
    expect(sent.length).toBe(0)
  })

  it('setPose suppresses Infinity in lookat', () => {
    const { camera, sent } = createCamera()
    camera.setPose(null, [0, Infinity, 0])
    expect(sent.length).toBe(0)
  })

  it('setPose sends valid pos', () => {
    const { camera, sent } = createCamera()
    camera.setPose([1, 2, 3], null)
    expect(sent.length).toBe(1)
    expect(sent[0].pos).toEqual([1, 2, 3])
  })

  it('setFov suppresses NaN', () => {
    const { camera, sent } = createCamera()
    camera.setFov(NaN)
    expect(sent.length).toBe(0)
  })

  it('setFov sends valid values', () => {
    const { camera, sent } = createCamera()
    camera.setFov(45)
    expect(sent.length).toBe(1)
    expect(sent[0].fov).toBe(45)
  })
})

describe('useCamera: focus guard', () => {
  it('skips server update when inputs are focused', () => {
    const { camera } = createCamera()
    camera.setCameraState({ pos: [1, 2, 3] })
    expect(camera.pos.value).toEqual([1, 2, 3])

    // Mark a field as focused
    camera.markFocused('pos-0')
    camera.handleCameraState({ pos: [9, 9, 9] })
    // Should not have updated
    expect(camera.pos.value).toEqual([1, 2, 3])

    // Blur and try again
    camera.markBlurred('pos-0')
    camera.handleCameraState({ pos: [9, 9, 9] })
    expect(camera.pos.value).toEqual([9, 9, 9])
  })
})
