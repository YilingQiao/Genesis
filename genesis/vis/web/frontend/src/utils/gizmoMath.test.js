import { describe, it, expect } from 'vitest'
import {
  vec3Sub, vec3Add, vec3Scale, vec3Dot, vec3Cross, vec3Len, vec3Norm,
  buildViewMatrix, buildProjMatrix, project3D, ptSegDist,
  axisAngleToQuat, quatMul, quatNormalize,
  eulerToQuat, quatToEuler,
} from './gizmoMath.js'

describe('vec3 operations', () => {
  it('vec3Sub subtracts two vectors', () => {
    expect(vec3Sub([3, 2, 1], [1, 1, 1])).toEqual([2, 1, 0])
  })

  it('vec3Add adds two vectors', () => {
    expect(vec3Add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9])
  })

  it('vec3Scale scales a vector', () => {
    expect(vec3Scale([1, 2, 3], 2)).toEqual([2, 4, 6])
  })

  it('vec3Dot computes dot product', () => {
    expect(vec3Dot([1, 0, 0], [0, 1, 0])).toBe(0)
    expect(vec3Dot([1, 2, 3], [4, 5, 6])).toBe(32)
  })

  it('vec3Cross computes cross product', () => {
    expect(vec3Cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1])
    expect(vec3Cross([0, 1, 0], [1, 0, 0])).toEqual([0, 0, -1])
  })

  it('vec3Len computes length', () => {
    expect(vec3Len([3, 4, 0])).toBe(5)
    expect(vec3Len([0, 0, 0])).toBe(0)
  })

  it('vec3Norm normalizes a vector', () => {
    const n = vec3Norm([3, 0, 0])
    expect(n[0]).toBeCloseTo(1)
    expect(n[1]).toBeCloseTo(0)
    expect(n[2]).toBeCloseTo(0)
  })

  it('vec3Norm returns zero for zero vector', () => {
    expect(vec3Norm([0, 0, 0])).toEqual([0, 0, 0])
  })
})

describe('matrix operations', () => {
  it('buildViewMatrix returns Float64Array of length 16', () => {
    const vm = buildViewMatrix([0, 0, 5], [0, 0, 0], [0, 1, 0])
    expect(vm).toBeInstanceOf(Float64Array)
    expect(vm.length).toBe(16)
  })

  it('buildProjMatrix returns Float64Array of length 16', () => {
    const pm = buildProjMatrix(Math.PI / 4, 1.5, 0.1, 100)
    expect(pm).toBeInstanceOf(Float64Array)
    expect(pm.length).toBe(16)
  })

  it('project3D returns screen coords for visible point', () => {
    const vm = buildViewMatrix([0, -5, 0], [0, 0, 0], [0, 0, 1])
    const pm = buildProjMatrix(Math.PI / 3, 1, 0.1, 100)
    const result = project3D([0, 0, 0], vm, pm, 800, 600)
    expect(result).not.toBeNull()
    expect(result[0]).toBeCloseTo(400, 0)
    expect(result[1]).toBeCloseTo(300, 0)
  })

  it('project3D returns null for point behind camera', () => {
    const vm = buildViewMatrix([0, -5, 0], [0, 0, 0], [0, 0, 1])
    const pm = buildProjMatrix(Math.PI / 3, 1, 0.1, 100)
    // Point far behind camera
    const result = project3D([0, -10, 0], vm, pm, 800, 600)
    expect(result).toBeNull()
  })
})

describe('ptSegDist', () => {
  it('computes distance from point to segment', () => {
    // Point on the segment
    expect(ptSegDist(5, 0, 0, 0, 10, 0)).toBeCloseTo(0)
    // Point perpendicular to segment midpoint
    expect(ptSegDist(5, 3, 0, 0, 10, 0)).toBeCloseTo(3)
    // Point closest to endpoint
    expect(ptSegDist(12, 0, 0, 0, 10, 0)).toBeCloseTo(2)
  })

  it('handles degenerate segment (zero length)', () => {
    expect(ptSegDist(3, 4, 0, 0, 0, 0)).toBeCloseTo(5)
  })
})

describe('quaternion operations', () => {
  it('axisAngleToQuat for zero angle returns identity', () => {
    const q = axisAngleToQuat([0, 0, 1], 0)
    expect(q.w).toBeCloseTo(1)
    expect(q.x).toBeCloseTo(0)
    expect(q.y).toBeCloseTo(0)
    expect(q.z).toBeCloseTo(0)
  })

  it('axisAngleToQuat for 90 degrees around Z', () => {
    const q = axisAngleToQuat([0, 0, 1], Math.PI / 2)
    expect(q.w).toBeCloseTo(Math.cos(Math.PI / 4))
    expect(q.z).toBeCloseTo(Math.sin(Math.PI / 4))
  })

  it('quatMul identity * q = q', () => {
    const id = { w: 1, x: 0, y: 0, z: 0 }
    const q = { w: 0.5, x: 0.5, y: 0.5, z: 0.5 }
    const result = quatMul(id, q)
    expect(result.w).toBeCloseTo(q.w)
    expect(result.x).toBeCloseTo(q.x)
    expect(result.y).toBeCloseTo(q.y)
    expect(result.z).toBeCloseTo(q.z)
  })

  it('quatNormalize produces unit quaternion', () => {
    const q = quatNormalize({ w: 2, x: 0, y: 0, z: 0 })
    const len = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    expect(len).toBeCloseTo(1)
  })

  it('quatNormalize handles near-zero quaternion', () => {
    const q = quatNormalize({ w: 0, x: 0, y: 0, z: 0 })
    expect(q.w).toBe(1)
  })
})

describe('euler/quaternion conversion', () => {
  it('eulerToQuat and quatToEuler round-trip', () => {
    const roll = 30, pitch = 45, yaw = 60
    const q = eulerToQuat(roll, pitch, yaw)
    const euler = quatToEuler(q.w, q.x, q.y, q.z)
    expect(euler.roll).toBeCloseTo(roll, 1)
    expect(euler.pitch).toBeCloseTo(pitch, 1)
    expect(euler.yaw).toBeCloseTo(yaw, 1)
  })

  it('identity quaternion gives zero euler angles', () => {
    const euler = quatToEuler(1, 0, 0, 0)
    expect(euler.roll).toBeCloseTo(0)
    expect(euler.pitch).toBeCloseTo(0)
    expect(euler.yaw).toBeCloseTo(0)
  })

  it('euler resync: quat edit followed by euler read gives consistent values', () => {
    // Simulate: user edits quaternion, then switches to euler mode
    // The euler values should reflect the new quaternion
    const roll = 45, pitch = 30, yaw = 90
    const q = eulerToQuat(roll, pitch, yaw)
    // Now convert back (simulating resync)
    const euler = quatToEuler(q.w, q.x, q.y, q.z)
    expect(euler.roll).toBeCloseTo(roll, 1)
    expect(euler.pitch).toBeCloseTo(pitch, 1)
    expect(euler.yaw).toBeCloseTo(yaw, 1)
  })

  it('euler resync after gizmo rotation gives finite values', () => {
    // Simulate a gizmo rotate: axisAngleToQuat around Z, then quatToEuler
    const rotQ = axisAngleToQuat([0, 0, 1], Math.PI / 4)
    const baseQ = { w: 1, x: 0, y: 0, z: 0 }
    const newQ = quatNormalize(quatMul(rotQ, baseQ))
    const euler = quatToEuler(newQ.w, newQ.x, newQ.y, newQ.z)
    expect(Number.isFinite(euler.roll)).toBe(true)
    expect(Number.isFinite(euler.pitch)).toBe(true)
    expect(Number.isFinite(euler.yaw)).toBe(true)
    // 45 degrees around Z should give ~45 yaw
    expect(euler.yaw).toBeCloseTo(45, 0)
  })
})

describe('invalid number guard patterns', () => {
  it('NaN is not finite', () => {
    expect(Number.isFinite(NaN)).toBe(false)
  })

  it('Infinity is not finite', () => {
    expect(Number.isFinite(Infinity)).toBe(false)
    expect(Number.isFinite(-Infinity)).toBe(false)
  })

  it('parseFloat of empty string is NaN', () => {
    expect(Number.isFinite(parseFloat(''))).toBe(false)
  })

  it('parseFloat of valid number is finite', () => {
    expect(Number.isFinite(parseFloat('3.14'))).toBe(true)
    expect(Number.isFinite(parseFloat('-0.5'))).toBe(true)
  })
})
