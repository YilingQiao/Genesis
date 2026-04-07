/**
 * Pure math functions for gizmo rendering and interaction.
 * Extracted from the original index.html gizmo implementation.
 */

// ---- Vector3 Operations ----

export function vec3Sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vec3Add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function vec3Scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s]
}

export function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function vec3Len(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

export function vec3Norm(v) {
  const l = vec3Len(v)
  return l > 1e-8 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0]
}

// ---- Matrix Operations ----

/**
 * Build a view matrix from eye, target, and up vectors.
 * Returns a column-major Float64Array (OpenGL convention).
 */
export function buildViewMatrix(eye, target, up) {
  const f = vec3Norm(vec3Sub(target, eye))
  const s = vec3Norm(vec3Cross(f, up))
  const u = vec3Cross(s, f)
  return new Float64Array([
    s[0], u[0], -f[0], 0,
    s[1], u[1], -f[1], 0,
    s[2], u[2], -f[2], 0,
    -vec3Dot(s, eye), -vec3Dot(u, eye), vec3Dot(f, eye), 1,
  ])
}

/**
 * Build a perspective projection matrix.
 * Returns a column-major Float64Array.
 */
export function buildProjMatrix(fovRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovRad / 2.0)
  const nf = 1.0 / (near - far)
  return new Float64Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ])
}

/**
 * Project a 3D world point to 2D screen coordinates.
 * Returns [screenX, screenY] or null if behind camera.
 */
export function project3D(wp, vm, pm, w, h) {
  const vx = vm[0] * wp[0] + vm[4] * wp[1] + vm[8] * wp[2] + vm[12]
  const vy = vm[1] * wp[0] + vm[5] * wp[1] + vm[9] * wp[2] + vm[13]
  const vz = vm[2] * wp[0] + vm[6] * wp[1] + vm[10] * wp[2] + vm[14]
  const vw = vm[3] * wp[0] + vm[7] * wp[1] + vm[11] * wp[2] + vm[15]
  const cx = pm[0] * vx + pm[4] * vy + pm[8] * vz + pm[12] * vw
  const cy = pm[1] * vx + pm[5] * vy + pm[9] * vz + pm[13] * vw
  const cw = pm[3] * vx + pm[7] * vy + pm[11] * vz + pm[15] * vw
  if (cw <= 0.001) return null
  return [(cx / cw * 0.5 + 0.5) * w, (1.0 - (cy / cw * 0.5 + 0.5)) * h]
}

/**
 * Distance from a point to a line segment (2D).
 */
export function ptSegDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax
  const dy = by - ay
  const l2 = dx * dx + dy * dy
  if (l2 < 1e-8) return Math.hypot(px - ax, py - ay)
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

// ---- Quaternion Operations ----

/**
 * Create a quaternion from an axis-angle rotation.
 */
export function axisAngleToQuat(axis, angle) {
  const ha = angle / 2
  const s = Math.sin(ha)
  return { w: Math.cos(ha), x: axis[0] * s, y: axis[1] * s, z: axis[2] * s }
}

/**
 * Multiply two quaternions: result = a * b.
 */
export function quatMul(a, b) {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  }
}

/**
 * Normalize a quaternion to unit length.
 */
export function quatNormalize(q) {
  const n = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
  if (n < 1e-8) return { w: 1, x: 0, y: 0, z: 0 }
  return { w: q.w / n, x: q.x / n, y: q.y / n, z: q.z / n }
}

// ---- Euler / Quaternion Conversion ----

/**
 * Convert Euler angles (degrees) to quaternion (w, x, y, z).
 * Uses ZYX intrinsic rotation order.
 */
export function eulerToQuat(rollDeg, pitchDeg, yawDeg) {
  const r = (rollDeg * Math.PI) / 180
  const p = (pitchDeg * Math.PI) / 180
  const y = (yawDeg * Math.PI) / 180
  const cr = Math.cos(r / 2), sr = Math.sin(r / 2)
  const cp = Math.cos(p / 2), sp = Math.sin(p / 2)
  const cy = Math.cos(y / 2), sy = Math.sin(y / 2)
  return {
    w: cr * cp * cy + sr * sp * sy,
    x: sr * cp * cy - cr * sp * sy,
    y: cr * sp * cy + sr * cp * sy,
    z: cr * cp * sy - sr * sp * cy,
  }
}

/**
 * Convert quaternion (w, x, y, z) to Euler angles (degrees).
 * Returns { roll, pitch, yaw }.
 */
export function quatToEuler(w, x, y, z) {
  const sinr = 2 * (w * x + y * z)
  const cosr = 1 - 2 * (x * x + y * y)
  const roll = Math.atan2(sinr, cosr)
  const sinp = 2 * (w * y - z * x)
  const pitch = Math.abs(sinp) >= 1
    ? Math.sign(sinp) * Math.PI / 2
    : Math.asin(sinp)
  const siny = 2 * (w * z + x * y)
  const cosy = 1 - 2 * (y * y + z * z)
  const yaw = Math.atan2(siny, cosy)
  return {
    roll: (roll * 180) / Math.PI,
    pitch: (pitch * 180) / Math.PI,
    yaw: (yaw * 180) / Math.PI,
  }
}

// ---- Gizmo Constants ----

export const GIZMO_SIZE_FACTOR = 0.15
export const GIZMO_HIT_RADIUS = 14
export const GIZMO_AXES = [
  { key: 'x', dir: [1, 0, 0], color: '#e74c3c' },
  { key: 'y', dir: [0, 1, 0], color: '#27ae60' },
  { key: 'z', dir: [0, 0, 1], color: '#2980b9' },
]
