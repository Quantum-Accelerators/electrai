import { useRef } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { Quaternion, Spherical, Vector3 } from 'three'
import type { RefObject, MutableRefObject } from 'react'

const ORBIT_SPEED = 1.5  // rad/s
const ZOOM_SPEED = 3.0   // multiplier/s
const PAN_SPEED = 5.0    // units/s
const ROLL_SPEED = 1.0   // rad/s

// Reusable temp objects to avoid GC pressure
const _offset = new Vector3()
const _spherical = new Spherical()
const _forward = new Vector3()
const _right = new Vector3()
const _up = new Vector3()
const _panVec = new Vector3()
const _viewDir = new Vector3()

export interface CameraSnapTarget {
  /** Direction vector (camera offset from orbit target, will be normalized and scaled to current distance) */
  direction: [number, number, number]
}

interface CameraControllerProps {
  activeMovements: RefObject<Set<string>>
  cameraSnap?: MutableRefObject<CameraSnapTarget | null>
  animationDuration?: number
}

interface SnapState {
  startQuat: Quaternion
  endQuat: Quaternion
  radius: number
  elapsed: number
  duration: number
}

const _snapQ = new Quaternion()
const _rotQ = new Quaternion()
const _axis = new Vector3()

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

export function CameraController({ activeMovements, cameraSnap, animationDuration = 0.5 }: CameraControllerProps) {
  const camera = useThree(s => s.camera)
  const controls = useThree(s => s.controls) as { target: Vector3; update: () => void } | null
  const snapState = useRef<SnapState | null>(null)

  useFrame((_, delta) => {
    if (!controls) return
    const target = controls.target

    // Handle snap animation
    if (cameraSnap?.current) {
      const snap = cameraSnap.current
      cameraSnap.current = null
      const radius = camera.position.distanceTo(target)

      // Current and target offset directions (camera position relative to orbit target)
      const currentDir = _offset.copy(camera.position).sub(target).normalize()
      const targetDir = new Vector3(...snap.direction).normalize()
      const dot = currentDir.dot(targetDir)

      if (dot > 0.9999) return // already aligned, skip

      // Capture start orientation from current camera
      const startQuat = camera.quaternion.clone()

      // Minimal rotation: rotate current viewing frame to align with target direction
      if (dot < -0.9999) {
        // Antiparallel: rotate 180° around camera up
        _axis.copy(camera.up).normalize()
        _rotQ.setFromAxisAngle(_axis, Math.PI)
      } else {
        // Shortest arc rotation from currentDir to targetDir
        _axis.crossVectors(currentDir, targetDir).normalize()
        _rotQ.setFromAxisAngle(_axis, Math.acos(Math.max(-1, Math.min(1, dot))))
      }

      // Apply rotation to current orientation: endQuat = rotQ * startQuat
      const endQuat = new Quaternion().multiplyQuaternions(_rotQ, startQuat)

      snapState.current = { startQuat, endQuat, radius, elapsed: 0, duration: animationDuration }
    }

    if (snapState.current) {
      const s = snapState.current
      s.elapsed += delta
      const t = Math.min(1, s.elapsed / s.duration)
      const e = easeInOutCubic(t)

      // Single quaternion slerp for the entire camera orientation
      _snapQ.copy(s.startQuat).slerp(s.endQuat, e)

      // Derive position: camera looks down local -Z, so offset is local +Z * radius
      _offset.set(0, 0, 1).applyQuaternion(_snapQ).multiplyScalar(s.radius)
      camera.position.copy(target).add(_offset)

      // Derive up: local +Y
      camera.up.set(0, 1, 0).applyQuaternion(_snapQ)
      camera.lookAt(target)
      controls.update()

      if (t >= 1) snapState.current = null
      return
    }

    const movements = activeMovements.current
    if (!movements?.size) return

    const dt = Math.min(delta, 0.05)

    // Orbit + Zoom: work in spherical coords
    _offset.copy(camera.position).sub(target)
    _spherical.setFromVector3(_offset)

    let posChanged = false

    if (movements.has('orbit-left'))  { _spherical.theta -= ORBIT_SPEED * dt; posChanged = true }
    if (movements.has('orbit-right')) { _spherical.theta += ORBIT_SPEED * dt; posChanged = true }
    if (movements.has('orbit-up'))    { _spherical.phi = Math.max(0.01, _spherical.phi - ORBIT_SPEED * dt); posChanged = true }
    if (movements.has('orbit-down'))  { _spherical.phi = Math.min(Math.PI - 0.01, _spherical.phi + ORBIT_SPEED * dt); posChanged = true }

    if (movements.has('zoom-in'))  { _spherical.radius = Math.max(0.5, _spherical.radius * (1 - ZOOM_SPEED * dt)); posChanged = true }
    if (movements.has('zoom-out')) { _spherical.radius *= (1 + ZOOM_SPEED * dt); posChanged = true }

    if (posChanged) {
      _offset.setFromSpherical(_spherical)
      camera.position.copy(target).add(_offset)
      camera.lookAt(target)
    }

    // Pan: move both camera and target in screen-relative direction
    if (movements.has('pan-left') || movements.has('pan-right') ||
        movements.has('pan-up') || movements.has('pan-down')) {
      camera.getWorldDirection(_forward)
      _right.crossVectors(_forward, camera.up).normalize()
      _up.crossVectors(_right, _forward).normalize()

      const panDist = PAN_SPEED * dt
      _panVec.set(0, 0, 0)
      if (movements.has('pan-left'))  _panVec.addScaledVector(_right, -panDist)
      if (movements.has('pan-right')) _panVec.addScaledVector(_right, panDist)
      if (movements.has('pan-up'))    _panVec.addScaledVector(_up, panDist)
      if (movements.has('pan-down'))  _panVec.addScaledVector(_up, -panDist)

      camera.position.add(_panVec)
      target.add(_panVec)
    }

    // Roll: rotate camera up-vector around viewing axis
    if (movements.has('roll-cw') || movements.has('roll-ccw')) {
      camera.getWorldDirection(_viewDir)
      const angle = (movements.has('roll-cw') ? -1 : 1) * ROLL_SPEED * dt
      camera.up.applyAxisAngle(_viewDir, angle)
      camera.lookAt(target)
    }

    controls.update()
  })

  return null
}
