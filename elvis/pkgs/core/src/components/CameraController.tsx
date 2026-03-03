import { useRef, useEffect } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { MathUtils, Quaternion, Spherical, Vector3 } from 'three'
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
const _upTarget = new Vector3()
const _upCurrent = new Vector3()

// Wheel handler temp vectors (separate to avoid conflicts with useFrame)
const _wFwd = new Vector3()
const _wRight = new Vector3()
const _wUp = new Vector3()
const _wPan = new Vector3()

export type CameraSnapTarget =
  | { type: 'look-down', direction: [number, number, number] }
  | { type: 'align-up', axis: [number, number, number] }
  | { type: 'orbit-step', direction: 'left' | 'right' | 'up' | 'down', degrees: number }
  | { type: 'roll-step', direction: 'cw' | 'ccw', degrees: number }
  | { type: 'zoom-step', direction: 'in' | 'out', factor: number }
  | { type: 'pan-step', direction: 'left' | 'right' | 'up' | 'down', distance: number }
  | { type: 'reset-pan' }

interface CameraControllerProps {
  activeMovements: RefObject<Set<string>>
  cameraSnap?: MutableRefObject<CameraSnapTarget | null>
  animationDuration?: number
  onCameraChange?: (theta: number, phi: number, zoom: number, roll: number, targetOffset?: [number, number, number]) => void
  initialCamera?: MutableRefObject<[number, number, number, number] | null>
  initialTargetOffset?: MutableRefObject<[number, number, number] | null>
  center?: Vector3
}

interface SnapState {
  startQuat: Quaternion
  endQuat: Quaternion
  startRadius: number
  endRadius: number
  startTarget?: Vector3
  endTarget?: Vector3
  elapsed: number
  duration: number
}

const _snapQ = new Quaternion()
const _rotQ = new Quaternion()
const _axis = new Vector3()
const _snapTarget = new Vector3()

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

export function CameraController({
  activeMovements,
  cameraSnap,
  animationDuration = 0.5,
  onCameraChange,
  initialCamera,
  initialTargetOffset,
  center,
}: CameraControllerProps) {
  const camera = useThree(s => s.camera)
  const controls = useThree(s => s.controls) as { target: Vector3; update: () => void } | null
  const snapState = useRef<SnapState | null>(null)
  const lastStepRef = useRef<CameraSnapTarget | null>(null)
  const initFramesRef = useRef(0)
  const camSyncRef = useRef<{ lastChange: number; reported: boolean }>({ lastChange: 0, reported: true })
  const prevCamRef = useRef<[number, number, number, number, number, number] | null>(null)
  const gl = useThree(s => s.gl)

  // Modifier+wheel: Shift=pan vertical, Ctrl/Cmd+Shift=pan horizontal, Alt=roll
  useEffect(() => {
    const canvas = gl.domElement
    const handler = (e: WheelEvent) => {
      if (!controls) return
      if (!e.shiftKey && !e.altKey) return
      e.preventDefault()
      e.stopPropagation()

      const target = controls.target
      const delta = e.deltaY * 0.01

      camera.getWorldDirection(_wFwd)
      _wRight.crossVectors(_wFwd, camera.up).normalize()
      _wUp.crossVectors(_wRight, _wFwd).normalize()

      if (e.altKey && !e.shiftKey) {
        // Alt+wheel → roll
        camera.up.applyAxisAngle(_wFwd, delta * 0.003)
        camera.lookAt(target)
      } else if (e.shiftKey && (e.ctrlKey || e.metaKey)) {
        // Ctrl/Cmd+Shift+wheel → pan left/right
        _wPan.copy(_wRight).multiplyScalar(delta)
        camera.position.add(_wPan)
        target.add(_wPan)
      } else if (e.shiftKey) {
        // Shift+wheel → pan up/down
        _wPan.copy(_wUp).multiplyScalar(-delta)
        camera.position.add(_wPan)
        target.add(_wPan)
      }

      controls.update()
    }

    canvas.addEventListener('wheel', handler, { passive: false, capture: true })
    return () => canvas.removeEventListener('wheel', handler, { capture: true })
  }, [gl, camera, controls])

  useFrame((_, delta) => {
    if (!controls) return
    const target = controls.target

    // Initial camera setup from URL — re-apply for several frames to
    // override OrbitControls which may reset orientation on early frames
    if (initialCamera?.current && initFramesRef.current < 5) {
      const [theta, phi, zoom, roll] = initialCamera.current
      // Apply target offset (pan) if provided
      if (initialTargetOffset?.current && center) {
        const [dx, dy, dz] = initialTargetOffset.current
        target.set(center.x + dx, center.y + dy, center.z + dz)
      }
      _spherical.set(zoom, MathUtils.degToRad(phi), MathUtils.degToRad(theta))
      _offset.setFromSpherical(_spherical)
      camera.position.copy(target).add(_offset)
      camera.up.set(0, 1, 0)
      camera.lookAt(target)
      if (Math.abs(roll) > 0.05) {
        camera.getWorldDirection(_viewDir)
        camera.up.applyAxisAngle(_viewDir, MathUtils.degToRad(roll))
        camera.lookAt(target)
      }
      controls.update()
      initFramesRef.current++
      if (initFramesRef.current >= 5) {
        initialCamera.current = null
        if (initialTargetOffset?.current) initialTargetOffset.current = null
      }
      return
    }

    // Handle snap trigger
    if (cameraSnap?.current) {
      const snap = cameraSnap.current
      cameraSnap.current = null
      const radius = camera.position.distanceTo(target)
      let endRadius = radius
      const startQuat = camera.quaternion.clone()
      let endQuat: Quaternion | null = null
      let startTarget: Vector3 | undefined
      let endTarget: Vector3 | undefined

      if (snap.type === 'look-down') {
        const currentDir = _offset.copy(camera.position).sub(target).normalize()
        const targetDir = new Vector3(...snap.direction).normalize()
        const dot = currentDir.dot(targetDir)
        if (dot <= 0.9999) {
          if (dot < -0.9999) {
            _axis.copy(camera.up).normalize()
            _rotQ.setFromAxisAngle(_axis, Math.PI)
          } else {
            _axis.crossVectors(currentDir, targetDir).normalize()
            _rotQ.setFromAxisAngle(_axis, Math.acos(Math.max(-1, Math.min(1, dot))))
          }
          endQuat = new Quaternion().multiplyQuaternions(_rotQ, startQuat)
        }
      } else if (snap.type === 'align-up') {
        camera.getWorldDirection(_viewDir)
        _upTarget.set(...snap.axis)
        const dotD = _upTarget.dot(_viewDir)
        _upTarget.addScaledVector(_viewDir, -dotD)
        if (_upTarget.lengthSq() >= 0.0001) {
          _upTarget.normalize()
          _upCurrent.copy(camera.up)
          const dotUp = _upCurrent.dot(_viewDir)
          _upCurrent.addScaledVector(_viewDir, -dotUp).normalize()
          _axis.crossVectors(_upCurrent, _upTarget)
          const cosA = Math.max(-1, Math.min(1, _upCurrent.dot(_upTarget)))
          const sinA = _axis.dot(_viewDir)
          const angle = Math.atan2(sinA, cosA)
          if (Math.abs(angle) >= 0.001) {
            _rotQ.setFromAxisAngle(_viewDir, angle)
            endQuat = new Quaternion().multiplyQuaternions(_rotQ, startQuat)
          }
        }
      } else if (snap.type === 'orbit-step') {
        camera.getWorldDirection(_viewDir)
        _right.crossVectors(_viewDir, camera.up).normalize()
        _up.crossVectors(_right, _viewDir).normalize()
        const stepRad = snap.degrees * Math.PI / 180
        if (snap.direction === 'left') _rotQ.setFromAxisAngle(_up, -stepRad)
        else if (snap.direction === 'right') _rotQ.setFromAxisAngle(_up, stepRad)
        else if (snap.direction === 'up') _rotQ.setFromAxisAngle(_right, -stepRad)
        else _rotQ.setFromAxisAngle(_right, stepRad)
        endQuat = new Quaternion().multiplyQuaternions(_rotQ, startQuat)
      } else if (snap.type === 'roll-step') {
        camera.getWorldDirection(_viewDir)
        const stepRad = snap.degrees * Math.PI / 180
        const sign = snap.direction === 'cw' ? -1 : 1
        _rotQ.setFromAxisAngle(_viewDir, sign * stepRad)
        endQuat = new Quaternion().multiplyQuaternions(_rotQ, startQuat)
      } else if (snap.type === 'zoom-step') {
        endQuat = startQuat.clone()
        endRadius = snap.direction === 'in' ? radius / snap.factor : radius * snap.factor
        if (snap.direction === 'in') endRadius = Math.max(0.5, endRadius)
      } else if (snap.type === 'pan-step') {
        endQuat = startQuat.clone()
        camera.getWorldDirection(_viewDir)
        _right.crossVectors(_viewDir, camera.up).normalize()
        _up.crossVectors(_right, _viewDir).normalize()
        const d = snap.distance
        const startT = target.clone()
        const endT = startT.clone()
        if (snap.direction === 'left')  endT.addScaledVector(_right, -d)
        else if (snap.direction === 'right') endT.addScaledVector(_right, d)
        else if (snap.direction === 'up')    endT.addScaledVector(_up, d)
        else endT.addScaledVector(_up, -d)
        startTarget = startT
        endTarget = endT
      } else if (snap.type === 'reset-pan' && center) {
        endQuat = startQuat.clone()
        startTarget = target.clone()
        endTarget = center.clone()
      }

      // Remember step snaps for key-hold chaining
      const isStep = snap.type === 'orbit-step' || snap.type === 'roll-step' || snap.type === 'zoom-step' || snap.type === 'pan-step'
      lastStepRef.current = isStep ? snap : null

      if (endQuat) {
        snapState.current = { startQuat, endQuat, startRadius: radius, endRadius, elapsed: 0, duration: animationDuration, startTarget, endTarget }
      }
    }

    // Animate snap
    let isSnapping = false
    let snapJustFinished = false
    if (snapState.current) {
      const s = snapState.current
      s.elapsed += delta
      const t = Math.min(1, s.elapsed / s.duration)
      const e = easeInOutCubic(t)

      _snapQ.copy(s.startQuat).slerp(s.endQuat, e)
      const r = s.startRadius + (s.endRadius - s.startRadius) * e

      // Interpolate target if pan-step
      if (s.startTarget && s.endTarget) {
        _snapTarget.lerpVectors(s.startTarget, s.endTarget, e)
        target.copy(_snapTarget)
      }

      _offset.set(0, 0, 1).applyQuaternion(_snapQ).multiplyScalar(r)
      camera.position.copy(target).add(_offset)
      camera.up.set(0, 1, 0).applyQuaternion(_snapQ)
      camera.lookAt(target)

      controls.update()

      if (t >= 1) {
        // Re-apply exact end state after controls.update()
        if (s.startTarget && s.endTarget) target.copy(s.endTarget)
        _offset.set(0, 0, 1).applyQuaternion(s.endQuat).multiplyScalar(s.endRadius)
        camera.position.copy(target).add(_offset)
        camera.up.set(0, 1, 0).applyQuaternion(s.endQuat)
        camera.lookAt(target)

        snapState.current = null
        snapJustFinished = true
        // Chain step if key is still held
        const last = lastStepRef.current
        if (last && activeMovements.current) {
          let key: string | null = null
          if (last.type === 'orbit-step') key = `orbit-${last.direction}`
          else if (last.type === 'roll-step') key = `roll-${last.direction}`
          else if (last.type === 'zoom-step') key = `zoom-${last.direction}`
          else if (last.type === 'pan-step') key = `pan-${last.direction}`
          if (key && activeMovements.current.has(key)) {
            cameraSnap!.current = last
          }
        }
      }
      isSnapping = true
    }

    // Movement processing (only when not snapping)
    if (!isSnapping) {
      const movements = activeMovements.current
      if (movements?.size) {
        const dt = Math.min(delta, 0.05)
        // Skip continuous movement if a discrete step chain is pending
        const pendingSnap = cameraSnap?.current?.type
        const discreteOrbit = pendingSnap === 'orbit-step'
        const discreteRoll = pendingSnap === 'roll-step'
        const discreteZoom = pendingSnap === 'zoom-step'
        const discretePan = pendingSnap === 'pan-step'

        // Screen-relative orbit
        _offset.copy(camera.position).sub(target)
        camera.getWorldDirection(_viewDir)
        _right.crossVectors(_viewDir, camera.up).normalize()
        _up.crossVectors(_right, _viewDir).normalize()

        let posChanged = false

        if (!discreteOrbit) {
          if (movements.has('orbit-left'))  { _offset.applyAxisAngle(_up, -ORBIT_SPEED * dt); camera.up.applyAxisAngle(_up, -ORBIT_SPEED * dt); posChanged = true }
          if (movements.has('orbit-right')) { _offset.applyAxisAngle(_up,  ORBIT_SPEED * dt); camera.up.applyAxisAngle(_up,  ORBIT_SPEED * dt); posChanged = true }
          if (movements.has('orbit-up'))    { _offset.applyAxisAngle(_right, -ORBIT_SPEED * dt); camera.up.applyAxisAngle(_right, -ORBIT_SPEED * dt); posChanged = true }
          if (movements.has('orbit-down'))  { _offset.applyAxisAngle(_right,  ORBIT_SPEED * dt); camera.up.applyAxisAngle(_right,  ORBIT_SPEED * dt); posChanged = true }
        }

        if (!discreteZoom) {
          if (movements.has('zoom-in')) {
            if (_offset.length() * (1 - ZOOM_SPEED * dt) >= 0.5) {
              _offset.multiplyScalar(1 - ZOOM_SPEED * dt)
              posChanged = true
            }
          }
          if (movements.has('zoom-out')) { _offset.multiplyScalar(1 + ZOOM_SPEED * dt); posChanged = true }
        }

        if (posChanged) {
          camera.position.copy(target).add(_offset)
          camera.lookAt(target)
        }

        // Pan: move both camera and target in screen-relative direction
        if (!discretePan && (movements.has('pan-left') || movements.has('pan-right') ||
            movements.has('pan-up') || movements.has('pan-down'))) {
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
        if (!discreteRoll && (movements.has('roll-cw') || movements.has('roll-ccw'))) {
          camera.getWorldDirection(_viewDir)
          const angle = (movements.has('roll-cw') ? -1 : 1) * ROLL_SPEED * dt
          camera.up.applyAxisAngle(_viewDir, angle)
          camera.lookAt(target)
        }

        controls.update()
      }
    }

    // Camera change detection for URL sync (detects keyboard + mouse orbit changes)
    if (onCameraChange) {
      const reportCamera = () => {
        _offset.copy(camera.position).sub(target)
        _spherical.setFromVector3(_offset)
        if (_spherical.radius >= 0.001) {
          camera.getWorldDirection(_viewDir)
          _upTarget.set(0, 1, 0)
          const dv = _upTarget.dot(_viewDir)
          _upTarget.addScaledVector(_viewDir, -dv)
          let roll = 0
          if (_upTarget.lengthSq() >= 0.0001) {
            _upTarget.normalize()
            _upCurrent.copy(camera.up)
            const du = _upCurrent.dot(_viewDir)
            _upCurrent.addScaledVector(_viewDir, -du).normalize()
            _axis.crossVectors(_upTarget, _upCurrent)
            const cosR = Math.max(-1, Math.min(1, _upTarget.dot(_upCurrent)))
            const sinR = _axis.dot(_viewDir)
            roll = Math.round(MathUtils.radToDeg(Math.atan2(sinR, cosR)) * 10) / 10
          }
          // Compute target offset from center
          let targetOffset: [number, number, number] | undefined
          if (center) {
            const dx = parseFloat((target.x - center.x).toPrecision(3))
            const dy = parseFloat((target.y - center.y).toPrecision(3))
            const dz = parseFloat((target.z - center.z).toPrecision(3))
            if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01 || Math.abs(dz) > 0.01) {
              targetOffset = [dx, dy, dz]
            }
          }
          onCameraChange(
            Math.round(MathUtils.radToDeg(_spherical.theta) * 10) / 10,
            Math.round(MathUtils.radToDeg(_spherical.phi) * 10) / 10,
            parseFloat(_spherical.radius.toPrecision(3)),
            roll,
            targetOffset,
          )
        }
      }

      // Report immediately when a snap animation just finished
      if (snapJustFinished) {
        reportCamera()
        camSyncRef.current = { lastChange: performance.now(), reported: true }
        prevCamRef.current = [camera.position.x, camera.position.y, camera.position.z, camera.up.x, camera.up.y, camera.up.z]
      } else {
        const { x: px, y: py, z: pz } = camera.position
        const { x: ux, y: uy, z: uz } = camera.up
        const prev = prevCamRef.current
        const changed = !prev ||
          Math.abs(px - prev[0]) > 1e-6 || Math.abs(py - prev[1]) > 1e-6 || Math.abs(pz - prev[2]) > 1e-6 ||
          Math.abs(ux - prev[3]) > 1e-6 || Math.abs(uy - prev[4]) > 1e-6 || Math.abs(uz - prev[5]) > 1e-6
        prevCamRef.current = [px, py, pz, ux, uy, uz]
        if (changed) {
          camSyncRef.current = { lastChange: performance.now(), reported: false }
        } else if (!camSyncRef.current.reported && !isSnapping) {
          if (performance.now() - camSyncRef.current.lastChange >= 500) {
            reportCamera()
            camSyncRef.current.reported = true
          }
        }
      }
    }
  })

  return null
}
