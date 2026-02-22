import { useThree, useFrame } from '@react-three/fiber'
import { Spherical, Vector3 } from 'three'
import type { RefObject } from 'react'

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

interface CameraControllerProps {
  activeMovements: RefObject<Set<string>>
}

export function CameraController({ activeMovements }: CameraControllerProps) {
  const camera = useThree(s => s.camera)
  const controls = useThree(s => s.controls) as { target: Vector3; update: () => void } | null

  useFrame((_, delta) => {
    const movements = activeMovements.current
    if (!movements?.size || !controls) return

    const dt = Math.min(delta, 0.05)
    const target = controls.target

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
