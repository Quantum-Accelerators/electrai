import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Group, Quaternion, Vector3 } from 'three'

interface ScreenOffsetGroupProps {
  offset: [number, number, number]
  children: React.ReactNode
}

const _invQ = new Quaternion()
const _worldPos = new Vector3()

/**
 * Places children at a fixed offset in screen-space (relative to parent's origin),
 * by counter-rotating the offset against the parent's world rotation each frame.
 * This prevents the group from orbiting around the parent when the camera/gizmo rotates.
 */
export function ScreenOffsetGroup({ offset, children }: ScreenOffsetGroupProps) {
  const groupRef = useRef<Group>(null!)

  useFrame(() => {
    const group = groupRef.current
    if (!group?.parent) return
    // Get parent's world quaternion and invert it
    group.parent.getWorldQuaternion(_invQ)
    _invQ.invert()
    // Apply inverse rotation to the desired offset → fixed screen position
    _worldPos.set(...offset).applyQuaternion(_invQ)
    group.position.copy(_worldPos)
  })

  return <group ref={groupRef}>{children}</group>
}
