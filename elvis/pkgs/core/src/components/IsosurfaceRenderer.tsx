import { useMemo } from 'react'
import type { VolumeData } from '../types.ts'
import { marchingCubes } from '../utils/marching-cubes.ts'

interface IsosurfaceRendererProps {
  volume: VolumeData
  isoLevel: number
  opacity: number
}

export function IsosurfaceRenderer({ volume, isoLevel, opacity }: IsosurfaceRendererProps) {
  const geometry = useMemo(() => {
    return marchingCubes(
      volume.grid.data,
      volume.grid.dims,
      isoLevel,
      volume.lattice,
    )
  }, [volume, isoLevel])

  if (geometry.getAttribute('position')?.count === 0) return null

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color="#44aaff"
        transparent
        opacity={opacity}
        side={2 /* DoubleSide */}
        depthWrite={false}
      />
    </mesh>
  )
}
