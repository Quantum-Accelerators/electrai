import { useMemo } from 'react'
import type { VolumeData } from '../types.ts'
import { marchingCubes, extendPeriodicGrid } from '../utils/marching-cubes.ts'

interface IsosurfaceRendererProps {
  volume: VolumeData
  isoLevel: number
  opacity: number
}

export function IsosurfaceRenderer({ volume, isoLevel, opacity }: IsosurfaceRendererProps) {
  const extended = useMemo(
    () => extendPeriodicGrid(volume.grid.data, volume.grid.dims),
    [volume],
  )

  const geometry = useMemo(() => {
    return marchingCubes(
      extended.data,
      extended.dims,
      isoLevel,
      volume.lattice,
      volume.grid.dims,
    )
  }, [extended, isoLevel, volume.lattice])

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
