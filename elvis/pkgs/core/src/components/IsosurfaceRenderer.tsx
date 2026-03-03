import { useMemo } from 'react'
import type { VolumeData } from '../types.ts'
import { marchingCubes, extendPeriodicGrid } from '../utils/marching-cubes.ts'
import type { TileInfo } from '../utils/tiling.ts'
import { tileFadeCompile } from '../utils/tile-fade.ts'

interface IsosurfaceRendererProps {
  volume: VolumeData
  isoLevel: number
  opacity: number
  tiles?: TileInfo[]
  tilePadding?: number
  tileFade?: number
}

export function IsosurfaceRenderer({ volume, isoLevel, opacity, tiles, tilePadding = 0, tileFade = 1 }: IsosurfaceRendererProps) {
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

  const fadeCompile = useMemo(() => {
    if (tilePadding <= 0) return undefined
    return tileFadeCompile(volume.lattice, tilePadding, tileFade)
  }, [volume.lattice, tilePadding, tileFade])

  const tileList = tiles ?? [{ fracOffset: [0, 0, 0] as [number, number, number], cartOffset: [0, 0, 0] as [number, number, number], opacity: 1, isPrimary: true }]

  return (
    <>
      {tileList.map((tile, i) => {
        if (tile.opacity <= 0) return null
        return (
          <mesh key={i} geometry={geometry} position={tile.cartOffset}>
            <meshStandardMaterial
              key={`iso-${tilePadding}-${tileFade}`}
              color="#44aaff"
              transparent
              opacity={opacity}
              side={2 /* DoubleSide */}
              depthWrite={false}
              onBeforeCompile={fadeCompile}
            />
          </mesh>
        )
      })}
    </>
  )
}
