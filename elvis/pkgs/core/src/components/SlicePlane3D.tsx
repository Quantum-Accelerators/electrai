import { useMemo } from 'react'
import { DoubleSide } from 'three'
import type { LatticeMatrix } from '../types.ts'
import { fracToCart } from '../utils/lattice.ts'

const AXIS_COLORS = ['#ff3653', '#0adb50', '#2c8fff'] as const

interface SlicePlane3DProps {
  lattice: LatticeMatrix
  axis: 0 | 1 | 2
  sliceIndex: number
  dims: [number, number, number]
}

export function SlicePlane3D({ lattice, axis, sliceIndex, dims }: SlicePlane3DProps) {
  const { vertices, color } = useMemo(() => {
    const t = (sliceIndex + 0.5) / dims[axis]
    // Build 4 corners of the slice quad in fractional coords
    const axes = [0, 1, 2].filter(a => a !== axis)
    const corners: [number, number, number][] = []
    for (const u of [0, 1]) {
      for (const v of [0, 1]) {
        const frac: [number, number, number] = [0, 0, 0]
        frac[axis] = t
        frac[axes[0]] = u
        frac[axes[1]] = v
        corners.push(frac)
      }
    }
    // Convert to Cartesian: corners are [00, 01, 10, 11]
    // Triangles: (00,10,01), (10,11,01)
    const c = corners.map(f => fracToCart(lattice, f))
    const verts = new Float32Array([
      ...c[0], ...c[2], ...c[1],
      ...c[2], ...c[3], ...c[1],
    ])
    return { vertices: verts, color: AXIS_COLORS[axis] }
  }, [lattice, axis, sliceIndex, dims])

  return (
    <mesh>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[vertices, 3]}
        />
      </bufferGeometry>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.25}
        side={DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}
