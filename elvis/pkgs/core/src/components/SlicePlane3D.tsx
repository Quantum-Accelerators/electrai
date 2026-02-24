import { useMemo } from 'react'
import { CanvasTexture, DoubleSide, NearestFilter } from 'three'
import type { LatticeMatrix } from '../types.ts'
import { fracToCart } from '../utils/lattice.ts'
import { viridis } from '../utils/colormap.ts'

interface SlicePlane3DProps {
  lattice: LatticeMatrix
  axis: 0 | 1 | 2
  sliceIndex: number
  dims: [number, number, number]
  data: Float32Array
}

export function SlicePlane3D({ lattice, axis, sliceIndex, dims, data }: SlicePlane3DProps) {
  const { vertices, uvs, texture } = useMemo(() => {
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
    // UVs matching the triangle vertex order
    const uv = new Float32Array([
      0, 0,  1, 0,  0, 1,
      1, 0,  1, 1,  0, 1,
    ])

    // Generate density texture for this slice
    const w = dims[axes[0]]
    const h = dims[axes[1]]
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    const imageData = ctx.createImageData(w, h)
    let min = Infinity, max = -Infinity
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        let ix: number, iy: number, iz: number
        if (axis === 0) { ix = sliceIndex; iy = i; iz = j }
        else if (axis === 1) { ix = i; iy = sliceIndex; iz = j }
        else { ix = i; iy = j; iz = sliceIndex }
        const val = data[ix + dims[0] * (iy + dims[1] * iz)]
        if (val < min) min = val
        if (val > max) max = val
      }
    }
    const range = max - min || 1
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        let ix: number, iy: number, iz: number
        if (axis === 0) { ix = sliceIndex; iy = i; iz = j }
        else if (axis === 1) { ix = i; iy = sliceIndex; iz = j }
        else { ix = i; iy = j; iz = sliceIndex }
        const val = data[ix + dims[0] * (iy + dims[1] * iz)]
        const nt = (val - min) / range
        const [r, g, b] = viridis(nt)
        const idx = (j * w + i) * 4
        imageData.data[idx] = r
        imageData.data[idx + 1] = g
        imageData.data[idx + 2] = b
        imageData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imageData, 0, 0)

    const tex = new CanvasTexture(canvas)
    tex.minFilter = NearestFilter
    tex.magFilter = NearestFilter
    tex.needsUpdate = true

    return { vertices: verts, uvs: uv, texture: tex }
  }, [lattice, axis, sliceIndex, dims, data])

  return (
    <mesh>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[vertices, 3]}
        />
        <bufferAttribute
          attach="attributes-uv"
          args={[uvs, 2]}
        />
      </bufferGeometry>
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.85}
        side={DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}
