import { useRef, useEffect, useMemo, useCallback } from 'react'
import type { VolumeData } from '../types.ts'

interface SliceViewerProps {
  volume: VolumeData
  axis: 0 | 1 | 2
  sliceIndex: number
}

// Viridis-like color map (16 stops)
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 26, 108], [71, 47, 126], [65, 68, 135],
  [57, 86, 140], [49, 104, 142], [42, 120, 142], [35, 137, 142],
  [31, 154, 138], [34, 170, 127], [53, 186, 109], [86, 199, 83],
  [128, 209, 54], [177, 214, 24], [225, 213, 13], [253, 231, 37],
]

function viridis(t: number): [number, number, number] {
  const idx = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, VIRIDIS.length - 1)
  const f = idx - lo
  return [
    VIRIDIS[lo][0] + f * (VIRIDIS[hi][0] - VIRIDIS[lo][0]),
    VIRIDIS[lo][1] + f * (VIRIDIS[hi][1] - VIRIDIS[lo][1]),
    VIRIDIS[lo][2] + f * (VIRIDIS[hi][2] - VIRIDIS[lo][2]),
  ]
}

export function SliceViewer({ volume, axis, sliceIndex }: SliceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { dims, data } = volume.grid

  // Determine the 2D slice dimensions
  const [width, height] = useMemo(() => {
    if (axis === 0) return [dims[1], dims[2]] // y, z
    if (axis === 1) return [dims[0], dims[2]] // x, z
    return [dims[0], dims[1]] // x, y
  }, [axis, dims])

  // Extract slice and compute min/max for normalization
  const extractSlice = useCallback(() => {
    const slice = new Float32Array(width * height)
    let min = Infinity, max = -Infinity
    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        let ix: number, iy: number, iz: number
        if (axis === 0) { ix = sliceIndex; iy = i; iz = j }
        else if (axis === 1) { ix = i; iy = sliceIndex; iz = j }
        else { ix = i; iy = j; iz = sliceIndex }
        const val = data[ix + dims[0] * (iy + dims[1] * iz)]
        slice[j * width + i] = val
        if (val < min) min = val
        if (val > max) max = val
      }
    }
    return { slice, min, max }
  }, [data, dims, axis, sliceIndex, width, height])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height

    const { slice, min, max } = extractSlice()
    const range = max - min || 1
    const imageData = ctx.createImageData(width, height)

    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        const t = (slice[j * width + i] - min) / range
        const [r, g, b] = viridis(t)
        const idx = (j * width + i) * 4
        imageData.data[idx] = r
        imageData.data[idx + 1] = g
        imageData.data[idx + 2] = b
        imageData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imageData, 0, 0)
  }, [extractSlice, width, height])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        imageRendering: 'pixelated',
      }}
    />
  )
}
