import type { LatticeMatrix, AtomInfo, VolumeData } from '../types.ts'

export interface CHGCARHeader {
  title: string
  scaleFactor: number
  lattice: LatticeMatrix
  elements: string[]
  counts: number[]
  atomCount: number
  atoms: AtomInfo[]
  gridDims: [number, number, number]
  /** Line index where density data begins (for range-request offset estimation) */
  dataLineIndex: number
}

/**
 * Parse only the header/metadata from a CHGCAR text (first ~4KB).
 * Useful for previewing file contents from range requests without downloading the full file.
 */
export function parseCHGCARHeader(text: string): CHGCARHeader {
  const lines = text.split('\n')
  let idx = 0

  const title = lines[idx++].trim()
  const scaleFactor = parseFloat(lines[idx++].trim())

  const lattice: LatticeMatrix = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  for (let row = 0; row < 3; row++) {
    const parts = lines[idx++].trim().split(/\s+/).map(Number)
    lattice[row * 3 + 0] = parts[0] * scaleFactor
    lattice[row * 3 + 1] = parts[1] * scaleFactor
    lattice[row * 3 + 2] = parts[2] * scaleFactor
  }

  const elements = lines[idx++].trim().split(/\s+/)
  const counts = lines[idx++].trim().split(/\s+/).map(Number)
  const atomCount = counts.reduce((a, b) => a + b, 0)

  let coordLine = lines[idx++].trim()
  if (coordLine.toLowerCase().startsWith('s')) {
    coordLine = lines[idx++].trim()
  }
  const isDirect = coordLine.toLowerCase().startsWith('d')
  if (!isDirect) {
    throw new Error(`Only Direct (fractional) coordinates supported, got: "${coordLine}"`)
  }

  const atoms: AtomInfo[] = []
  let elemIdx = 0
  let elemCount = 0
  for (let i = 0; i < atomCount; i++) {
    while (elemCount >= counts[elemIdx]) {
      elemIdx++
      elemCount = 0
    }
    const parts = lines[idx++].trim().split(/\s+/).map(Number)
    atoms.push({
      element: elements[elemIdx],
      fracCoords: [parts[0], parts[1], parts[2]],
    })
    elemCount++
  }

  // Blank line
  idx++

  // Grid dimensions
  const gridDims = lines[idx++].trim().split(/\s+/).map(Number) as [number, number, number]
  const dataLineIndex = idx

  return {
    title,
    scaleFactor,
    lattice,
    elements,
    counts,
    atomCount,
    atoms,
    gridDims,
    dataLineIndex,
  }
}

export function parseCHGCAR(text: string): VolumeData {
  const header = parseCHGCARHeader(text)
  const lines = text.split('\n')
  let idx = header.dataLineIndex

  const totalPoints = header.gridDims[0] * header.gridDims[1] * header.gridDims[2]
  const data = new Float32Array(totalPoints)
  let dataIdx = 0
  while (dataIdx < totalPoints) {
    const parts = lines[idx++].trim().split(/\s+/)
    for (const part of parts) {
      if (dataIdx < totalPoints) {
        data[dataIdx++] = parseFloat(part)
      }
    }
  }

  return {
    title: header.title,
    scaleFactor: header.scaleFactor,
    lattice: header.lattice,
    structure: {
      elements: header.elements,
      counts: header.counts,
      atoms: header.atoms,
    },
    grid: { dims: header.gridDims, data },
  }
}
