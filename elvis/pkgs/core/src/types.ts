/** 3x3 matrix as row-major flat array [a1x,a1y,a1z, a2x,a2y,a2z, a3x,a3y,a3z] */
export type LatticeMatrix = [
  number, number, number,
  number, number, number,
  number, number, number,
]

export interface AtomInfo {
  element: string
  /** Fractional coordinates */
  fracCoords: [number, number, number]
}

export interface CrystalStructure {
  elements: string[]
  counts: number[]
  atoms: AtomInfo[]
}

export interface DensityGrid {
  /** Grid dimensions [nx, ny, nz] */
  dims: [number, number, number]
  /** Flat density values in VASP order: x varies fastest */
  data: Float32Array
}

export interface VolumeData {
  title: string
  scaleFactor: number
  lattice: LatticeMatrix
  structure: CrystalStructure
  grid: DensityGrid
}
