import { Matrix4, Vector3 } from 'three'
import type { LatticeMatrix } from '../types.ts'

/**
 * Build a THREE.Matrix4 that maps fractional → Cartesian coordinates.
 * Lattice vectors are rows of the 3x3 matrix; we set them as columns in Matrix4
 * so that multiplying by (fx, fy, fz) yields the Cartesian position.
 */
export function latticeToMatrix4(lat: LatticeMatrix): Matrix4 {
  const m = new Matrix4()
  // Matrix4.set is row-major: (n11,n12,n13,n14, n21,...)
  // We want columns = lattice vectors a, b, c
  m.set(
    lat[0], lat[3], lat[6], 0,
    lat[1], lat[4], lat[7], 0,
    lat[2], lat[5], lat[8], 0,
    0,      0,      0,      1,
  )
  return m
}

/**
 * Cell volume via scalar triple product: a · (b × c)
 */
export function cellVolume(lat: LatticeMatrix): number {
  const a = new Vector3(lat[0], lat[1], lat[2])
  const b = new Vector3(lat[3], lat[4], lat[5])
  const c = new Vector3(lat[6], lat[7], lat[8])
  return Math.abs(a.dot(new Vector3().crossVectors(b, c)))
}

/**
 * 12 edges of the unit cell as pairs of fractional-coordinate corners.
 * Each corner is (i, j, k) with i,j,k ∈ {0, 1}.
 */
export function unitCellEdges(): Array<[[number, number, number], [number, number, number]]> {
  const corners: Array<[number, number, number]> = []
  for (let i = 0; i <= 1; i++) {
    for (let j = 0; j <= 1; j++) {
      for (let k = 0; k <= 1; k++) {
        corners.push([i, j, k])
      }
    }
  }

  const edges: Array<[[number, number, number], [number, number, number]]> = []
  for (let a = 0; a < corners.length; a++) {
    for (let b = a + 1; b < corners.length; b++) {
      // Two corners share an edge if they differ in exactly one coordinate
      let diff = 0
      for (let d = 0; d < 3; d++) {
        if (corners[a][d] !== corners[b][d]) diff++
      }
      if (diff === 1) {
        edges.push([corners[a], corners[b]])
      }
    }
  }
  return edges
}

/**
 * Transform a fractional coordinate to Cartesian using the lattice matrix.
 */
export function fracToCart(lat: LatticeMatrix, frac: [number, number, number]): [number, number, number] {
  return [
    frac[0] * lat[0] + frac[1] * lat[3] + frac[2] * lat[6],
    frac[0] * lat[1] + frac[1] * lat[4] + frac[2] * lat[7],
    frac[0] * lat[2] + frac[1] * lat[5] + frac[2] * lat[8],
  ]
}
