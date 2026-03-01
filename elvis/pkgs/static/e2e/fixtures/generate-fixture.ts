#!/usr/bin/env -S npx tsx
/**
 * Generate a minimal pymatgen-format .json.gz test fixture.
 *
 * Structure: 4 Å cubic cell, Fe at origin, O at (½,½,½).
 * Grid: 32×32×32 with gaussian blobs at atom sites.
 * task_id: "mp-1000020" (matches default material).
 *
 * Usage: npx tsx e2e/fixtures/generate-fixture.ts
 */
import { gzipSync } from 'node:zlib'
import { writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const N = 32
const a = 4.0 // lattice constant in Å

// Build 3D grid with gaussian blobs at atom positions
function gaussian(dx: number, dy: number, dz: number, sigma = 0.8): number {
  return Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma))
}

const atomsFrac = [
  [0.0, 0.0, 0.0], // Fe
  [0.5, 0.5, 0.5], // O
]

const grid: number[][][] = []
for (let iz = 0; iz < N; iz++) {
  const plane: number[][] = []
  for (let iy = 0; iy < N; iy++) {
    const row: number[] = []
    for (let ix = 0; ix < N; ix++) {
      const fx = ix / N
      const fy = iy / N
      const fz = iz / N
      let val = 0
      for (const [ax, ay, az] of atomsFrac) {
        // Minimum-image convention for periodic distance
        let dx = fx - ax; dx -= Math.round(dx)
        let dy = fy - ay; dy -= Math.round(dy)
        let dz = fz - az; dz -= Math.round(dz)
        val += gaussian(dx * a, dy * a, dz * a)
      }
      row.push(val)
    }
    plane.push(row)
  }
  grid.push(plane)
}

const fixture = {
  task_id: "mp-1000020",
  data: {
    poscar: {
      structure: {
        lattice: {
          matrix: [
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a],
          ],
        },
        sites: [
          { species: [{ element: "Fe" }], abc: [0.0, 0.0, 0.0] },
          { species: [{ element: "O" }], abc: [0.5, 0.5, 0.5] },
        ],
      },
    },
    data: {
      total: { data: grid },
    },
  },
}

const json = JSON.stringify(fixture)
const gz = gzipSync(Buffer.from(json))

const outPath = join(dirname(fileURLToPath(import.meta.url)), 'mp-test.json.gz')
writeFileSync(outPath, gz)
console.log(`Wrote ${outPath} (${gz.length} bytes)`)
