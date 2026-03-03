import type { LatticeMatrix } from '../types.ts'
import { fracToCart } from './lattice.ts'

export interface TileInfo {
  fracOffset: [number, number, number]
  cartOffset: [number, number, number]
  opacity: number
  isPrimary: boolean
}

/**
 * Chebyshev distance from a fractional-coordinate point to the primary cell [0,1]³.
 * Returns 0 for points inside the cell, positive for points outside.
 */
export function distFromPrimaryCell(frac: [number, number, number]): number {
  const da = Math.max(0, -frac[0], frac[0] - 1)
  const db = Math.max(0, -frac[1], frac[1] - 1)
  const dc = Math.max(0, -frac[2], frac[2] - 1)
  return Math.max(da, db, dc)
}

/**
 * Compute per-atom opacity based on fractional position relative to the primary cell.
 * padding: how far beyond the primary cell boundary to show atoms.
 * fade: power exponent for fade curve. 0 = no fade (hard cutoff at padding edge),
 *       1 = linear, >1 = steep initial drop then gradual tail.
 *       opacity = (1 - d/padding)^fade
 */
export function atomOpacity(
  fracPos: [number, number, number],
  padding: number,
  fade: number,
): number {
  const dist = distFromPrimaryCell(fracPos)
  if (dist >= padding) return 0
  if (fade <= 0) return 1
  return Math.pow(1 - dist / padding, fade)
}

/**
 * Compute tile offsets for symmetric periodic tiling.
 *
 * padding: extra cells around the primary cell, in each direction (along abc axes).
 *   0 = primary only. Symmetric offsets from -ceil(padding) to +ceil(padding).
 *
 * fade: power exponent for opacity curve. 0 = no fade, 1 = linear, >1 = steep then gradual.
 *
 * Per-tile opacity is an approximation (for isosurface/slice/edges); atoms use atomOpacity().
 */
export function computeTiles(
  lattice: LatticeMatrix,
  padding: number,
  fade: number,
): TileInfo[] {
  if (padding <= 0) return [{ fracOffset: [0, 0, 0], cartOffset: [0, 0, 0] as [number, number, number], opacity: 1, isPrimary: true }]

  const shell = Math.ceil(padding)
  const tiles: TileInfo[] = []

  for (let ia = -shell; ia <= shell; ia++) {
    for (let ib = -shell; ib <= shell; ib++) {
      for (let ic = -shell; ic <= shell; ic++) {
        const isPrimary = ia === 0 && ib === 0 && ic === 0
        const fracOffset: [number, number, number] = [ia, ib, ic]
        const cartOffset = fracToCart(lattice, fracOffset)

        if (isPrimary) {
          tiles.push({ fracOffset, cartOffset, opacity: 1, isPrimary: true })
          continue
        }

        // Minimum Chebyshev distance from any point in this tile to [0,1]³.
        // For immediate neighbors (|offset|=1) this is 0; for further tiles it's |offset|-1.
        const minDist = Math.max(
          Math.max(0, Math.abs(ia) - 1),
          Math.max(0, Math.abs(ib) - 1),
          Math.max(0, Math.abs(ic) - 1),
        )

        // Skip tiles entirely outside the padding boundary
        if (minDist >= padding) continue

        let opacity = 1
        if (fade > 0) {
          opacity = Math.pow(1 - minDist / padding, fade)
        }

        tiles.push({ fracOffset, cartOffset, opacity, isPrimary: false })
      }
    }
  }
  return tiles
}
