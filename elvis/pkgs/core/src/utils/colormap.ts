/** Viridis-like color map (16 stops) */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 26, 108], [71, 47, 126], [65, 68, 135],
  [57, 86, 140], [49, 104, 142], [42, 120, 142], [35, 137, 142],
  [31, 154, 138], [34, 170, 127], [53, 186, 109], [86, 199, 83],
  [128, 209, 54], [177, 214, 24], [225, 213, 13], [253, 231, 37],
]

/** Interpolate the viridis colormap at t ∈ [0, 1] → [r, g, b] in 0–255. */
export function viridis(t: number): [number, number, number] {
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
