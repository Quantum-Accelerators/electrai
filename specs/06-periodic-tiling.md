# 06 — Periodic Tiling & Origin Offset

## Motivation
Crystal structures are periodic. Currently the app renders a single unit cell, so atoms sitting at cell boundaries get clipped or appear incomplete (e.g. Au₃Li shows Li split across all 8 corners). Rendering neighboring cells and allowing origin shifts gives a more physically intuitive view.

With even a 2×2×2 tiling, the user can orbit + pan to inspect any region of interest — effectively accomplishing the same thing as "rotating the abc frame" without needing that complexity.

## Feature

### Multi-cell tiling
- Render NxNxN copies of the unit cell around the primary cell (default 1×1×1, i.e. no tiling).
- User selects tiling count per axis, e.g. 2×2×2 or 3×1×1, via Controls sliders or number inputs.
- Primary cell renders at full opacity.

### Fade-out for tiled copies
- Tiled copies fade from full opacity at the primary cell boundary to transparent at the outer edge.
- Configurable fade: a `tileFade` parameter (0–1, default 0.5) controls where the fade begins as a fraction of the tiled extent. At 0.5, the outer half of each tiled cell fades to transparent.
- Applies to both atoms (instance opacity) and isosurface copies (material opacity).
- This gives boundary context without visual clutter — you see enough of the neighbors to understand the periodicity without a jarring hard cutoff.

### Origin offset
- Three sliders (a, b, c) in Controls let the user shift the origin in fractional coordinates (0–1 range along each lattice vector).
- Shifting the origin re-centers the view so the user can inspect a region of interest without edge clipping.
- The "active" primary cell moves with the offset; neighboring tiled cells update accordingly.

### URL params
- `?ta=2&tb=2&tc=2` — tiling count along a, b, c axes (integers, default 1).
- `?oa=0.5&ob=0&oc=0` — fractional origin offset along a, b, c (floats 0–1, default 0).
- `?tf=0.5` — tile fade fraction (float 0–1, default 0.5).

## Key files
- `CrystalStructure.tsx` — atom rendering; needs to iterate over tiled copies and apply ghost styling.
- `DensityViewer.tsx` — isosurface mesh; needs tiled copies with reduced opacity.
- Dataloader code — may need to supply repeated/offset coordinate data, or tiling can be handled purely in the rendering layer via instanced transforms.
- `Controls.tsx` — UI for tiling counts, offset sliders, and fade slider.

## Notes
- Instanced rendering (`<instancedMesh>`) is a natural fit for tiling atoms without multiplying draw calls. Per-instance color/opacity can encode the fade.
- For density, duplicating the isosurface geometry with offset transforms and a transparent material is simplest; re-computing the full tiled scalar field is expensive and likely unnecessary.
- Consider clamping max tiling to something reasonable (e.g. 5 per axis) to avoid performance issues.
- With 2×2×2 tiling + orbit/pan, the user can effectively view the crystal from any perspective with full boundary context — this subsumes the "rotate abc frame" idea without additional complexity.
