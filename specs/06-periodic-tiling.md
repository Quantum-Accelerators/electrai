# 06 — Periodic Tiling & Origin Offset

## Motivation
Crystal structures are periodic. Currently the app renders a single unit cell, so atoms sitting at cell boundaries get clipped or appear incomplete. Rendering neighboring cells and allowing origin shifts would give a more physically intuitive view.

## Feature

### Multi-cell tiling
- Render NxNxN copies of the unit cell around the primary cell (default 1x1x1, i.e. no tiling).
- User selects tiling count per axis, e.g. 2x2x2 or 3x1x1, via Controls sliders or number inputs.
- Atoms and electron density in the primary cell render at full opacity; copies in neighboring cells are dimmed/ghosted (reduced opacity, desaturated color, or wireframe-only density).

### Origin offset
- Three sliders (a, b, c) in Controls let the user shift the origin in fractional coordinates (0–1 range along each lattice vector).
- Shifting the origin re-centers the view so the user can inspect a region of interest without edge clipping.
- The "active" primary cell moves with the offset; neighboring tiled cells update accordingly.

### URL params
- `?ta=2&tb=2&tc=2` — tiling count along a, b, c axes (integers, default 1).
- `?oa=0.5&ob=0&oc=0` — fractional origin offset along a, b, c (floats 0–1, default 0).

## Key files
- `CrystalStructure.tsx` — atom rendering; needs to iterate over tiled copies and apply ghost styling.
- `DensityViewer.tsx` — isosurface mesh; needs tiled copies with reduced opacity.
- Dataloader code — may need to supply repeated/offset coordinate data, or tiling can be handled purely in the rendering layer via instanced transforms.
- `Controls.tsx` — UI for tiling counts and offset sliders.

## Notes
- Instanced rendering (`<instancedMesh>`) is a natural fit for tiling atoms without multiplying draw calls.
- For density, duplicating the isosurface geometry with offset transforms and a transparent material is simplest; re-computing the full tiled scalar field is expensive and likely unnecessary.
- Consider clamping max tiling to something reasonable (e.g. 5 per axis) to avoid performance issues.
