# 03 — Elvis: ELectron VISualization Webapp

## Status: Complete (MVP)

Implemented as a pnpm monorepo (`elvis/`) with two packages:
- `pkgs/core` — shared components and logic (DensityViewer, Controls, parsers, etc.)
- `pkgs/static` — Vite-built static site (App.tsx, OPFS storage, S3 fetching)

Deployed via GitHub Pages (`gh-pages.yml`).

## What was built

- Isosurface rendering (marching cubes via three.js)
- Crystal structure overlay (instanced atom spheres, abc cell + XYZ box outlines)
- 2D slice viewer with viridis heatmap + 3D textured slice plane
- Keyboard-driven camera (orbit, pan, zoom, roll, axis-snaps, discrete orbit steps)
- URL state for all controls (`use-prms`), camera position (`?c=`), orbit step (`?od=`)
- OPFS caching of loaded volumes
- Materials Project `.json.gz` loading via anonymous S3 HTTPS
- File upload (CHGCAR, ELFCAR, .json.gz)
- Comparison view (side-by-side, basic)
- Omnibar, editable shortcuts, speed dial (`use-kbd`)

## Resolved open questions

- Lives in same repo as monorepo sub-directory (`elvis/`)
- No backend needed; client-only with anonymous S3 HTTPS for MP data
- Default iso-level: mean + 2*sigma of density data
- `.json.gz` (pymatgen format) is the primary data format; raw CHGCAR also supported
- Port: 3150
