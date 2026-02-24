# Elvis: ELectron VISualization Webapp

## Overview

**Elvis** (ELectron VISualization) — a Vite/TypeScript/React webapp using **Three.js** (via `@react-three/fiber`) for interactive 3D visualization of electron density grids, model inputs/outputs, and crystal structures. Addresses the gap left by underwhelming pymatgen static plots.

The name works for both CHGCAR (charge density) and ELFCAR (electron localization function) data — it's about electron visualization broadly, not tied to one property.

## Motivation

- Current visualization (pymatgen) is static, 2D projections of 3D data
- Researchers need to inspect model predictions vs ground truth interactively
- Could serve as a demo/showcase for the project (deployable as static site)
- "Elvis" branding with the elf character in sunglasses (`elfvis.svg`) makes it memorable

## Features (MVP)

1. **Isosurface rendering**: render electron density as 3D isosurfaces at configurable levels
2. **Crystal structure overlay**: show atom positions, unit cell wireframe, bonds
3. **Side-by-side comparison**: input (SAD guess) vs label (DFT) vs prediction (model output)
4. **Slice viewer**: 2D cross-section planes through the 3D volume
5. **File loading**: drag-and-drop CHGCAR / ELFCAR / `.npy` files

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Build | Vite | Fast, standard |
| UI | React + TypeScript | Ecosystem, type safety |
| 3D | `@react-three/fiber` + `@react-three/drei` | React-native Three.js |
| Isosurfaces | Marching cubes (three.js `MarchingCubes` or custom) | Standard algorithm |
| Data parsing | Custom CHGCAR parser (TS) | CHGCAR is simple text format |
| Styling | SASS or CSS modules | Per project convention |

### Optional: Rust/WASM Core

For large grids (128+), a Rust core compiled to WASM could handle:
- CHGCAR parsing (faster than JS for large files)
- Marching cubes computation
- Grid downsampling / interpolation

This follows the pattern from other projects: Rust core -> WASM for browser, native for server.

## Architecture

```
packages/
  elvis/
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── DensityViewer.tsx      # Main 3D viewport
    │   │   ├── IsosurfaceRenderer.tsx # Marching cubes → mesh
    │   │   ├── CrystalStructure.tsx   # Atoms + unit cell
    │   │   ├── SliceViewer.tsx        # 2D cross-sections
    │   │   ├── ComparisonView.tsx     # Side-by-side
    │   │   └── Controls.tsx           # Iso-level slider, toggles
    │   ├── parsers/
    │   │   ├── chgcar.ts              # CHGCAR/ELFCAR text parser
    │   │   └── npy.ts                 # NumPy .npy binary parser
    │   ├── utils/
    │   │   └── marching-cubes.ts      # Or use three.js built-in
    │   └── types.ts
    └── public/
        └── samples/                   # Small example CHGCARs
```

## Data Flow

```
CHGCAR file (drag-drop or URL)
    → parse lattice vectors, atom positions, density grid
    → Float32Array (Nx × Ny × Nz)
    → marching cubes at iso-level → THREE.BufferGeometry
    → render with lighting, controls, periodic images
```

## Tasks

- [ ] Scaffold Vite/React/TS project in `packages/elvis/`
- [ ] Implement CHGCAR parser (lattice, atoms, density grid)
- [ ] Basic Three.js scene with orbit controls
- [ ] Marching cubes isosurface from density grid
- [ ] Crystal structure overlay (spheres for atoms, lines for unit cell)
- [ ] Iso-level slider control
- [ ] Side-by-side comparison mode
- [ ] 2D slice viewer
- [ ] Deploy as static site (GitHub Pages or similar)
- [ ] Bundle small example CHGCARs for demo

## Open Questions

- Should this live in the same repo (`packages/elvis/`) or separate?
- Do we need a backend? For large files, a Flask/Node server could stream data. For MVP, client-only is simpler.
- What iso-levels are scientifically meaningful? Need domain expert input for defaults.
- Should we support `.npy` output files from Hananeh's prediction pipeline (#62)?
- Port number: hash "elvis" -> pick something in 3000-9000 range, avoid conflicts.
