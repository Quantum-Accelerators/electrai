# 10 — Volumetric Density Visualization

## Motivation
The current isosurface rendering shows a single threshold — regions above the iso-level are opaque, everything else is invisible. This makes it impossible to see:
- Low-density "floating" electrons not localized on atoms
- The full spatial distribution of charge/ELF throughout the cell
- How density varies continuously from atom cores to interstitial regions

A volumetric rendering would show the entire 3D density field as a semi-transparent colored volume, making all of these features immediately apparent.

## Approaches

We should implement multiple approaches and let the user switch between them, since each has different strengths. All approaches share the same colormap (viridis or user-selectable) and transfer function (mapping density → color + opacity).

### 1. Volume raycasting (highest quality)

Render the 3D density grid directly using GPU raycasting through a `DataTexture3D`.

**How it works:**
- Upload the density grid as a 3D texture
- Render a bounding-box cube
- Fragment shader ray-marches from camera through the volume, sampling the 3D texture at each step
- Each sample maps density → (color, opacity) via a transfer function
- Accumulate color using front-to-back compositing

**Pros:** Continuous, smooth, physically accurate appearance. No geometry to generate. Handles arbitrary viewpoints.

**Cons:** Most complex to implement. Performance depends on ray step count and grid resolution. Needs careful transfer function tuning.

**Implementation:**
- Three.js `Data3DTexture` for the density grid
- Custom `ShaderMaterial` with vertex + fragment shaders
- Transfer function as a 1D texture (256-entry colormap × opacity curve)
- Controls: step count (quality vs performance), opacity multiplier, density range clamp
- The existing viridis colormap from `SliceViewer` can be reused

### 2. Multi-isosurface (layered shells)

Render several isosurfaces at different thresholds with decreasing opacity.

**How it works:**
- Run marching cubes at N thresholds (e.g. 5–10 levels spanning the density range)
- Each surface gets a color from the colormap and opacity proportional to its threshold
- Higher-density surfaces are more opaque; low-density surfaces are very transparent

**Pros:** Builds on existing marching cubes infrastructure. Intuitive — each "shell" shows a density contour. Easier to implement than raycasting.

**Cons:** Discrete layers can show banding artifacts. More geometry = more memory. Transparency sorting issues with multiple overlapping transparent meshes.

**Implementation:**
- Reuse existing `marchingCubes()` function, called at multiple thresholds
- Each level gets its own `MeshStandardMaterial` with `transparent: true`, `opacity` scaled by threshold, `depthWrite: false`
- Render order: back-to-front (lowest density first)
- Controls: number of levels, density range, overall opacity scale

### 3. 3D scatter / point cloud

Sample the density field at grid points and render as colored, sized points.

**How it works:**
- For each voxel (or subsampled subset), emit a point colored by density
- Point size and/or opacity scales with density value
- Low-density points are small/transparent; high-density points are large/opaque

**Pros:** Simple to implement. Fast with `<points>` / `PointsMaterial`. Good for getting a quick sense of the density distribution. Works well with three.js instancing.

**Cons:** Can look noisy/grainy. Dense grids produce millions of points. Doesn't capture fine surface detail.

**Implementation:**
- `BufferGeometry` with position, color, size attributes
- Custom `ShaderMaterial` or `PointsMaterial` with `size` attenuation
- Density threshold filter: skip points below a minimum density to reduce point count
- Controls: point size multiplier, density floor, subsampling factor

## Shared components

### Transfer function
All approaches need density → (color, opacity) mapping:
- **Colormap:** Viridis (default), with option for others (plasma, inferno, magma)
- **Opacity curve:** Configurable — linear, exponential, or custom ramp. Key parameter: the density range [min, max] mapped to [0, 1] opacity.
- **Density windowing:** Min/max sliders to focus on a density range of interest (like medical imaging "window/level")

### Controls UI
Add a "Volume" section to Controls (or extend the existing Surface section):
- **Mode selector:** Off / Isosurface (current) / Volume raycast / Multi-iso / Point cloud
- **Opacity:** Global opacity multiplier (slider, 0–1)
- **Density range:** Min/max sliders (or a range slider) to window into the density
- **Quality:** For raycasting: step count. For multi-iso: number of levels. For points: subsampling.

### URL params
- `?vm=ray` — volume mode (`iso` default, `ray`, `multi`, `pts`)
- `?vr=0.1,0.9` — density range as fraction of [min, max] (default full range)
- `?vo=0.5` — volume opacity multiplier

## Key files
- `DensityViewer.tsx` — currently renders single isosurface; needs mode switch
- New: `VolumeRaycast.tsx` — raycasting shader + 3D texture
- New: `MultiIsosurface.tsx` — multi-threshold marching cubes
- New: `DensityPoints.tsx` — point cloud renderer
- `Controls.tsx` — volume mode selector and associated sliders
- `SliceViewer.tsx` — already has viridis colormap to reuse

## Implementation order

1. **Multi-isosurface** — lowest effort, builds on existing MC code, immediately useful
2. **Point cloud** — quick to implement, good for exploration
3. **Volume raycasting** — highest effort but best results, implement last

## Notes
- ELF data (0–1 range, electron localization) is particularly well-suited to volumetric rendering since the values have clear physical meaning across the full range.
- CHGCAR charge density varies over many orders of magnitude — log-scale transfer functions may be needed.
- Performance: raycasting on a 96×108×96 grid should be fine on modern GPUs. Point cloud may need subsampling for large grids.
- Consider allowing the isosurface and volumetric modes to be shown simultaneously (iso surface + transparent volume behind it).
