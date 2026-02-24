# 08 — Dashed Lines for Cell/Box Outlines

## Motivation
The unit cell (abc lattice vectors) and XYZ bounding box outlines are currently solid lines. Dashed lines are a common convention in crystallography visualizations and can help distinguish the cell boundary from actual structural features.

## Feature
- Option to render the cell and/or box outlines with dashed lines instead of solid.
- Dash pattern should be visually clear at typical zoom levels (e.g. dash length ~0.1-0.2 in world units, gap ~0.05-0.1).

## Toggle
- Checkbox in Controls (label: "Dashed outlines"), or a simpler approach: just a URL param.
- URL param `?dl=1` (dashed lines on, default off).

## Implementation
- Three.js provides `LineDashedMaterial` with `dashSize` and `gapSize` properties.
- Important: dashed materials require calling `computeLineDistances()` on the `BufferGeometry` after construction. Without this call the dashes won't render (a common gotcha).
- In R3F this means getting a ref to the line geometry and calling `computeLineDistances()` in a `useEffect` or `onUpdate` callback.
- Example pattern:
  ```tsx
  <line ref={lineRef}>
    <bufferGeometry ... />
    <lineDashedMaterial color="white" dashSize={0.15} gapSize={0.08} />
  </line>
  ```
  with a `useEffect` that calls `lineRef.current.computeLineDistances()`.

## Key files
- `CrystalStructure.tsx` — the cell outline and box outline rendering code; swap `lineBasicMaterial` for `lineDashedMaterial` when the toggle is active.
- `Controls.tsx` — add toggle if using a checkbox (optional; URL param alone may suffice).

## Notes
- `LineDashedMaterial` extends `LineBasicMaterial`, so color and opacity props carry over.
- If the cell and box outlines are separate components, the toggle should apply to both (or offer independent control, but that's probably overkill).
- Dash proportions may need tuning depending on typical cell sizes; consider scaling dash/gap relative to the longest lattice vector.
