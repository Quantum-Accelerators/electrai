# 07 — Atom Labels in 3D View

## Motivation
The Controls panel legend shows colored dots with element symbols, but the 3D atoms themselves are unlabeled. For structures with multiple similar-looking elements, labels in the viewport make identification much easier.

## Feature

### Labels on atoms
- Each atom can display a text label showing its element symbol (e.g. "Fe", "O").
- Labels face the camera (billboard behavior) and stay readable at any viewing angle.

### Visibility modes
Pick one (or make configurable):
1. **Always on, semi-transparent** — labels rendered at ~40-50% opacity so they don't dominate the scene.
2. **Hover only** — labels appear when the cursor is near/over an atom (requires raycasting against atom meshes).
3. **Zoom-dependent** — labels fade in when the camera is close enough.

Option 1 is simplest to implement. Options 2/3 are nicer but more work.

### Toggle
- Checkbox in Controls panel (label: "Atom labels").
- Keyboard shortcut `t l` (mnemonic: toggle labels).
- URL param `?al=1` (atom labels on).

## Implementation
- drei's `<Billboard>` + `<Text>` is a straightforward approach: one `<Billboard>` per atom positioned at the atom center, with a `<Text>` child.
- Alternatively, `<Html>` gives DOM-based labels (easier styling, but can feel disconnected from the 3D scene and has higher overhead at scale).
- For structures with many atoms, instanced labels or sprite-based text would be more performant, but `<Text>` should be fine for typical unit cells (tens of atoms).

## Key files
- `CrystalStructure.tsx` — `AtomInstances` component; add label children co-located with each atom sphere.
- `Controls.tsx` — add toggle checkbox.
- `utils/elements.ts` — already has element data (symbols, colors); labels pull symbols from here.

## Notes
- Label font size should be small relative to atom radius (e.g. 0.15-0.25 units) to avoid clutter.
- Consider a slight vertical offset so the label doesn't sit inside the sphere.
- If hover mode is chosen, a shared raycast (drei's `useIntersect` or `onPointerOver`) on the instanced mesh can identify which instance is hovered.
