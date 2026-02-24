# 09 — Discrete Zoom & Pan Steps

## Motivation

Orbit already supports discrete steps (`?od=90`, arrow keys snap 90 degrees). But zoom (`-`/`=`) and pan (`shift+arrows`) are continuous: hold duration determines travel distance, making them non-invertible. Pressing `-` for 0.5s then `=` for 0.5s doesn't return to the original zoom because hold times are never exactly equal.

## Feature

### Discrete zoom steps
- A single `-` press moves the camera a fixed distance outward; `=` moves the same distance inward.
- If held, the key chains (like orbit steps): one step completes, then the next begins, giving smooth but quantized movement.
- Step size is configurable (URL param, Controls input, `\d+ z` keyboard binding).
- Default step size: TBD (maybe 20% of current distance, or a fixed world-unit amount).

### Discrete pan steps
- Same pattern for `shift+arrows`: each press pans a fixed distance in screen-relative direction.
- Step size configurable separately from zoom.

### URL params
- `?zd=<number>` — zoom step (0 = continuous, >0 = discrete step as percentage of current distance)
- `?pd=<number>` — pan step (0 = continuous, >0 = discrete step in world units)

### Keyboard bindings
- `\d+ z` — set zoom step size (like `\d+ o` for orbit)
- `\d+ p` — set pan step size
- `t z` / `t p` — toggle discrete zoom/pan on/off (defaults to reasonable step)

## Implementation notes

- Zoom step logic in `CameraController.tsx`: on key press, compute target distance as `currentDistance * (1 ± stepFraction)`, animate to it.
- Reuses the snap animation system (start/end quaternion + radius) but only changes radius for zoom.
- Pan step: compute target offset as `currentTarget + stepSize * screenDirection`, animate camera+target together.
- Key-hold chaining: same pattern as orbit steps (`lastZoomStepRef`, `lastPanStepRef`).

## Key files
- `CameraController.tsx` — add zoom-step and pan-step snap types
- `App.tsx` — URL params, keyboard bindings
- `Controls.tsx` — inputs for step sizes

## Notes
- Zoom step as percentage (e.g. 20%) is more natural than absolute distance, since it's scale-invariant.
- Pan step in world units makes sense since it relates to the structure size.
- For zoom, invertibility requires: `distance * (1 + step) * (1 - step') = distance`. With percentage steps, `step' = step / (1 + step)` — not exactly symmetric. Consider using multiplicative steps: `distance * factor` for zoom-in, `distance / factor` for zoom-out. Then `-` followed by `=` is exactly invertible.
