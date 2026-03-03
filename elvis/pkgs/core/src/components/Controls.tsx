import { getElement } from '../utils/elements.ts'
import styles from './Controls.module.css'

const ResetIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M3.5 2.5v4h4" />
    <path d="M3.5 6.5A5.5 5.5 0 1 1 2.5 9.5" />
  </svg>
)

interface ControlsProps {
  isoLevel: number
  defaultIsoLevel: number
  maxDensity: number
  onIsoLevelChange: (v: number) => void
  opacity: number
  onOpacityChange: (v: number) => void
  showAtoms: boolean
  onShowAtomsChange: (v: boolean) => void
  showAtomLabels: boolean
  onShowAtomLabelsChange: (v: boolean) => void
  showAbcCell: boolean
  onShowAbcCellChange: (v: boolean) => void
  showXyzBox: boolean
  onShowXyzBoxChange: (v: boolean) => void
  dashedLines: boolean
  onDashedLinesChange: (v: boolean) => void
  orbitDeg: number
  onOrbitDegChange: (v: number) => void
  zoomPct: number
  onZoomPctChange: (v: number) => void
  panStep: number
  onPanStepChange: (v: number) => void
  rollDeg: number
  onRollDegChange: (v: number) => void
  showSlice: boolean
  onShowSliceChange: (v: boolean) => void
  sliceAxis: 0 | 1 | 2
  onSliceAxisChange: (v: 0 | 1 | 2) => void
  sliceIndex: number
  maxSliceIndex: number
  onSliceIndexChange: (v: number) => void
  animDuration: number
  onAnimDurationChange: (v: number) => void
  sweepMode: 'd' | 'r'
  sweepDuration: number
  onSweepDurationChange: (v: number) => void
  onSweepModeToggle: () => void
  sliceSpeed: number
  onSliceSpeedChange: (v: number) => void
  cam: [number, number, number, number] | null
  filename: string
  elements?: string[]
  counts?: number[]
  tilePadding?: number
  onTilePaddingChange?: (v: number) => void
  tileFade?: number
  onTileFadeChange?: (v: number) => void
}

const AXIS_LABELS = ['X', 'Y', 'Z'] as const

// Non-linear padding slider: 0-20 ticks → 0-1 (step 0.05), 20-30 ticks → 1-2 (step 0.1)
const PADDING_TICKS = 30
function paddingToTick(v: number): number {
  if (v <= 1) return Math.round(v / 0.05)
  return 20 + Math.round((v - 1) / 0.1)
}
function tickToPadding(t: number): number {
  if (t <= 20) return Math.round(t * 0.05 * 100) / 100
  return Math.round((1 + (t - 20) * 0.1) * 100) / 100
}

function fmtAngle(n: number): string {
  const s = n.toFixed(1)
  return s.endsWith('.0') ? s.slice(0, -2) : s
}

export function Controls({
  isoLevel,
  defaultIsoLevel,
  maxDensity,
  onIsoLevelChange,
  opacity,
  onOpacityChange,
  showAtoms,
  onShowAtomsChange,
  showAtomLabels,
  onShowAtomLabelsChange,
  showAbcCell,
  onShowAbcCellChange,
  showXyzBox,
  onShowXyzBoxChange,
  dashedLines,
  onDashedLinesChange,
  orbitDeg,
  onOrbitDegChange,
  zoomPct,
  onZoomPctChange,
  panStep,
  onPanStepChange,
  rollDeg,
  onRollDegChange,
  showSlice,
  onShowSliceChange,
  sliceAxis,
  onSliceAxisChange,
  sliceIndex,
  maxSliceIndex,
  onSliceIndexChange,
  animDuration,
  onAnimDurationChange,
  sweepMode,
  sweepDuration,
  onSweepDurationChange,
  onSweepModeToggle,
  sliceSpeed,
  onSliceSpeedChange,
  cam,
  filename,
  elements,
  counts,
  tilePadding,
  onTilePaddingChange,
  tileFade,
  onTileFadeChange,
}: ControlsProps) {
  const tp = tilePadding ?? 0
  const hasTiling = tp > 0

  return (
    <div className={styles.controls}>
      <div className={styles.controlTitle}>{filename}</div>
      {elements && elements.length > 0 && (
        <div style={{ display: 'flex', gap: 8, padding: '2px 0 6px', flexWrap: 'wrap' }}>
          {elements.map((el, i) => {
            const { color } = getElement(el)
            const css = `#${color.toString(16).padStart(6, '0')}`
            const count = counts?.[i]
            return (
              <span key={el} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#ccc' }}>
                <span style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: css,
                  display: 'inline-block',
                  flexShrink: 0,
                }} />
                {el}{count != null && count > 1 && <sub>{count}</sub>}
              </span>
            )
          })}
        </div>
      )}

      <details className={styles.section} open>
        <summary className={styles.sectionTitle}>Surface</summary>
        <div className={styles.sectionBody}>
          <div className={styles.controlLabel}>
            <div className={styles.sliderHeader}>
              <span>Iso-level: {isoLevel.toFixed(1)}</span>
              <button
                className={styles.resetBtn}
                onClick={() => onIsoLevelChange(defaultIsoLevel)}
                title={`Reset to ${defaultIsoLevel.toFixed(1)}`}
                disabled={Math.abs(isoLevel - defaultIsoLevel) <= 0.1}
              >
                <ResetIcon />
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={maxDensity}
              step={maxDensity / 500}
              value={isoLevel}
              onChange={e => onIsoLevelChange(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </div>

          <div className={styles.controlLabel}>
            <div className={styles.sliderHeader}>
              <span>Opacity: {opacity.toFixed(2)}</span>
              <button
                className={styles.resetBtn}
                onClick={() => onOpacityChange(0.6)}
                title="Reset to 0.60"
                disabled={Math.abs(opacity - 0.6) <= 0.005}
              >
                <ResetIcon />
              </button>
            </div>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={opacity}
              onChange={e => onOpacityChange(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </div>
        </div>
      </details>

      <details className={styles.section} open>
        <summary className={styles.sectionTitle}>Display</summary>
        <div className={styles.sectionBody}>
          <label className={styles.toggle}>
            <input type="checkbox" checked={showAtoms} onChange={e => onShowAtomsChange(e.target.checked)} />
            Show atoms
          </label>

          <label className={styles.toggle}>
            <input type="checkbox" checked={showAtomLabels} onChange={e => onShowAtomLabelsChange(e.target.checked)} />
            Atom labels
          </label>

          <label className={styles.toggle}>
            <input type="checkbox" checked={showAbcCell} onChange={e => onShowAbcCellChange(e.target.checked)} />
            Show abc box
          </label>

          <label className={styles.toggle}>
            <input type="checkbox" checked={showXyzBox} onChange={e => onShowXyzBoxChange(e.target.checked)} />
            Show XYZ box
          </label>

          <label className={styles.toggle}>
            <input type="checkbox" checked={dashedLines} onChange={e => onDashedLinesChange(e.target.checked)} />
            Dashed outlines
          </label>
        </div>
      </details>

      {onTilePaddingChange && (
        <details className={styles.section} open={hasTiling}>
          <summary className={styles.sectionTitle}>Tiling</summary>
          <div className={styles.sectionBody}>
            <label className={styles.controlLabel}>
              <div className={styles.sliderHeader}>
                <span>Padding: {tp.toFixed(2)}</span>
                <button
                  className={styles.resetBtn}
                  onClick={() => onTilePaddingChange(0)}
                  title="Reset to 0"
                  disabled={tp === 0}
                >
                  <ResetIcon />
                </button>
              </div>
              <input
                type="range"
                min={0}
                max={PADDING_TICKS}
                step={1}
                value={paddingToTick(tp)}
                onChange={e => onTilePaddingChange(tickToPadding(parseInt(e.target.value)))}
                className={styles.slider}
              />
            </label>
            {hasTiling && onTileFadeChange && (
              <label className={styles.controlLabel}>
                Fade: {(tileFade ?? 1) === 0 ? 'off' : (tileFade ?? 1).toFixed(1)}
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.5}
                  value={tileFade ?? 1}
                  onChange={e => onTileFadeChange(parseFloat(e.target.value))}
                  className={styles.slider}
                />
              </label>
            )}
          </div>
        </details>
      )}

      <details className={styles.section} open>
        <summary className={styles.sectionTitle}>Camera</summary>
        <div className={styles.sectionBody}>
          <label className={styles.toggle}>
            <input type="checkbox" checked={orbitDeg > 0} onChange={e => onOrbitDegChange(e.target.checked ? 90 : 0)} />
            Orbit step{orbitDeg > 0 && <>:
              <input
                type="number"
                min={1}
                max={360}
                value={orbitDeg}
                onChange={e => { const v = parseInt(e.target.value); if (v > 0) onOrbitDegChange(v) }}
                onClick={e => e.stopPropagation()}
                style={{ width: 32, marginLeft: 4, padding: '0 2px', background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3, fontSize: 12, textAlign: 'right' }}
              /><span style={{ marginLeft: 1 }}>°</span>
            </>}
          </label>
          <label className={styles.toggle}>
            <input type="checkbox" checked={zoomPct > 0} onChange={e => onZoomPctChange(e.target.checked ? 20 : 0)} />
            Zoom step{zoomPct > 0 && <>:
              <input
                type="number"
                min={1}
                max={200}
                value={zoomPct}
                onChange={e => { const v = parseInt(e.target.value); if (v > 0) onZoomPctChange(v) }}
                onClick={e => e.stopPropagation()}
                style={{ width: 32, marginLeft: 4, padding: '0 2px', background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3, fontSize: 12, textAlign: 'right' }}
              /><span style={{ marginLeft: 1 }}>%</span>
            </>}
          </label>
          <label className={styles.toggle}>
            <input type="checkbox" checked={panStep > 0} onChange={e => onPanStepChange(e.target.checked ? 1 : 0)} />
            Pan step{panStep > 0 && <>:
              <input
                type="number"
                min={0.1}
                max={20}
                step={0.1}
                value={panStep}
                onChange={e => { const v = parseFloat(e.target.value); if (v > 0) onPanStepChange(v) }}
                onClick={e => e.stopPropagation()}
                style={{ width: 36, marginLeft: 4, padding: '0 2px', background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3, fontSize: 12, textAlign: 'right' }}
              />
            </>}
          </label>
          <label className={styles.toggle}>
            <input type="checkbox" checked={rollDeg > 0} onChange={e => onRollDegChange(e.target.checked ? 90 : 0)} />
            Roll step{rollDeg > 0 && <>:
              <input
                type="number"
                min={1}
                max={360}
                value={rollDeg}
                onChange={e => { const v = parseInt(e.target.value); if (v > 0) onRollDegChange(v) }}
                onClick={e => e.stopPropagation()}
                style={{ width: 32, marginLeft: 4, padding: '0 2px', background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3, fontSize: 12, textAlign: 'right' }}
              /><span style={{ marginLeft: 1 }}>°</span>
            </>}
          </label>
          <label className={styles.controlLabel}>
            Animation: {animDuration.toFixed(1)}s
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={animDuration}
              onChange={e => onAnimDurationChange(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </label>
          {cam && (
            <div className={styles.camInfo}>
              <span title="Azimuth (theta)">θ {fmtAngle(cam[0])}°</span>
              <span title="Elevation (phi)">φ {fmtAngle(cam[1])}°</span>
              <span title="Distance">d {parseFloat(cam[2].toPrecision(3))}</span>
              {Math.abs(cam[3]) >= 0.05 && <span title="Roll">↻ {fmtAngle(cam[3])}°</span>}
            </div>
          )}
        </div>
      </details>

      <details className={styles.section} open>
        <summary className={styles.sectionTitle}>Slice</summary>
        <div className={styles.sectionBody}>
          <label className={styles.toggle}>
            <input type="checkbox" checked={showSlice} onChange={e => onShowSliceChange(e.target.checked)} />
            2D Slice
          </label>

          {showSlice && (
            <>
              <div className={styles.axisButtons}>
                {AXIS_LABELS.map((label, i) => (
                  <button
                    key={label}
                    className={`${styles.axisBtn} ${sliceAxis === i ? styles.axisBtnActive : ''}`}
                    onClick={() => onSliceAxisChange(i as 0 | 1 | 2)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <label className={styles.controlLabel}>
                Slice: {sliceIndex} / {maxSliceIndex}
                <input
                  type="range"
                  min={0}
                  max={maxSliceIndex}
                  step={1}
                  value={sliceIndex}
                  onChange={e => onSliceIndexChange(parseInt(e.target.value))}
                  className={styles.slider}
                />
              </label>
              <label className={styles.controlLabel}>
                <div className={styles.sliderHeader}>
                  <span>Sweep: {sweepMode === 'd' ? `${sweepDuration.toFixed(1)}s` : `${sliceSpeed} sl/s`}</span>
                  <button
                    className={styles.resetBtn}
                    onClick={e => { e.preventDefault(); onSweepModeToggle() }}
                    title={sweepMode === 'd' ? 'Switch to slices/sec' : 'Switch to duration'}
                  >
                    ↔
                  </button>
                </div>
                {sweepMode === 'd' ? (
                  <input
                    type="range"
                    min={0.5}
                    max={10}
                    step={0.5}
                    value={sweepDuration}
                    onChange={e => onSweepDurationChange(parseFloat(e.target.value))}
                    className={styles.slider}
                  />
                ) : (
                  <input
                    type="range"
                    min={5}
                    max={500}
                    step={5}
                    value={sliceSpeed}
                    onChange={e => onSliceSpeedChange(parseInt(e.target.value))}
                    className={styles.slider}
                  />
                )}
              </label>
            </>
          )}
        </div>
      </details>
    </div>
  )
}
