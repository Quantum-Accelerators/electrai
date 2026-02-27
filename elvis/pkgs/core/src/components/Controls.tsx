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
  showAbcCell: boolean
  onShowAbcCellChange: (v: boolean) => void
  showXyzBox: boolean
  onShowXyzBoxChange: (v: boolean) => void
  showWorldAxes: boolean
  onShowWorldAxesChange: (v: boolean) => void
  orbitDeg: number
  onOrbitDegChange: (v: number) => void
  showSlice: boolean
  onShowSliceChange: (v: boolean) => void
  sliceAxis: 0 | 1 | 2
  onSliceAxisChange: (v: 0 | 1 | 2) => void
  sliceIndex: number
  maxSliceIndex: number
  onSliceIndexChange: (v: number) => void
  animDuration: number
  onAnimDurationChange: (v: number) => void
  sliceSpeed: number
  onSliceSpeedChange: (v: number) => void
  cam: [number, number, number, number] | null
  filename: string
  elements?: string[]
  counts?: number[]
  abcIsXyz?: boolean
}

const AXIS_LABELS = ['X', 'Y', 'Z'] as const

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
  showAbcCell,
  onShowAbcCellChange,
  showXyzBox,
  onShowXyzBoxChange,
  showWorldAxes,
  onShowWorldAxesChange,
  orbitDeg,
  onOrbitDegChange,
  showSlice,
  onShowSliceChange,
  sliceAxis,
  onSliceAxisChange,
  sliceIndex,
  maxSliceIndex,
  onSliceIndexChange,
  animDuration,
  onAnimDurationChange,
  sliceSpeed,
  onSliceSpeedChange,
  cam,
  filename,
  elements,
  counts,
  abcIsXyz,
}: ControlsProps) {
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

          {!abcIsXyz && (
            <label className={styles.toggle}>
              <input type="checkbox" checked={showAbcCell} onChange={e => onShowAbcCellChange(e.target.checked)} />
              Show abc cell
            </label>
          )}

          <label className={styles.toggle}>
            <input type="checkbox" checked={showXyzBox} onChange={e => onShowXyzBoxChange(e.target.checked)} />
            Show XYZ box
          </label>

          <label className={styles.toggle}>
            <input type="checkbox" checked={showWorldAxes} onChange={e => onShowWorldAxesChange(e.target.checked)} />
            Show XYZ axes
          </label>
        </div>
      </details>

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
                Sweep: {sliceSpeed} slices/s
                <input
                  type="range"
                  min={10}
                  max={500}
                  step={10}
                  value={sliceSpeed}
                  onChange={e => onSliceSpeedChange(parseInt(e.target.value))}
                  className={styles.slider}
                />
              </label>
            </>
          )}
        </div>
      </details>
    </div>
  )
}
