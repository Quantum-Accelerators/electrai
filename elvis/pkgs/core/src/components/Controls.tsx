import { getElement } from '../utils/elements.ts'
import styles from './Controls.module.css'

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
  discreteOrbit: boolean
  onDiscreteOrbitChange: (v: boolean) => void
  showSlice: boolean
  onShowSliceChange: (v: boolean) => void
  sliceAxis: 0 | 1 | 2
  onSliceAxisChange: (v: 0 | 1 | 2) => void
  sliceIndex: number
  maxSliceIndex: number
  onSliceIndexChange: (v: number) => void
  filename: string
  elements?: string[]
}

const AXIS_LABELS = ['X', 'Y', 'Z'] as const

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
  discreteOrbit,
  onDiscreteOrbitChange,
  showSlice,
  onShowSliceChange,
  sliceAxis,
  onSliceAxisChange,
  sliceIndex,
  maxSliceIndex,
  onSliceIndexChange,
  filename,
  elements,
}: ControlsProps) {
  return (
    <div className={styles.controls}>
      <div className={styles.controlTitle}>{filename}</div>
      {elements && elements.length > 0 && (
        <div style={{ display: 'flex', gap: 8, padding: '2px 0 6px', flexWrap: 'wrap' }}>
          {elements.map(el => {
            const { color } = getElement(el)
            return (
              <span key={el} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: '#ccc' }}>
                <span style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: color,
                  display: 'inline-block',
                  flexShrink: 0,
                }} />
                {el}
              </span>
            )
          })}
        </div>
      )}

      <div className={styles.controlLabel}>
        <div className={styles.sliderHeader}>
          <span>Iso-level: {isoLevel.toFixed(1)}</span>
          {Math.abs(isoLevel - defaultIsoLevel) > 0.1 && (
            <button
              className={styles.resetBtn}
              onClick={() => onIsoLevelChange(defaultIsoLevel)}
              title={`Reset to ${defaultIsoLevel.toFixed(1)}`}
            >
              {'\u21ba'}
            </button>
          )}
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
          {Math.abs(opacity - 0.6) > 0.005 && (
            <button
              className={styles.resetBtn}
              onClick={() => onOpacityChange(0.6)}
              title="Reset to 0.60"
            >
              {'\u21ba'}
            </button>
          )}
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

      <label className={styles.toggle}>
        <input type="checkbox" checked={showAtoms} onChange={e => onShowAtomsChange(e.target.checked)} />
        Show atoms
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showAbcCell} onChange={e => onShowAbcCellChange(e.target.checked)} />
        Show abc cell
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showXyzBox} onChange={e => onShowXyzBoxChange(e.target.checked)} />
        Show XYZ box
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showWorldAxes} onChange={e => onShowWorldAxesChange(e.target.checked)} />
        Show XYZ axes
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={discreteOrbit} onChange={e => onDiscreteOrbitChange(e.target.checked)} />
        90° orbit
      </label>

      <hr className={styles.divider} />

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
        </>
      )}
    </div>
  )
}
