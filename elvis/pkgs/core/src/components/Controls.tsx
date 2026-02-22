import styles from './Controls.module.css'

interface ControlsProps {
  isoLevel: number
  maxDensity: number
  onIsoLevelChange: (v: number) => void
  opacity: number
  onOpacityChange: (v: number) => void
  showAtoms: boolean
  onShowAtomsChange: (v: boolean) => void
  showUnitCell: boolean
  onShowUnitCellChange: (v: boolean) => void
  showWorldAxes: boolean
  onShowWorldAxesChange: (v: boolean) => void
  showSlice: boolean
  onShowSliceChange: (v: boolean) => void
  sliceAxis: 0 | 1 | 2
  onSliceAxisChange: (v: 0 | 1 | 2) => void
  sliceIndex: number
  maxSliceIndex: number
  onSliceIndexChange: (v: number) => void
  filename: string
}

const AXIS_LABELS = ['X', 'Y', 'Z'] as const

export function Controls({
  isoLevel,
  maxDensity,
  onIsoLevelChange,
  opacity,
  onOpacityChange,
  showAtoms,
  onShowAtomsChange,
  showUnitCell,
  onShowUnitCellChange,
  showWorldAxes,
  onShowWorldAxesChange,
  showSlice,
  onShowSliceChange,
  sliceAxis,
  onSliceAxisChange,
  sliceIndex,
  maxSliceIndex,
  onSliceIndexChange,
  filename,
}: ControlsProps) {
  return (
    <div className={styles.controls}>
      <div className={styles.controlTitle}>{filename}</div>

      <label className={styles.controlLabel}>
        Iso-level: {isoLevel.toFixed(1)}
        <input
          type="range"
          min={0}
          max={maxDensity}
          step={maxDensity / 500}
          value={isoLevel}
          onChange={e => onIsoLevelChange(parseFloat(e.target.value))}
          className={styles.slider}
        />
      </label>

      <label className={styles.controlLabel}>
        Opacity: {opacity.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.01}
          value={opacity}
          onChange={e => onOpacityChange(parseFloat(e.target.value))}
          className={styles.slider}
        />
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showAtoms} onChange={e => onShowAtomsChange(e.target.checked)} />
        Show atoms
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showUnitCell} onChange={e => onShowUnitCellChange(e.target.checked)} />
        Show unit cell
      </label>

      <label className={styles.toggle}>
        <input type="checkbox" checked={showWorldAxes} onChange={e => onShowWorldAxesChange(e.target.checked)} />
        Show XYZ axes
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
