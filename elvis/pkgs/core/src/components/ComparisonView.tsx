import { DensityViewer } from './DensityViewer.tsx'
import type { VolumeData } from '../types.ts'
import styles from './ComparisonView.module.css'

interface ComparisonViewProps {
  volumes: Array<{ data: VolumeData; label: string }>
  isoLevel: number
  opacity: number
  showAtoms: boolean
  showUnitCell: boolean
}

export function ComparisonView({ volumes, isoLevel, opacity, showAtoms, showUnitCell }: ComparisonViewProps) {
  return (
    <div className={styles.comparisonGrid}>
      {volumes.map(({ data, label }) => (
        <div key={label} className={styles.comparisonPanel}>
          <DensityViewer
            volume={data}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showUnitCell={showUnitCell}
            label={label}
          />
        </div>
      ))}
    </div>
  )
}
