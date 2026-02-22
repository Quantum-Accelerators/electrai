import type { RefObject } from 'react'
import { DensityViewer } from './DensityViewer.tsx'
import type { VolumeData } from '../types.ts'
import styles from './ComparisonView.module.css'

interface ComparisonViewProps {
  volumes: Array<{ data: VolumeData; label: string }>
  isoLevel: number
  opacity: number
  showAtoms: boolean
  showUnitCell: boolean
  showWorldAxes: boolean
  activeMovements?: RefObject<Set<string>>
}

export function ComparisonView({ volumes, isoLevel, opacity, showAtoms, showUnitCell, showWorldAxes, activeMovements }: ComparisonViewProps) {
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
            showWorldAxes={showWorldAxes}
            activeMovements={activeMovements}
            label={label}
          />
        </div>
      ))}
    </div>
  )
}
