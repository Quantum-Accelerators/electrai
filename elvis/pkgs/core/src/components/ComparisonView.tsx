import type { RefObject } from 'react'
import { DensityViewer } from './DensityViewer.tsx'
import type { VolumeData } from '../types.ts'
import styles from './ComparisonView.module.css'

interface ComparisonViewProps {
  volumes: Array<{ data: VolumeData; label: string }>
  isoLevel: number
  opacity: number
  showAtoms: boolean
  showAtomLabels: boolean
  showAbcCell: boolean
  showXyzBox: boolean
  dashedLines: boolean
  activeMovements?: RefObject<Set<string>>
  tilePadding?: number
  tileFade?: number
}

export function ComparisonView({ volumes, isoLevel, opacity, showAtoms, showAtomLabels, showAbcCell, showXyzBox, dashedLines, activeMovements, tilePadding, tileFade }: ComparisonViewProps) {
  return (
    <div className={styles.comparisonGrid}>
      {volumes.map(({ data, label }) => (
        <div key={label} className={styles.comparisonPanel}>
          <DensityViewer
            volume={data}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showAtomLabels={showAtomLabels}
            showAbcCell={showAbcCell}
            showXyzBox={showXyzBox}
            dashedLines={dashedLines}
            activeMovements={activeMovements}
            label={label}
            tilePadding={tilePadding}
            tileFade={tileFade}
          />
        </div>
      ))}
    </div>
  )
}
