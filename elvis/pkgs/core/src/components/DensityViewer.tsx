import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei'
import { Vector3 } from 'three'
import type { VolumeData } from '../types.ts'
import { IsosurfaceRenderer } from './IsosurfaceRenderer.tsx'
import { CrystalStructure } from './CrystalStructure.tsx'
import { fracToCart } from '../utils/lattice.ts'

interface DensityViewerProps {
  volume: VolumeData
  isoLevel: number
  opacity: number
  showAtoms: boolean
  showUnitCell: boolean
  label?: string
}

export function DensityViewer({
  volume,
  isoLevel,
  opacity,
  showAtoms,
  showUnitCell,
  label,
}: DensityViewerProps) {
  const center = useMemo(() => {
    const c = fracToCart(volume.lattice, [0.5, 0.5, 0.5])
    return new Vector3(...c)
  }, [volume.lattice])

  const cameraPosition = useMemo(() => {
    // Position camera relative to the cell center, at a reasonable distance
    const c = fracToCart(volume.lattice, [0.5, 0.5, 0.5])
    return new Vector3(c[0] + 15, c[1] + 10, c[2] + 15)
  }, [volume.lattice])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {label && (
        <div style={{
          position: 'absolute',
          top: 8,
          left: 12,
          color: '#ccc',
          fontSize: 14,
          fontWeight: 600,
          zIndex: 1,
          pointerEvents: 'none',
        }}>
          {label}
        </div>
      )}
      <Canvas
        camera={{ position: cameraPosition.toArray(), fov: 50, near: 0.1, far: 500 }}
        style={{ background: '#1a1a2e' }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 10]} intensity={0.8} />
        <directionalLight position={[-5, -5, 5]} intensity={0.3} />

        <IsosurfaceRenderer volume={volume} isoLevel={isoLevel} opacity={opacity} />
        <CrystalStructure volume={volume} showAtoms={showAtoms} showUnitCell={showUnitCell} />

        <OrbitControls makeDefault target={center.toArray()} />
        <GizmoHelper alignment="bottom-right" margin={[36, 36]}>
          <GizmoViewport axisHeadScale={0.8} labelColor="white" />
        </GizmoHelper>
      </Canvas>
    </div>
  )
}
