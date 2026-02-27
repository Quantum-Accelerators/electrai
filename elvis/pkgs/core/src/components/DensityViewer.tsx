import { useMemo } from 'react'
import type { MutableRefObject, RefObject } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei'
import { Vector3 } from 'three'
import type { VolumeData } from '../types.ts'
import { IsosurfaceRenderer } from './IsosurfaceRenderer.tsx'
import { CrystalStructure } from './CrystalStructure.tsx'
import { LatticeGizmo } from './LatticeGizmo.tsx'
import { ScreenOffsetGroup } from './ScreenOffsetGroup.tsx'
import { CameraController } from './CameraController.tsx'
import type { CameraSnapTarget } from './CameraController.tsx'
import { SlicePlane3D } from './SlicePlane3D.tsx'
import { fracToCart } from '../utils/lattice.ts'
import { computeTiles } from '../utils/tiling.ts'

interface DensityViewerProps {
  volume: VolumeData
  isoLevel: number
  opacity: number
  showAtoms: boolean
  showAtomLabels: boolean
  showAbcCell: boolean
  showXyzBox: boolean
  showWorldAxes: boolean
  dashedLines: boolean
  lineWidth?: number
  activeMovements?: RefObject<Set<string>>
  cameraSnap?: MutableRefObject<CameraSnapTarget | null>
  animationDuration?: number
  onCameraChange?: (theta: number, phi: number, zoom: number, roll: number) => void
  initialCamera?: MutableRefObject<[number, number, number, number] | null>
  showSlice?: boolean
  sliceAxis?: 0 | 1 | 2
  sliceIndex?: number
  label?: string
  tilePadding?: number
  tileFade?: boolean
  abcIsXyz?: boolean
}

export function DensityViewer({
  volume,
  isoLevel,
  opacity,
  showAtoms,
  showAtomLabels,
  showAbcCell,
  showXyzBox,
  showWorldAxes,
  dashedLines,
  lineWidth = 1,
  activeMovements,
  cameraSnap,
  animationDuration,
  onCameraChange,
  initialCamera,
  showSlice,
  sliceAxis,
  sliceIndex,
  label,
  tilePadding = 0,
  tileFade = true,
  abcIsXyz,
}: DensityViewerProps) {
  const tiles = useMemo(() => {
    if (tilePadding <= 0) return undefined
    return computeTiles(volume.lattice, tilePadding, tileFade)
  }, [volume.lattice, tilePadding, tileFade])

  const center = useMemo(() => {
    const c = fracToCart(volume.lattice, [0.5, 0.5, 0.5])
    return new Vector3(...c)
  }, [volume.lattice])

  const cameraPosition = useMemo(() => {
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
        style={{ background: '#000' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 10]} intensity={0.8} />
        <directionalLight position={[-5, -5, 5]} intensity={0.4} />

        <IsosurfaceRenderer volume={volume} isoLevel={isoLevel} opacity={opacity} tiles={tiles} tilePadding={tilePadding} tileFade={tileFade} />
        <CrystalStructure volume={volume} showAtoms={showAtoms} showAtomLabels={showAtomLabels} showAbcCell={showAbcCell} showXyzBox={showXyzBox} showWorldAxes={showWorldAxes} dashedLines={dashedLines} lineWidth={lineWidth} tiles={tiles} tilePadding={tilePadding} tileFade={tileFade} />
        {showSlice && sliceAxis !== undefined && sliceIndex !== undefined && (
          <SlicePlane3D lattice={volume.lattice} axis={sliceAxis} sliceIndex={sliceIndex} dims={volume.grid.dims} data={volume.grid.data} />
        )}

        {activeMovements && <CameraController activeMovements={activeMovements} cameraSnap={cameraSnap} animationDuration={animationDuration} onCameraChange={onCameraChange} initialCamera={initialCamera} />}

        <OrbitControls makeDefault target={center.toArray()} />
        <GizmoHelper alignment="bottom-right" margin={[80, 36]}>
          <GizmoViewport axisHeadScale={0.8} labelColor="white" />
          {!abcIsXyz && (
            <ScreenOffsetGroup offset={[-100, 0, 0]}>
              <group scale={40}>
                <LatticeGizmo lattice={volume.lattice} />
              </group>
            </ScreenOffsetGroup>
          )}
        </GizmoHelper>
      </Canvas>
    </div>
  )
}
