import { useMemo, useCallback } from 'react'
import { Vector3, Object3D, CanvasTexture } from 'three'
import { useGizmoContext } from '@react-three/drei'
import type { LatticeMatrix } from '../types.ts'
import { fracToCart } from '../utils/lattice.ts'

const COLORS = ['#ffcc00', '#ff8822', '#aa55ff'] as const
const LABELS = ['a', 'b', 'c'] as const

function makeAxisTexture(color: string, label: string): CanvasTexture {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  ctx.font = 'bold 36px sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#000'
  ctx.fillText(label, size / 2, size / 2 + 2)
  return new CanvasTexture(canvas)
}

function GizmoAxisArrow({ direction, color, texture, onClick }: {
  direction: Vector3
  color: string
  texture: CanvasTexture
  onClick: () => void
}) {
  const AXIS_LEN = 0.8

  const end = useMemo(() => direction.clone().multiplyScalar(AXIS_LEN), [direction])

  const { position, quaternion, length } = useMemo(() => {
    const mid = end.clone().multiplyScalar(0.5)
    const len = end.length()
    const yAxis = new Vector3(0, 1, 0)
    const obj = new Object3D()
    obj.quaternion.setFromUnitVectors(yAxis, direction.clone().normalize())
    return { position: mid, quaternion: obj.quaternion, length: len }
  }, [end, direction])

  return (
    <group>
      <mesh position={position} quaternion={quaternion}>
        <cylinderGeometry args={[0.02, 0.02, length, 6]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <sprite
        position={end.toArray() as [number, number, number]}
        scale={[0.35, 0.35, 0.35]}
        onClick={(e) => { e.stopPropagation(); onClick() }}
      >
        <spriteMaterial map={texture} toneMapped={false} />
      </sprite>
    </group>
  )
}

interface LatticeGizmoProps {
  lattice: LatticeMatrix
}

export function LatticeGizmo({ lattice }: LatticeGizmoProps) {
  const { tweenCamera } = useGizmoContext()

  const axes = useMemo(() =>
    [0, 1, 2].map(i => {
      const frac: [number, number, number] = [0, 0, 0]
      frac[i] = 1
      const cart = fracToCart(lattice, frac)
      return new Vector3(...cart).normalize()
    }),
  [lattice])

  const textures = useMemo(() =>
    LABELS.map((label, i) => makeAxisTexture(COLORS[i], label)),
  [])

  const handleClick = useCallback((i: number) => {
    tweenCamera(axes[i])
  }, [axes, tweenCamera])

  return (
    <group>
      {axes.map((dir, i) => (
        <GizmoAxisArrow
          key={i}
          direction={dir}
          color={COLORS[i]}
          texture={textures[i]}
          onClick={() => handleClick(i)}
        />
      ))}
      <mesh>
        <sphereGeometry args={[0.04, 12, 8]} />
        <meshBasicMaterial color="#888" />
      </mesh>
    </group>
  )
}
