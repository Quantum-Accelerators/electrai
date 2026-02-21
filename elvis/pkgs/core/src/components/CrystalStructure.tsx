import { useMemo, useRef, useEffect } from 'react'
import { Object3D, Color, Vector3, BufferGeometry, Float32BufferAttribute, LineBasicMaterial, InstancedMesh } from 'three'
import { Html } from '@react-three/drei'
import type { VolumeData } from '../types.ts'
import { fracToCart, unitCellEdges } from '../utils/lattice.ts'
import { getElement } from '../utils/elements.ts'

interface CrystalStructureProps {
  volume: VolumeData
  showAtoms: boolean
  showUnitCell: boolean
}

// Lattice vector colors — warm/distinct tones, intentionally different from gizmo RGB
const LATTICE_COLORS = ['#ff6644', '#44dd44', '#44aaff'] as const
const LATTICE_LABELS = ['a', 'b', 'c'] as const

function AxisCylinder({ from, to, color }: {
  from: [number, number, number]
  to: [number, number, number]
  color: string
}) {
  const { position, quaternion, length } = useMemo(() => {
    const a = new Vector3(...from)
    const b = new Vector3(...to)
    const dir = new Vector3().subVectors(b, a)
    const len = dir.length()
    const mid = new Vector3().addVectors(a, b).multiplyScalar(0.5)
    const quat = new Object3D()
    quat.position.copy(mid)
    quat.lookAt(b)
    // CylinderGeometry is along Y by default, so rotate from Y to dir
    const yAxis = new Vector3(0, 1, 0)
    const q = new Object3D()
    q.quaternion.setFromUnitVectors(yAxis, dir.normalize())
    return { position: mid, quaternion: q.quaternion, length: len }
  }, [from, to])

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[0.06, 0.06, length, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
  )
}

function AxisLabel({ position, label, color }: {
  position: [number, number, number]
  label: string
  color: string
}) {
  return (
    <Html position={position} center style={{ pointerEvents: 'none' }}>
      <span style={{
        color,
        fontSize: 13,
        fontWeight: 700,
        textShadow: '0 0 4px #000, 0 0 2px #000',
        userSelect: 'none',
      }}>
        {label}
      </span>
    </Html>
  )
}

export function CrystalStructure({ volume, showAtoms, showUnitCell }: CrystalStructureProps) {
  const { lattice, structure } = volume

  // Group atoms by element for instanced rendering
  const atomGroups = useMemo(() => {
    const groups = new Map<string, Array<[number, number, number]>>()
    for (const atom of structure.atoms) {
      const arr = groups.get(atom.element) ?? []
      arr.push(fracToCart(lattice, atom.fracCoords))
      groups.set(atom.element, arr)
    }
    return groups
  }, [lattice, structure])

  // Lattice axis endpoints from origin
  const axisEndpoints = useMemo(() => [
    fracToCart(lattice, [1, 0, 0]),
    fracToCart(lattice, [0, 1, 0]),
    fracToCart(lattice, [0, 0, 1]),
  ], [lattice])

  const origin = useMemo(() => fracToCart(lattice, [0, 0, 0]), [lattice])

  // Label positions: slightly past the endpoint
  const labelPositions = useMemo(() =>
    axisEndpoints.map(ep => {
      const dir = new Vector3(...ep).sub(new Vector3(...origin))
      const labelPos = new Vector3(...ep).add(dir.normalize().multiplyScalar(0.5))
      return labelPos.toArray() as [number, number, number]
    }),
  [axisEndpoints, origin])

  // Non-axis edges (all edges except the 3 from origin)
  const remainingGeo = useMemo(() => {
    const edges = unitCellEdges()
    const axisTargets: [number, number, number][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    const positions: number[] = []

    for (const [a, b] of edges) {
      const isOriginA = a[0] === 0 && a[1] === 0 && a[2] === 0
      const isAxisEdge = isOriginA && axisTargets.some(t => t[0] === b[0] && t[1] === b[1] && t[2] === b[2])
      if (!isAxisEdge) {
        const ca = fracToCart(lattice, a)
        const cb = fracToCart(lattice, b)
        positions.push(...ca, ...cb)
      }
    }

    const geo = new BufferGeometry()
    geo.setAttribute('position', new Float32BufferAttribute(new Float32Array(positions), 3))
    return geo
  }, [lattice])

  const cellLineMat = useMemo(() => new LineBasicMaterial({ color: 0xffffff, opacity: 0.35, transparent: true }), [])

  return (
    <group>
      {showAtoms && Array.from(atomGroups.entries()).map(([element, positions]) => (
        <AtomInstances key={element} element={element} positions={positions} />
      ))}
      {showUnitCell && (
        <>
          <lineSegments geometry={remainingGeo} material={cellLineMat} />
          {axisEndpoints.map((ep, i) => (
            <AxisCylinder key={i} from={origin} to={ep} color={LATTICE_COLORS[i]} />
          ))}
          {labelPositions.map((pos, i) => (
            <AxisLabel key={`label-${i}`} position={pos} label={LATTICE_LABELS[i]} color={LATTICE_COLORS[i]} />
          ))}
        </>
      )}
    </group>
  )
}

function AtomInstances({ element, positions }: { element: string; positions: Array<[number, number, number]> }) {
  const meshRef = useRef<InstancedMesh>(null!)
  const { color, radius } = getElement(element)
  const dummy = useMemo(() => new Object3D(), [])

  useEffect(() => {
    for (let i = 0; i < positions.length; i++) {
      dummy.position.set(...positions[i])
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  }, [positions, dummy])

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, positions.length]}>
      <sphereGeometry args={[radius * 0.4, 16, 12]} />
      <meshStandardMaterial color={new Color(color)} />
    </instancedMesh>
  )
}
