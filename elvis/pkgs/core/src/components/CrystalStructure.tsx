import { useMemo, useRef, useEffect } from 'react'
import { Object3D, Color, Vector3, InstancedMesh } from 'three'
import { Html } from '@react-three/drei'
import type { VolumeData } from '../types.ts'
import { fracToCart, unitCellEdges, unitCellBoundingBox } from '../utils/lattice.ts'
import { getElement } from '../utils/elements.ts'

interface CrystalStructureProps {
  volume: VolumeData
  showAtoms: boolean
  showAbcCell: boolean
  showXyzBox: boolean
  showWorldAxes: boolean
  lineWidth?: number
}

// Lattice vector colors — YOV (yellow, orange, violet), warm complement of gizmo RGB
const LATTICE_COLORS = ['#ffcc00', '#ff8822', '#aa55ff'] as const
const LATTICE_LABELS = ['a', 'b', 'c'] as const

// World axis colors — match gizmo RGB
const WORLD_COLORS = ['#ff3653', '#0adb50', '#2c8fff'] as const
const WORLD_LABELS = ['X', 'Y', 'Z'] as const

function AxisCylinder({ from, to, color, radius = 0.06 }: {
  from: [number, number, number]
  to: [number, number, number]
  color: string
  radius?: number
}) {
  const { position, quaternion, length } = useMemo(() => {
    const a = new Vector3(...from)
    const b = new Vector3(...to)
    const dir = new Vector3().subVectors(b, a)
    const len = dir.length()
    const mid = new Vector3().addVectors(a, b).multiplyScalar(0.5)
    const yAxis = new Vector3(0, 1, 0)
    const q = new Object3D()
    q.quaternion.setFromUnitVectors(yAxis, dir.normalize())
    return { position: mid, quaternion: q.quaternion, length: len }
  }, [from, to])

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[radius, radius, length, 6]} />
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
    <Html position={position} center zIndexRange={[1, 0]} style={{ pointerEvents: 'none' }}>
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

export function CrystalStructure({ volume, showAtoms, showAbcCell, showXyzBox, showWorldAxes, lineWidth = 1 }: CrystalStructureProps) {
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

  const origin = useMemo(() => fracToCart(lattice, [0, 0, 0]), [lattice])

  // All 12 unit cell edges, grouped by which basis vector they're parallel to
  const cellEdgesByAxis = useMemo(() => {
    const edges = unitCellEdges()
    const grouped: [
      Array<[[number, number, number], [number, number, number]]>,
      Array<[[number, number, number], [number, number, number]]>,
      Array<[[number, number, number], [number, number, number]]>,
    ] = [[], [], []]
    for (const [a, b] of edges) {
      for (let d = 0; d < 3; d++) {
        if (a[d] !== b[d]) {
          grouped[d].push([a, b])
          break
        }
      }
    }
    return grouped
  }, [])

  // Lattice axis endpoints from origin (for labels)
  const axisEndpoints = useMemo(() => [
    fracToCart(lattice, [1, 0, 0]),
    fracToCart(lattice, [0, 1, 0]),
    fracToCart(lattice, [0, 0, 1]),
  ], [lattice])

  // Label positions: slightly past the endpoint
  const labelPositions = useMemo(() =>
    axisEndpoints.map(ep => {
      const dir = new Vector3(...ep).sub(new Vector3(...origin))
      const labelPos = new Vector3(...ep).add(dir.normalize().multiplyScalar(0.5))
      return labelPos.toArray() as [number, number, number]
    }),
  [axisEndpoints, origin])

  // XYZ axis-aligned bounding box edges (12 edges, RGB-colored by axis)
  const xyzBoxEdges = useMemo(() => {
    const { min, max } = unitCellBoundingBox(lattice)
    const corners: [number, number, number][] = []
    for (let i = 0; i <= 1; i++) {
      for (let j = 0; j <= 1; j++) {
        for (let k = 0; k <= 1; k++) {
          corners.push([
            i ? max[0] : min[0],
            j ? max[1] : min[1],
            k ? max[2] : min[2],
          ])
        }
      }
    }
    // 12 edges: pairs of corners differing in exactly one coordinate
    const edges: Array<{ from: [number, number, number]; to: [number, number, number]; axis: number }> = []
    for (let a = 0; a < corners.length; a++) {
      for (let b = a + 1; b < corners.length; b++) {
        let diffAxis = -1
        let diffCount = 0
        for (let d = 0; d < 3; d++) {
          if (corners[a][d] !== corners[b][d]) { diffAxis = d; diffCount++ }
        }
        if (diffCount === 1) {
          edges.push({ from: corners[a], to: corners[b], axis: diffAxis })
        }
      }
    }
    return edges
  }, [lattice])

  // World axes: 3 edges of the XYZ bounding box from its min corner
  const worldAxesData = useMemo(() => {
    const { min, max } = unitCellBoundingBox(lattice)
    const o: [number, number, number] = [min[0], min[1], min[2]]
    const endpoints: [number, number, number][] = [
      [max[0], min[1], min[2]],
      [min[0], max[1], min[2]],
      [min[0], min[1], max[2]],
    ]
    const labels = endpoints.map((ep, i) => {
      const dir = new Vector3(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0)
      const labelPos = new Vector3(...ep).add(dir.multiplyScalar(0.4))
      return labelPos.toArray() as [number, number, number]
    })
    return { origin: o, endpoints, labels }
  }, [lattice])

  return (
    <group>
      {showAtoms && Array.from(atomGroups.entries()).map(([element, positions]) => (
        <AtomInstances key={element} element={element} positions={positions} />
      ))}
      {showAbcCell && (
        <>
          {cellEdgesByAxis.map((edges, axisIdx) =>
            edges.map(([a, b], edgeIdx) => {
              const ca = fracToCart(lattice, a)
              const cb = fracToCart(lattice, b)
              const isFromOrigin = a[0] === 0 && a[1] === 0 && a[2] === 0
              return (
                <AxisCylinder
                  key={`cell-${axisIdx}-${edgeIdx}`}
                  from={ca}
                  to={cb}
                  color={LATTICE_COLORS[axisIdx]}
                  radius={(isFromOrigin ? 0.06 : 0.03) * lineWidth}
                />
              )
            })
          )}
          {labelPositions.map((pos, i) => (
            <AxisLabel key={`label-${i}`} position={pos} label={LATTICE_LABELS[i]} color={LATTICE_COLORS[i]} />
          ))}
        </>
      )}
      {showXyzBox && xyzBoxEdges.map((edge, i) => (
        <AxisCylinder
          key={`xyz-box-${i}`}
          from={edge.from}
          to={edge.to}
          color={WORLD_COLORS[edge.axis]}
          radius={0.03 * lineWidth}
        />
      ))}
      {showWorldAxes && (
        <>
          {worldAxesData.endpoints.map((ep, i) => (
            <AxisCylinder key={`world-${i}`} from={worldAxesData.origin} to={ep} color={WORLD_COLORS[i]} radius={0.06 * lineWidth} />
          ))}
          {worldAxesData.labels.map((pos, i) => (
            <AxisLabel key={`world-label-${i}`} position={pos} label={WORLD_LABELS[i]} color={WORLD_COLORS[i]} />
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
