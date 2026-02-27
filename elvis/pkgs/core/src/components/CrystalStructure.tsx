import { useMemo, useRef, useEffect } from 'react'
import { Object3D, Color, Vector3, InstancedMesh, BufferGeometry, Line as ThreeLine, LineDashedMaterial } from 'three'
import { Html, Billboard, Text } from '@react-three/drei'
import type { VolumeData } from '../types.ts'
import { fracToCart, unitCellEdges, unitCellBoundingBox } from '../utils/lattice.ts'
import { getElement } from '../utils/elements.ts'
import { atomOpacity } from '../utils/tiling.ts'
import type { TileInfo } from '../utils/tiling.ts'
import { tileFadeCompile } from '../utils/tile-fade.ts'

interface CrystalStructureProps {
  volume: VolumeData
  showAtoms: boolean
  showAtomLabels: boolean
  showAbcCell: boolean
  showXyzBox: boolean
  showWorldAxes: boolean
  dashedLines: boolean
  lineWidth?: number
  tiles?: TileInfo[]
  tilePadding?: number
  tileFade?: boolean
}

// Lattice vector colors — YOV (yellow, orange, violet), warm complement of gizmo RGB
const LATTICE_COLORS = ['#ffcc00', '#ff8822', '#aa55ff'] as const
const LATTICE_LABELS = ['a', 'b', 'c'] as const

// World axis colors — match gizmo RGB
const WORLD_COLORS = ['#ff3653', '#0adb50', '#2c8fff'] as const
const WORLD_LABELS = ['X', 'Y', 'Z'] as const

function AxisCylinder({ from, to, color, radius = 0.06, opacity = 1, fadeCompile, fadeKey }: {
  from: [number, number, number]
  to: [number, number, number]
  color: string
  radius?: number
  opacity?: number
  fadeCompile?: (shader: any) => void
  fadeKey?: string
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

  const transparent = opacity < 1 || !!fadeCompile
  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[radius, radius, length, 6]} />
      <meshStandardMaterial key={fadeKey} color={color} transparent={transparent} opacity={opacity} depthWrite={!transparent} onBeforeCompile={fadeCompile} />
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

function DashedEdge({ from, to, color, opacity = 1, fadeCompile }: {
  from: [number, number, number]
  to: [number, number, number]
  color: string
  opacity?: number
  fadeCompile?: (shader: any) => void
}) {
  const groupRef = useRef<Object3D>(null!)
  const line = useMemo(() => {
    const geo = new BufferGeometry().setFromPoints([new Vector3(...from), new Vector3(...to)])
    const transparent = opacity < 1 || !!fadeCompile
    const mat = new LineDashedMaterial({ color, dashSize: 0.15, gapSize: 0.08, transparent, opacity })
    if (fadeCompile) mat.onBeforeCompile = fadeCompile
    const l = new ThreeLine(geo, mat)
    l.computeLineDistances()
    return l
  }, [from, to, color, opacity, fadeCompile])
  useEffect(() => {
    groupRef.current.add(line)
    return () => { groupRef.current?.remove(line) }
  }, [line])
  return <group ref={groupRef} />
}

/** Quantize opacity to 0.05 steps to reduce instanced mesh count */
function quantizeOpacity(o: number): number {
  return Math.round(o * 20) / 20
}

export function CrystalStructure({ volume, showAtoms, showAtomLabels, showAbcCell, showXyzBox, showWorldAxes, dashedLines, lineWidth = 1, tiles, tilePadding = 0, tileFade = true }: CrystalStructureProps) {
  const { lattice, structure } = volume

  // Group atoms by (element, quantized opacity) for tiled instanced rendering
  // When tilePadding > 0, per-atom opacity is computed from fractional-coordinate distance
  const { atomGroups, primaryAtomGroups } = useMemo(() => {
    const groups = new Map<string, Array<[number, number, number]>>()
    const primaryGroups = new Map<string, Array<[number, number, number]>>()
    const tileList = tiles ?? [{ fracOffset: [0, 0, 0] as [number, number, number], cartOffset: [0, 0, 0] as [number, number, number], opacity: 1, isPrimary: true }]

    for (const tile of tileList) {
      if (tile.opacity <= 0) continue
      for (const atom of structure.atoms) {
        // Compute per-atom opacity from actual fractional position
        const fracPos: [number, number, number] = [
          atom.fracCoords[0] + tile.fracOffset[0],
          atom.fracCoords[1] + tile.fracOffset[1],
          atom.fracCoords[2] + tile.fracOffset[2],
        ]
        const opacity = tile.isPrimary ? 1 : atomOpacity(fracPos, tilePadding, tileFade)
        if (opacity <= 0) continue

        const qo = quantizeOpacity(opacity)
        const basePos = fracToCart(lattice, atom.fracCoords)
        const pos: [number, number, number] = [
          basePos[0] + tile.cartOffset[0],
          basePos[1] + tile.cartOffset[1],
          basePos[2] + tile.cartOffset[2],
        ]
        const key = `${atom.element}:${qo}`
        const arr = groups.get(key) ?? []
        arr.push(pos)
        groups.set(key, arr)

        if (tile.isPrimary) {
          const pArr = primaryGroups.get(atom.element) ?? []
          pArr.push(pos)
          primaryGroups.set(atom.element, pArr)
        }
      }
    }
    return { atomGroups: groups, primaryAtomGroups: primaryGroups }
  }, [lattice, structure, tiles, tilePadding, tileFade])

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

  // Effective tile list (default: single primary tile)
  const tileList = tiles ?? [{ fracOffset: [0, 0, 0] as [number, number, number], cartOffset: [0, 0, 0] as [number, number, number], opacity: 1, isPrimary: true }]

  const fadeCompile = useMemo(() => {
    if (tilePadding <= 0) return undefined
    return tileFadeCompile(lattice, tilePadding, tileFade)
  }, [lattice, tilePadding, tileFade])
  const fadeKey = tilePadding > 0 ? `fade-${tilePadding}-${tileFade}` : undefined

  return (
    <group>
      {showAtoms && Array.from(atomGroups.entries()).map(([key, positions]) => {
        const [element, opStr] = key.split(':')
        const opacity = parseFloat(opStr)
        return (
          <AtomInstances key={key} element={element} positions={positions} opacity={opacity} />
        )
      })}
      {showAtoms && showAtomLabels && Array.from(primaryAtomGroups.entries()).map(([element, positions]) => {
        const { color, radius } = getElement(element)
        const css = `#${color.toString(16).padStart(6, '0')}`
        return positions.map((pos, i) => (
          <Billboard key={`label-${element}-${i}`} position={[pos[0], pos[1] + radius * 0.4 + 0.15, pos[2]]}>
            <Text fontSize={0.2} color={css} anchorY="bottom" fillOpacity={0.45}>
              {element}
            </Text>
          </Billboard>
        ))
      })}
      {showAbcCell && (
        <>
          {tileList.map((tile, tileIdx) => {
            if (tile.opacity <= 0) return null
            return (
              <group key={`cell-tile-${tileIdx}`} position={tile.cartOffset}>
                {cellEdgesByAxis.map((edges, axisIdx) =>
                  edges.map(([a, b], edgeIdx) => {
                    const ca = fracToCart(lattice, a)
                    const cb = fracToCart(lattice, b)
                    const isFromOrigin = a[0] === 0 && a[1] === 0 && a[2] === 0
                    return dashedLines ? (
                      <DashedEdge
                        key={`cell-${axisIdx}-${edgeIdx}`}
                        from={ca}
                        to={cb}
                        color={LATTICE_COLORS[axisIdx]}
                        opacity={fadeCompile ? 1 : tile.opacity}
                        fadeCompile={fadeCompile}
                      />
                    ) : (
                      <AxisCylinder
                        key={`cell-${axisIdx}-${edgeIdx}`}
                        from={ca}
                        to={cb}
                        color={LATTICE_COLORS[axisIdx]}
                        radius={(isFromOrigin && tile.isPrimary ? 0.06 : 0.03) * lineWidth}
                        opacity={fadeCompile ? 1 : tile.opacity}
                        fadeCompile={fadeCompile}
                        fadeKey={fadeKey}
                      />
                    )
                  })
                )}
              </group>
            )
          })}
          {labelPositions.map((pos, i) => (
            <AxisLabel key={`label-${i}`} position={pos} label={LATTICE_LABELS[i]} color={LATTICE_COLORS[i]} />
          ))}
        </>
      )}
      {showXyzBox && xyzBoxEdges.map((edge, i) => (
        dashedLines ? (
          <DashedEdge
            key={`xyz-box-${i}`}
            from={edge.from}
            to={edge.to}
            color={WORLD_COLORS[edge.axis]}
          />
        ) : (
          <AxisCylinder
            key={`xyz-box-${i}`}
            from={edge.from}
            to={edge.to}
            color={WORLD_COLORS[edge.axis]}
            radius={0.03 * lineWidth}
          />
        )
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

function AtomInstances({ element, positions, opacity = 1 }: { element: string; positions: Array<[number, number, number]>; opacity?: number }) {
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
      <meshStandardMaterial color={new Color(color)} transparent={opacity < 1} opacity={opacity} depthWrite={opacity >= 1} />
    </instancedMesh>
  )
}
