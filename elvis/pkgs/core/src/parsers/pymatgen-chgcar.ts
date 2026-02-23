import type { LatticeMatrix, AtomInfo, VolumeData } from '../types.ts'

/**
 * Parse a pymatgen-serialized Chgcar JSON object into VolumeData.
 *
 * Expected JSON structure (from Materials Project S3 bucket .json.gz files):
 *   data.poscar.structure.lattice.matrix → 3×3 lattice (Å)
 *   data.poscar.structure.sites[].species[0].element → element symbol
 *   data.poscar.structure.sites[].abc → fractional coordinates
 *   data.data.total.data → number[Nx][Ny][Nz] charge density grid
 *   task_id → title fallback
 */
export function parsePymatgenChgcar(json: unknown, filename?: string): VolumeData {
  const root = json as Record<string, unknown>
  const data = root.data as Record<string, unknown>
  const poscar = data.poscar as Record<string, unknown>
  const structure = poscar.structure as Record<string, unknown>
  const latticeObj = structure.lattice as Record<string, unknown>
  const matrix = latticeObj.matrix as number[][]

  // Flatten 3×3 row-major → LatticeMatrix (already in Å, no scale factor needed)
  const lattice: LatticeMatrix = [
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
    matrix[2][0], matrix[2][1], matrix[2][2],
  ]

  // Parse sites → atoms, tracking element counts in sequential order
  const sites = structure.sites as Array<Record<string, unknown>>
  const atoms: AtomInfo[] = []
  const elements: string[] = []
  const counts: number[] = []
  for (const site of sites) {
    const species = site.species as Array<Record<string, unknown>>
    const element = species[0].element as string
    const abc = site.abc as [number, number, number]
    atoms.push({ element, fracCoords: [abc[0], abc[1], abc[2]] })
    if (elements.length === 0 || elements[elements.length - 1] !== element) {
      elements.push(element)
      counts.push(1)
    } else {
      counts[counts.length - 1]++
    }
  }

  // Parse 3D grid: total.data is number[Nx][Ny][Nz]
  const dataSection = data.data as Record<string, unknown>
  const total = dataSection.total as Record<string, unknown>
  const gridData = total.data as number[][][]

  const nx = gridData.length
  const ny = gridData[0].length
  const nz = gridData[0][0].length

  // Flatten to Float32Array: flat[x + y*Nx + z*Nx*Ny] = total[x][y][z]
  // This matches VASP order (x varies fastest)
  const flat = new Float32Array(nx * ny * nz)
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        flat[x + y * nx + z * nx * ny] = gridData[x][y][z]
      }
    }
  }

  const title = filename
    ?? (root.task_id as string | undefined)
    ?? 'Pymatgen Chgcar'

  return {
    title,
    scaleFactor: 1.0,
    lattice,
    structure: { elements, counts, atoms },
    grid: { dims: [nx, ny, nz], data: flat },
  }
}
