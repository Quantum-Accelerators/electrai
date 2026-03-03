import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { VolumeData, StoredVolume, CameraSnapTarget } from '@elvis/core'
import {
  DensityViewer,
  ComparisonView,
  Controls,
  SliceViewer,
  VolumeGallery,
  URLInput,
  AWSCredentialsModal,
  SizeConfirmModal,
  Settings,
  ErrorBoundary,
  parseCHGCAR,
  parsePymatgenChgcar,
  useSettings,
  fracToCart,
} from '@elvis/core'
import { ShortcutsModal, Omnibar, SequenceModal, LookupModal, SpeedDial, ModeIndicator, useAction, useActionPair, useActionTriplet, useArrowGroup, useMode } from 'use-kbd'
import type { SpeedDialAction } from 'use-kbd'
import 'use-kbd/styles.css'
import { useUrlState, floatParam, optFloatParam, boolParam, intParam, optIntParam, stringParam } from 'use-prms'
import type { Param } from 'use-prms'
import { OpfsVolumeStore, isOPFSSupported } from './storage/OpfsVolumeStore.ts'
import { loadCredentials, saveCredentials } from './utils/aws-credentials.ts'
import { fetchVolumeFromUrl, fetchVolumeFromS3, s3UriToHttps, fetchVolumeJsonGz } from './utils/fetch-volume.ts'
import { decompressGzip } from './utils/gzip.ts'
import { SSOAuthFlow } from './components/SSOAuthFlow.tsx'
import type { FetchProgress } from './utils/fetch-volume.ts'
import type { AWSCredentials } from './utils/aws-credentials.ts'
import Tooltip from '@mui/material/Tooltip'
import styles from './App.module.css'

interface LoadedFile {
  data: VolumeData
  filename: string
}

function computeDefaultIsoLevel(data: Float32Array): number {
  let sum = 0
  let sumSq = 0
  const n = data.length
  for (let i = 0; i < n; i++) {
    sum += data[i]
    sumSq += data[i] * data[i]
  }
  const mean = sum / n
  const variance = sumSq / n - mean * mean
  const sigma = Math.sqrt(Math.max(0, variance))
  return mean + 2 * sigma
}

// Bool param defaulting to true (present in URL = disabled)
const boolTrueParam: Param<boolean> = {
  encode: (v) => v ? undefined : '',
  decode: (e) => e === undefined,
}

// Camera state: theta°, phi°, zoom, roll° (roll optional, defaults to 0)
// Encoded as space-separated values: `?c=-90 150.1 23.5 72.8`
// Spaces become `+` in query strings, so: `?c=-90+150.1+23.5+72.8`
type CamState = [number, number, number, number] | null
const camParam: Param<CamState> = {
  encode: (v) => {
    if (!v) return undefined
    const fmtAngle = (n: number) => {
      const s = n.toFixed(1)
      return s.endsWith('.0') ? s.slice(0, -2) : s
    }
    const base = `${fmtAngle(v[0])} ${fmtAngle(v[1])} ${parseFloat(v[2].toPrecision(3))}`
    return (Math.abs(v[3]) < 0.05) ? base : `${base} ${fmtAngle(v[3])}`
  },
  decode: (e) => {
    if (e === undefined) return null
    const parts = e.trim().split(/[\s,]+/).map(Number)
    if (!parts.every(isFinite)) return null
    if (parts.length === 3) return [parts[0], parts[1], parts[2], 0]
    if (parts.length === 4) return parts as [number, number, number, number]
    return null
  },
}

// Camera target offset (pan): `?ct=1.2+0+-0.5` (dx dy dz, spaces → + in URL)
type CamTarget = [number, number, number] | null
const camTargetParam: Param<CamTarget> = {
  encode: (v) => {
    if (!v) return undefined
    const fmt = (n: number) => parseFloat(n.toPrecision(3)).toString()
    return `${fmt(v[0])} ${fmt(v[1])} ${fmt(v[2])}`
  },
  decode: (e) => {
    if (e === undefined) return null
    const parts = e.trim().split(/[\s,]+/).map(Number)
    if (parts.length !== 3 || !parts.every(isFinite)) return null
    return parts as [number, number, number]
  },
}

const opfsStore = isOPFSSupported() ? new OpfsVolumeStore() : null

async function parseBlob(blob: Blob, filename: string): Promise<VolumeData> {
  if (filename.toLowerCase().endsWith('.json.gz')) {
    const buf = await decompressGzip(blob)
    const json = JSON.parse(new TextDecoder().decode(buf))
    return parsePymatgenChgcar(json, filename)
  }
  return parseCHGCAR(await blob.text())
}

type MpSource = 'chgcars' | 'elfcars'
const MP_S3_BUCKET = 's3://materialsproject-parsed'

/** Extract mp-XXXXXX ID from a filename or S3 URI, if present. */
function extractMpId(s: string): string | undefined {
  const m = s.match(/(mp-\d+)/)
  return m?.[1]
}

/** Build the canonical MP S3 URI for a material ID. */
function mpS3Uri(mpId: string, source: MpSource = 'chgcars'): string {
  return `${MP_S3_BUCKET}/${source}/${mpId}.json.gz`
}

const DEFAULT_MP_ID = 'mp-1000020'

interface Example { mpId: string; label: string; source?: MpSource }

const EXAMPLES: Example[] = [
  { mpId: 'mp-1000020', label: 'Fe\u2082Cu\u2082O\u2084 (8 MB)' },
  { mpId: 'mp-1828986', label: 'Na\u2082Al\u2082Si\u2084O\u2081\u2082 (10 MB)' },
  { mpId: 'mp-1000005', label: '17 MB' },
  { mpId: 'mp-1523390', label: 'Au\u2083Li ELF (172 KB)', source: 'elfcars' },
  { mpId: 'mp-1524033', label: 'Ag\u2082Lu ELF (304 KB)', source: 'elfcars' },
  { mpId: 'mp-2049718', label: 'Pt\u2084P\u2088 ELF (519 KB)', source: 'elfcars' },
]

const GithubIcon = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
)

const MuiTooltip = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <Tooltip title={title} arrow>
    <span style={{ display: 'contents' }}>{children}</span>
  </Tooltip>
)

const speedDialActions: SpeedDialAction[] = [
  {
    key: 'github',
    label: 'View source on GitHub',
    icon: <GithubIcon />,
    href: 'https://github.com/Quantum-Accelerators/electrai/tree/elvis/elvis#readme',
  },
]

export default function App() {
  const [files, setFiles] = useState<LoadedFile[]>([])
  const [isoLevel, setIsoLevel] = useUrlState('iso', optFloatParam({ encoding: 'string', decimals: 1 }), { debounce: 300 })
  const [opacity, setOpacity] = useUrlState('op', floatParam({ default: 0.6, encoding: 'string', decimals: 2 }), { debounce: 300 })
  const [showAtoms, setShowAtoms] = useUrlState('ha', boolTrueParam)
  const [showAbcCell, setShowAbcCell] = useUrlState('hc', boolTrueParam)
  const [showXyzBox, setShowXyzBox] = useUrlState('xb', boolParam)
  const [showAtomLabels, setShowAtomLabels] = useUrlState('al', boolParam)
  const [showWorldAxes, setShowWorldAxes] = useUrlState('xa', boolParam)
  const [dashedLines, setDashedLines] = useUrlState('dl', boolParam)
  const [showSlice, setShowSlice] = useUrlState('sl', boolTrueParam)
  const [sliceAxis, setSliceAxis] = useUrlState('sa', intParam(2)) as [0 | 1 | 2, (v: 0 | 1 | 2) => void]
  const [sliceIndex, setSliceIndex] = useUrlState('si', optIntParam, { debounce: 300 })
  const [orbitDeg, setOrbitDeg] = useUrlState('od', intParam(30))
  const [zoomPct, setZoomPct] = useUrlState('zd', intParam(0))
  const [panStep, setPanStep] = useUrlState('pd', floatParam({ default: 0, encoding: 'string', decimals: 1 }))
  const [animDuration, setAnimDuration] = useUrlState('a', floatParam({ default: 0.5, encoding: 'string', decimals: 1 }))
  const [sweepMode, setSweepMode] = useUrlState('sm', stringParam('d'))
  const [sweepDuration, setSweepDuration] = useUrlState('sd', floatParam({ default: 2, encoding: 'string', decimals: 1 }), { debounce: 300 })
  const [sliceSpeed, setSliceSpeed] = useUrlState('ss', intParam(120))
  const [lineWidth, setLineWidth] = useUrlState('lw', floatParam({ default: 1, encoding: 'string', decimals: 1 }))
  const [tilePadding, setTilePadding] = useUrlState('tp', floatParam({ default: 0.5, encoding: 'string', decimals: 2 }), { debounce: 300 })
  const [tileFade, setTileFade] = useUrlState('nf', boolTrueParam)
  const [cam, setCam] = useUrlState('c', camParam)
  const [camTarget, setCamTarget] = useUrlState('ct', camTargetParam, { debounce: 500 })
  const [materialId, setMaterialId] = useUrlState('m', stringParam(DEFAULT_MP_ID), { push: true })
  const [currentVolumeId, setCurrentVolumeIdRaw] = useState<string | null>(
    () => sessionStorage.getItem('elvis-active-volume'),
  )
  const setCurrentVolumeId = useCallback((id: string | null) => {
    setCurrentVolumeIdRaw(id)
    if (id) sessionStorage.setItem('elvis-active-volume', id)
    else sessionStorage.removeItem('elvis-active-volume')
  }, [])
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0)
  const [cachedMpIds, setCachedMpIds] = useState<Set<string>>(new Set())
  const [examplesOpen, setExamplesOpen] = useState(() => {
    return sessionStorage.getItem('elvis-examples-open') === 'true'
  })
  const [urlLoading, setUrlLoading] = useState(false)
  const [fetchStatus, setFetchStatus] = useState<string | null>(null)
  const [awsModalOpen, setAwsModalOpen] = useState(false)
  const [awsCreds, setAwsCreds] = useState<AWSCredentials | null>(loadCredentials)
  const [sizeConfirm, setSizeConfirm] = useState<{
    blob: Blob
    filename: string
    fileSizeMB: number
    meta: Omit<StoredVolume, 'id' | 'addedAt' | 'fileSize'>
    data: VolumeData
  } | null>(null)
  const addFileInputRef = useRef<HTMLInputElement>(null)
  const { settings, update: updateSettings } = useSettings()

  // Derive cached material IDs from OPFS (refreshes when gallery changes)
  useEffect(() => {
    if (!opfsStore) return
    opfsStore.list().then(volumes => {
      const ids = new Set<string>()
      for (const v of volumes) {
        const mpId = extractMpId(v.filename)
        if (mpId) ids.add(mpId)
      }
      setCachedMpIds(ids)
    })
  }, [galleryRefreshKey])

  // Camera movement + snap state (shared with CameraController inside Canvas)
  const activeMovements = useRef(new Set<string>())
  const cameraSnap = useRef<CameraSnapTarget | null>(null)
  const initialCamera = useRef<CamState>(cam)
  const initialTargetOffset = useRef<CamTarget>(camTarget)

  const startMovement = useCallback((dir: string) => {
    activeMovements.current.add(dir)
  }, [])

  const handleCameraChange = useCallback((theta: number, phi: number, zoom: number, roll: number, targetOffset?: [number, number, number]) => {
    setCam([theta, phi, zoom, roll])
    setCamTarget(targetOffset ?? null)
  }, [setCam, setCamTarget])

  const MOVEMENT_KEYS = useMemo(() => new Set([
    'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
    '-', '=', '[', ']', 'Shift',
  ]), [])

  useEffect(() => {
    const onKeyUp = (e: KeyboardEvent) => {
      if (MOVEMENT_KEYS.has(e.key)) {
        activeMovements.current.clear()
      }
    }
    window.addEventListener('keyup', onKeyUp)
    return () => window.removeEventListener('keyup', onKeyUp)
  }, [MOVEMENT_KEYS])

  const primaryFile = files[0] ?? null

  const maxSliceIndex = useMemo(() => {
    if (!primaryFile) return 0
    return primaryFile.data.grid.dims[sliceAxis] - 1
  }, [primaryFile, sliceAxis])

  // Refs for values used in useActionPair/useArrowGroup handlers,
  // which have stale closures due to useMemo in those hooks
  const sliceIndexRef = useRef(sliceIndex ?? 0)
  sliceIndexRef.current = sliceIndex ?? 0
  const maxSliceIndexRef = useRef(maxSliceIndex)
  maxSliceIndexRef.current = maxSliceIndex
  const orbitDegRef = useRef(orbitDeg)
  orbitDegRef.current = orbitDeg
  const panStepRef = useRef(panStep)
  panStepRef.current = panStep
  const zoomPctRef = useRef(zoomPct)
  zoomPctRef.current = zoomPct

  const handleSliceAxisChange = useCallback((axis: 0 | 1 | 2) => {
    setSliceAxis(axis)
    if (primaryFile) setSliceIndex(Math.floor(primaryFile.data.grid.dims[axis] / 2))
  }, [setSliceAxis, setSliceIndex, primaryFile])

  const handleSweepModeToggle = useCallback(() => {
    if (sweepMode === 'd') {
      sessionStorage.setItem('elvis-sweep-d', String(sweepDuration))
      const stored = sessionStorage.getItem('elvis-sweep-r')
      setSweepMode('r')
      setSweepDuration(2)
      setSliceSpeed(stored ? parseInt(stored) : 120)
    } else {
      sessionStorage.setItem('elvis-sweep-r', String(sliceSpeed))
      const stored = sessionStorage.getItem('elvis-sweep-d')
      setSweepMode('d')
      setSliceSpeed(120)
      setSweepDuration(stored ? parseFloat(stored) : 2)
    }
  }, [sweepMode, sweepDuration, sliceSpeed, setSweepMode, setSweepDuration, setSliceSpeed])

  // Keyboard modes: s/o/p toggle slice/orbit/pan, remapping arrows to each mode's primary actions
  useMode('mode:slice', {
    label: 'Slice',
    color: '#ff9800',
    defaultBindings: ['s'],
    onActivate: () => setShowSlice(true),
  })
  useMode('mode:orbit', {
    label: 'Orbit',
    color: '#4fc3f7',
    defaultBindings: ['o'],
  })
  useMode('mode:pan', {
    label: 'Pan',
    color: '#66bb6a',
    defaultBindings: ['p'],
  })

  // Lattice directions for abc camera snaps (computed from loaded file)
  const latticeDirections = useMemo(() => {
    if (!primaryFile) return null
    const lat = primaryFile.data.lattice
    return {
      a: fracToCart(lat, [1, 0, 0]) as [number, number, number],
      b: fracToCart(lat, [0, 1, 0]) as [number, number, number],
      c: fracToCart(lat, [0, 0, 1]) as [number, number, number],
    }
  }, [primaryFile])

  // Compute sign so arrow-left/right match visual screen direction for slice stepping.
  // Dot the slice-axis world direction with the camera's screen-right vector; if positive,
  // increasing index moves rightward on screen (ArrowRight → +si), else flip.
  const sliceStepSign = useMemo(() => {
    if (!primaryFile || !cam) return 1
    const lat = primaryFile.data.lattice
    const unitVec: [number, number, number] = [0, 0, 0]
    unitVec[sliceAxis] = 1
    const sliceDir = fracToCart(lat, unitVec)
    const theta = cam[0] * Math.PI / 180
    const phi = cam[1] * Math.PI / 180
    const roll = cam[3] * Math.PI / 180
    // Camera right (before roll) = (cosθ, 0, -sinθ)
    const rx = Math.cos(theta), rz = -Math.sin(theta)
    // Camera up (before roll) = (-sinθ cosφ, sinφ, -cosθ cosφ)
    const ux = -Math.sin(theta) * Math.cos(phi)
    const uy = Math.sin(phi)
    const uz = -Math.cos(theta) * Math.cos(phi)
    // After roll: right_rolled = right*cos(ρ) − up*sin(ρ)
    // (camera right = cross(viewDir, cameraUp); the − follows from expanding cameraUp
    //  = worldUp_perp·cos(ρ) + right_noroll·sin(ρ) and using cross identities)
    const cr = Math.cos(roll), sr = Math.sin(roll)
    const rrx = rx * cr - ux * sr
    const rry = -uy * sr  // right_y before roll is 0
    const rrz = rz * cr - uz * sr
    const dot = sliceDir[0] * rrx + sliceDir[1] * rry + sliceDir[2] * rrz
    return dot >= 0 ? 1 : -1
  }, [primaryFile, cam, sliceAxis])
  const sliceStepSignRef = useRef(sliceStepSign)
  sliceStepSignRef.current = sliceStepSign

  // Detect axis-aligned (orthogonal) lattice: abc ≡ xyz when off-diagonal elements are ~0
  const abcIsXyz = useMemo(() => {
    if (!primaryFile) return false
    const lat = primaryFile.data.lattice
    // lat is row-major [a0,a1,a2, b0,b1,b2, c0,c1,c2]
    // Off-diagonal: a1,a2, b0,b2, c0,c1
    const eps = 1e-6
    return Math.abs(lat[1]) < eps && Math.abs(lat[2]) < eps &&
           Math.abs(lat[3]) < eps && Math.abs(lat[5]) < eps &&
           Math.abs(lat[6]) < eps && Math.abs(lat[7]) < eps
  }, [primaryFile])

  // Auto-hide abc cell when lattice is axis-aligned (abc ≡ xyz)
  useEffect(() => {
    if (abcIsXyz && showAbcCell) {
      setShowAbcCell(false)
      if (!showXyzBox) setShowXyzBox(true)
    }
  }, [abcIsXyz]) // eslint-disable-line react-hooks/exhaustive-deps

  const snapCamera = useCallback((snap: CameraSnapTarget) => {
    cameraSnap.current = snap
  }, [])

  // View toggles: single-letter where free, `t _` sequence where letter conflicts
  useAction('view:toggle-xyz', {
    label: 'Toggle XYZ (axes + box)',
    description: 'Show/hide XYZ coordinate axes and bounding box',
    group: 'View',
    defaultBindings: ['t x'],
    handler: () => {
      const anyOn = showXyzBox || showWorldAxes
      setShowXyzBox(!anyOn)
      setShowWorldAxes(!anyOn)
    },
  })
  useAction('view:toggle-xyz-axes', {
    label: 'Toggle XYZ axes',
    description: 'Show/hide XYZ coordinate axes only',
    group: 'View',
    defaultBindings: ['t shift+x'],
    handler: () => setShowWorldAxes(!showWorldAxes),
  })
  useAction('view:toggle-abc', {
    label: 'Toggle abc (cell + atoms)',
    description: 'Show/hide lattice cell and atoms together',
    group: 'View',
    defaultBindings: ['t a'],
    enabled: !abcIsXyz,
    handler: () => {
      const anyOn = showAbcCell || showAtoms
      setShowAbcCell(!anyOn)
      setShowAtoms(!anyOn)
    },
  })
  useAction('view:toggle-abc-atoms', {
    label: 'Toggle atoms',
    description: 'Show/hide atom spheres',
    group: 'View',
    defaultBindings: ['h'],
    handler: () => setShowAtoms(!showAtoms),
  })
  useAction('view:toggle-labels', {
    label: 'Toggle atom labels',
    description: 'Show/hide element labels on atoms',
    group: 'View',
    defaultBindings: ['l'],
    handler: () => setShowAtomLabels(!showAtomLabels),
  })
  useAction('view:toggle-dashed', {
    label: 'Toggle dashed outlines',
    description: 'Switch between dashed and solid cell outlines',
    group: 'View',
    defaultBindings: ['d'],
    handler: () => setDashedLines(!dashedLines),
  })
  useAction('view:toggle-slice', {
    label: 'Toggle 2D slice',
    description: 'Show/hide 2D density slice plane',
    group: 'View',
    defaultBindings: ['t s'],
    handler: () => setShowSlice(!showSlice),
  })
  useAction('view:toggle-tiling', {
    label: 'Toggle tiling',
    description: 'Toggle periodic tiling of unit cell',
    group: 'View',
    defaultBindings: ['g'],
    handler: () => setTilePadding(tilePadding > 0 ? 0 : 1),
  })
  useAction('view:set-tile-padding', {
    label: 'Set tile padding',
    description: 'Gap between periodic tile copies',
    keywords: ['tiling', 'padding', 'periodic'],
    group: 'View',
    defaultBindings: ['\\f t'],
    handler: (_e, captures) => setTilePadding(captures?.[0] ?? 1),
  })
  useAction('view:toggle-tile-fade', {
    label: 'Toggle tile fade',
    description: 'Fade periodic tile copies',
    keywords: ['tiling', 'fade', 'opacity'],
    group: 'View',
    defaultBindings: ['f'],
    handler: () => setTileFade(!tileFade),
  })
  useActionPair('iso:step', {
    label: 'Decrease / increase iso level',
    description: 'Adjust isosurface threshold by keyboard step',
    keywords: ['iso', 'isosurface', 'threshold', 'density'],
    group: 'Surface',
    actions: [
      {
        defaultBindings: [','],
        handler: () => {
          const step = maxDensityRef.current / 50
          setIsoLevel(Math.max(0, isoLevelRef.current - step))
        },
      },
      {
        defaultBindings: ['.'],
        handler: () => {
          const step = maxDensityRef.current / 50
          setIsoLevel(Math.min(maxDensityRef.current, isoLevelRef.current + step))
        },
      },
    ],
  })
  useAction('iso:set', {
    label: 'Set iso level',
    description: 'Set isosurface threshold to a specific value',
    keywords: ['iso', 'isosurface', 'threshold'],
    group: 'Surface',
    defaultBindings: ['\\f i'],
    handler: (_e, captures) => {
      const v = captures?.[0] ?? 0
      setIsoLevel(Math.max(0, Math.min(v, maxDensityRef.current)))
    },
  })
  useAction('iso:reset', {
    label: 'Reset iso level',
    description: 'Reset isosurface threshold to default (mean + 2\u03c3)',
    keywords: ['iso', 'isosurface', 'reset', 'default'],
    group: 'Surface',
    defaultBindings: ['t i'],
    handler: () => setIsoLevel(defaultIsoLevelRef.current),
  })
  useActionPair('view:orbit-step', {
    label: 'Orbit step: set / toggle',
    description: 'Discrete angle steps (0 = smooth continuous)',
    keywords: ['deg', '90deg', '90', 'discrete', 'step', 'angle'],
    group: 'View',
    actions: [
      { defaultBindings: ['\\d+ o'], handler: (_e, captures) => setOrbitDeg(captures?.[0] ?? 90) },
      { defaultBindings: ['t o'], handler: () => setOrbitDeg(orbitDeg > 0 ? 0 : 90) },
    ],
  })
  useActionPair('view:pan-step', {
    label: 'Pan step: set / toggle',
    description: 'Fixed-distance steps (0 = smooth continuous)',
    keywords: ['discrete', 'step', 'pan'],
    group: 'View',
    actions: [
      { defaultBindings: ['\\d+ p'], handler: (_e, captures) => setPanStep(captures?.[0] ?? 1) },
      { defaultBindings: ['t p'], handler: () => setPanStep(panStep > 0 ? 0 : 1) },
    ],
  })
  useActionPair('view:zoom-step', {
    label: 'Zoom step: set / toggle',
    description: 'Discrete percentage steps (0 = smooth continuous)',
    keywords: ['discrete', 'step', 'zoom'],
    group: 'View',
    actions: [
      { defaultBindings: ['\\d+ z'], handler: (_e, captures) => setZoomPct(captures?.[0] ?? 20) },
      { defaultBindings: ['t z'], handler: () => setZoomPct(zoomPct > 0 ? 0 : 20) },
    ],
  })
  useAction('view:set-anim-speed', {
    label: 'Set animation speed',
    description: 'Camera snap animation duration (seconds)',
    keywords: ['duration', 'animation', 'speed'],
    group: 'View',
    defaultBindings: ['\\f s'],
    handler: (_e, captures) => setAnimDuration(captures?.[0] ?? 0.5),
  })
  useAction('view:set-slice-speed', {
    label: 'Set slice speed',
    description: 'Slice sweep rate (slices/sec)',
    keywords: ['slice', 'animation', 'speed', 'sweep'],
    group: 'View',
    defaultBindings: ['\\d+ shift+s'],
    handler: (_e, captures) => setSliceSpeed(captures?.[0] ?? 120),
  })
  useActionTriplet('slice:axis', {
    label: 'Slice along X / Y / Z',
    description: 'Set slice plane perpendicular to axis',
    group: 'Slice',
    mode: 'mode:slice',
    actions: [
      { defaultBindings: ['x'], handler: () => handleSliceAxisChange(0) },
      { defaultBindings: ['y'], handler: () => handleSliceAxisChange(1) },
      { defaultBindings: ['z'], handler: () => handleSliceAxisChange(2) },
    ],
  })

  // Slice mode: arrows step index, up/down change axis
  useActionPair('mode:slice:step', {
    label: 'Step slice back / forward',
    description: 'Move slice plane one grid step',
    group: 'Slice',
    mode: 'mode:slice',
    actions: [
      {
        defaultBindings: ['arrowleft'],
        handler: () => { const s = sliceStepSignRef.current; cancelSliceAnim(); setSliceIndex(Math.max(0, Math.min(maxSliceIndexRef.current, sliceIndexRef.current - s))) },
      },
      {
        defaultBindings: ['arrowright'],
        handler: () => { const s = sliceStepSignRef.current; cancelSliceAnim(); setSliceIndex(Math.max(0, Math.min(maxSliceIndexRef.current, sliceIndexRef.current + s))) },
      },
    ],
  })
  useActionPair('mode:slice:step10', {
    label: 'Step slice back / forward \u00d710',
    description: 'Move slice plane ten grid steps',
    group: 'Slice',
    mode: 'mode:slice',
    actions: [
      {
        defaultBindings: ['shift+arrowleft'],
        handler: () => { const s = sliceStepSignRef.current * 10; cancelSliceAnim(); setSliceIndex(Math.max(0, Math.min(maxSliceIndexRef.current, sliceIndexRef.current - s))) },
      },
      {
        defaultBindings: ['shift+arrowright'],
        handler: () => { const s = sliceStepSignRef.current * 10; cancelSliceAnim(); setSliceIndex(Math.max(0, Math.min(maxSliceIndexRef.current, sliceIndexRef.current + s))) },
      },
    ],
  })
  useActionPair('mode:slice:axis', {
    label: 'Prev / next slice axis',
    description: 'Cycle between X/Y/Z slice axes',
    group: 'Slice',
    mode: 'mode:slice',
    actions: [
      {
        defaultBindings: ['arrowdown'],
        handler: () => handleSliceAxisChange(((sliceAxis + 2) % 3) as 0 | 1 | 2),
      },
      {
        defaultBindings: ['arrowup'],
        handler: () => handleSliceAxisChange(((sliceAxis + 1) % 3) as 0 | 1 | 2),
      },
    ],
  })

  // Pan mode: unmodified arrows pan
  useActionPair('mode:pan:horizontal', {
    label: 'Pan left / right',
    description: 'Move camera sideways in pan mode',
    group: 'Camera',
    mode: 'mode:pan',
    actions: [
      {
        defaultBindings: ['arrowleft'],
        handler: (e) => { if (e?.repeat) return; startMovement('pan-left'); if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'left', distance: panStepRef.current }) },
      },
      {
        defaultBindings: ['arrowright'],
        handler: (e) => { if (e?.repeat) return; startMovement('pan-right'); if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'right', distance: panStepRef.current }) },
      },
    ],
  })
  useActionPair('mode:pan:vertical', {
    label: 'Pan up / down',
    description: 'Move camera vertically in pan mode',
    group: 'Camera',
    mode: 'mode:pan',
    actions: [
      {
        defaultBindings: ['arrowup'],
        handler: (e) => { if (e?.repeat) return; startMovement('pan-up'); if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'up', distance: panStepRef.current }) },
      },
      {
        defaultBindings: ['arrowdown'],
        handler: (e) => { if (e?.repeat) return; startMovement('pan-down'); if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'down', distance: panStepRef.current }) },
      },
    ],
  })

  // Slice animation: sweep sliceIndex to start or end
  const sliceAnimRef = useRef<{ target: number; raf: number } | null>(null)
  const sliceDirectionRef = useRef<'forward' | 'backward'>('forward')
  const cancelSliceAnim = useCallback(() => {
    if (sliceAnimRef.current) {
      cancelAnimationFrame(sliceAnimRef.current.raf)
      sliceAnimRef.current = null
    }
  }, [])

  const animateSliceTo = useCallback((target: number) => {
    cancelSliceAnim()
    const totalSlices = maxSliceIndex + 1
    const effectiveSlicesPerSec = sweepMode === 'd' ? totalSlices / sweepDuration : sliceSpeed
    let last = performance.now()
    let current = sliceIndex ?? 0
    let fractional = 0
    const step = target > current ? 1 : -1
    const tick = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      fractional += effectiveSlicesPerSec * dt
      const advance = Math.floor(fractional)
      if (advance < 1) {
        sliceAnimRef.current = { target, raf: requestAnimationFrame(tick) }
        return
      }
      fractional -= advance
      const remaining = Math.abs(target - current)
      if (remaining <= 0) { sliceAnimRef.current = null; return }
      const move = Math.min(advance, remaining)
      current += step * move
      setSliceIndex(current)
      if (current === target) { sliceAnimRef.current = null; return }
      sliceAnimRef.current = { target, raf: requestAnimationFrame(tick) }
    }
    sliceAnimRef.current = { target, raf: requestAnimationFrame(tick) }
  }, [sliceIndex, sliceSpeed, sweepMode, sweepDuration, maxSliceIndex, cancelSliceAnim, setSliceIndex])

  // Ref so useActionPair handlers can access the latest animateSliceTo
  const animateSliceToRef = useRef(animateSliceTo)
  animateSliceToRef.current = animateSliceTo

  useActionPair('slice:animate', {
    label: 'Animate slice to start / end',
    description: 'Sweep slice to boundary at current speed',
    group: 'Slice',
    actions: [
      {
        defaultBindings: ['meta+arrowleft'],
        handler: () => { const s = sliceStepSignRef.current; const tgt = s > 0 ? 0 : maxSliceIndexRef.current; sliceDirectionRef.current = s > 0 ? 'backward' : 'forward'; setShowSlice(true); animateSliceToRef.current(tgt) },
      },
      {
        defaultBindings: ['meta+arrowright'],
        handler: () => { const s = sliceStepSignRef.current; const tgt = s > 0 ? maxSliceIndexRef.current : 0; sliceDirectionRef.current = s > 0 ? 'forward' : 'backward'; setShowSlice(true); animateSliceToRef.current(tgt) },
      },
    ],
  })
  useAction('slice:play-pause', {
    label: 'Play/pause slice',
    description: 'Start/stop slice sweep animation',
    group: 'Slice',
    defaultBindings: ['space'],
    handler: () => {
      if (sliceAnimRef.current) {
        cancelSliceAnim()
      } else {
        setShowSlice(true)
        const si = sliceIndex ?? 0
        if (si >= maxSliceIndex && sliceDirectionRef.current === 'forward') {
          sliceDirectionRef.current = 'backward'
        } else if (si <= 0 && sliceDirectionRef.current === 'backward') {
          sliceDirectionRef.current = 'forward'
        }
        animateSliceTo(sliceDirectionRef.current === 'forward' ? maxSliceIndex : 0)
      }
    },
  })
  useActionPair('slice:jump', {
    label: 'Jump slice to start / end',
    description: 'Instantly move slice to boundary',
    group: 'Slice',
    actions: [
      {
        defaultBindings: ['meta+shift+arrowleft'],
        handler: () => { const tgt = sliceStepSignRef.current > 0 ? 0 : maxSliceIndexRef.current; cancelSliceAnim(); setShowSlice(true); setSliceIndex(tgt) },
      },
      {
        defaultBindings: ['meta+shift+arrowright'],
        handler: () => { const tgt = sliceStepSignRef.current > 0 ? maxSliceIndexRef.current : 0; cancelSliceAnim(); setShowSlice(true); setSliceIndex(tgt) },
      },
    ],
  })

  // Camera axis-snap (look down lattice vectors or world axes)
  useActionTriplet('cam:snap-abc', {
    label: 'Look down a / b / c',
    description: 'Point camera along lattice vector',
    group: 'Axes',
    enabled: !abcIsXyz,
    actions: [
      { defaultBindings: ['a'], handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.a }) } },
      { defaultBindings: ['b'], handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.b }) } },
      { defaultBindings: ['c'], handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.c }) } },
    ],
  })
  useActionTriplet('cam:snap-xyz', {
    label: 'Look down X / Y / Z',
    description: 'Point camera along axis',
    group: 'Axes',
    actions: [
      { defaultBindings: ['x'], handler: () => snapCamera({ type: 'look-down', direction: [1, 0, 0] }) },
      { defaultBindings: ['y'], handler: () => snapCamera({ type: 'look-down', direction: [0, 1, 0] }) },
      { defaultBindings: ['z'], handler: () => snapCamera({ type: 'look-down', direction: [0, 0, 1] }) },
    ],
  })
  useActionTriplet('cam:align-abc', {
    label: 'Align a / b / c up',
    description: 'Roll camera so lattice vector points up',
    group: 'Axes',
    enabled: !abcIsXyz,
    actions: [
      { defaultBindings: ['shift+a'], handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.a }) } },
      { defaultBindings: ['shift+b'], handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.b }) } },
      { defaultBindings: ['shift+c'], handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.c }) } },
    ],
  })
  useActionTriplet('cam:align-xyz', {
    label: 'Align X / Y / Z up',
    description: 'Roll camera so axis points up',
    group: 'Axes',
    actions: [
      { defaultBindings: ['shift+x'], handler: () => snapCamera({ type: 'align-up', axis: [1, 0, 0] }) },
      { defaultBindings: ['shift+y'], handler: () => snapCamera({ type: 'align-up', axis: [0, 1, 0] }) },
      { defaultBindings: ['shift+z'], handler: () => snapCamera({ type: 'align-up', axis: [0, 0, 1] }) },
    ],
  })

  // Camera navigation (orbit: continuous or discrete step snaps)
  // In discrete mode, startMovement tracks held state so CameraController can chain snaps
  useArrowGroup('nav:orbit', {
    label: 'Orbit',
    description: 'Rotate camera around crystal',
    group: 'Camera',
    defaultModifiers: [],
    handlers: {
      left:  (e) => { if (e?.repeat) return; startMovement('orbit-left');  if (orbitDegRef.current > 0) snapCamera({ type: 'orbit-step', direction: 'left',  degrees: orbitDegRef.current }) },
      right: (e) => { if (e?.repeat) return; startMovement('orbit-right'); if (orbitDegRef.current > 0) snapCamera({ type: 'orbit-step', direction: 'right', degrees: orbitDegRef.current }) },
      up:    (e) => { if (e?.repeat) return; startMovement('orbit-up');    if (orbitDegRef.current > 0) snapCamera({ type: 'orbit-step', direction: 'up',    degrees: orbitDegRef.current }) },
      down:  (e) => { if (e?.repeat) return; startMovement('orbit-down');  if (orbitDegRef.current > 0) snapCamera({ type: 'orbit-step', direction: 'down',  degrees: orbitDegRef.current }) },
    },
  })
  useArrowGroup('nav:pan', {
    label: 'Pan',
    description: 'Shift camera position without rotating',
    group: 'Camera',
    defaultModifiers: ['shift'],
    handlers: {
      left:  (e) => { if (e?.repeat) return; startMovement('pan-left');  if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'left',  distance: panStepRef.current }) },
      right: (e) => { if (e?.repeat) return; startMovement('pan-right'); if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'right', distance: panStepRef.current }) },
      up:    (e) => { if (e?.repeat) return; startMovement('pan-up');    if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'up',    distance: panStepRef.current }) },
      down:  (e) => { if (e?.repeat) return; startMovement('pan-down');  if (panStepRef.current > 0) snapCamera({ type: 'pan-step', direction: 'down',  distance: panStepRef.current }) },
    },
  })
  useActionPair('nav:zoom', {
    label: 'Zoom in / out',
    description: 'Move camera closer or further from crystal',
    group: 'Camera',
    actions: [
      {
        defaultBindings: ['='],
        handler: (e) => {
          if (e?.repeat) return
          startMovement('zoom-in')
          if (zoomPctRef.current > 0) snapCamera({ type: 'zoom-step', direction: 'in', factor: 1 + zoomPctRef.current / 100 })
        },
      },
      {
        defaultBindings: ['-'],
        handler: (e) => {
          if (e?.repeat) return
          startMovement('zoom-out')
          if (zoomPctRef.current > 0) snapCamera({ type: 'zoom-step', direction: 'out', factor: 1 + zoomPctRef.current / 100 })
        },
      },
    ],
  })
  useActionPair('nav:roll', {
    label: 'Roll CCW / CW',
    description: 'Roll camera around view axis',
    group: 'Camera',
    actions: [
      {
        defaultBindings: ['['],
        handler: (e) => { if (e?.repeat) return; startMovement('roll-ccw') },
      },
      {
        defaultBindings: [']'],
        handler: (e) => { if (e?.repeat) return; startMovement('roll-cw') },
      },
    ],
  })
  useAction('nav:reset-pan', {
    label: 'Reset center',
    description: 'Return crystal to center of viewport',
    group: 'Camera',
    defaultBindings: ['r'],
    handler: () => { setCamTarget(null); snapCamera({ type: 'reset-pan' }) },
  })

  // Auto-restore on mount: ?m= param (OPFS cache → fetch) or last active OPFS volume
  const initialMaterialId = useRef(materialId)
  const initialVolumeId = useRef(currentVolumeId)
  const initialQuery = useQuery({
    queryKey: ['initial-material', initialMaterialId.current],
    queryFn: async (): Promise<{ data: VolumeData; filename: string; volumeId?: string } | null> => {
      const m = initialMaterialId.current
      if (opfsStore) {
        const volumes = await opfsStore.list()
        if (m) {
          const cachedFilename = `${m}.json.gz`
          const cached = volumes.find(v => v.filename === cachedFilename)
          if (cached) {
            const blob = await opfsStore.get(cached.id)
            if (blob) {
              const data = await parseBlob(blob, cached.filename)
              return { data, filename: cached.filename, volumeId: cached.id }
            }
          }
        }
        if (!m && initialVolumeId.current) {
          const vol = volumes.find(v => v.id === initialVolumeId.current)
          if (vol) {
            const blob = await opfsStore.get(vol.id)
            if (blob) {
              const data = await parseBlob(blob, vol.filename)
              return { data, filename: vol.filename, volumeId: vol.id }
            }
          }
        }
      }
      if (m) {
        const fetchUrl = s3UriToHttps(mpS3Uri(m))
        const { blob, json, filename } = await fetchVolumeJsonGz(fetchUrl)
        const data = parsePymatgenChgcar(json, filename)
        let volumeId: string | undefined
        if (opfsStore) {
          const meta = { filename, elements: data.structure.elements, counts: data.structure.counts, atomCount: data.structure.atoms.length, gridDims: data.grid.dims }
          const stored = await opfsStore.store(blob, filename, meta)
          volumeId = stored.id
        }
        return { data, filename, volumeId }
      }
      return null
    },
    staleTime: Infinity,
    enabled: files.length === 0,
  })

  // Populate files from initial query result
  useEffect(() => {
    const result = initialQuery.data
    if (result && files.length === 0) {
      setFiles([{ data: result.data, filename: result.filename }])
      // Only set iso/slice defaults when the URL didn't specify them
      if (isoLevel === null) setIsoLevel(computeDefaultIsoLevel(result.data.grid.data))
      if (sliceIndex === null) setSliceIndex(Math.floor(result.data.grid.dims[sliceAxis] / 2))
      if (result.volumeId) {
        setCurrentVolumeId(result.volumeId)
        setGalleryRefreshKey(k => k + 1)
      }
    }
  }, [initialQuery.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleLoad = useCallback(async (data: VolumeData, filename: string, blob?: Blob) => {
    // Best-effort set ?m= param from MP material ID in filename
    const mpId = extractMpId(filename)
    setMaterialId(mpId ?? '')

    setFiles([{ data, filename }])
    setIsoLevel(computeDefaultIsoLevel(data.grid.data))
    setSliceIndex(Math.floor(data.grid.dims[sliceAxis] / 2))

    // Cache in OPFS if enabled (skip if same filename already cached)
    if (opfsStore && settings.cacheInOPFS && blob) {
      const existing = (await opfsStore.list()).find(v => v.filename === filename)
      if (existing) {
        setCurrentVolumeId(existing.id)
      } else {
        const fileSizeMB = blob.size / (1024 * 1024)
        const meta = {
          filename,
          elements: data.structure.elements,
          counts: data.structure.counts,
          atomCount: data.structure.atoms.length,
          gridDims: data.grid.dims,
        }
        if (fileSizeMB > settings.maxUploadSizeMB) {
          setSizeConfirm({ blob, filename, fileSizeMB, meta, data })
        } else {
          const stored = await opfsStore.store(blob, filename, meta)
          setCurrentVolumeId(stored.id)
          setGalleryRefreshKey(k => k + 1)
        }
      }
    }
  }, [settings, setCurrentVolumeId])

  const handleGallerySelect = useCallback(async (_id: string, blob: Blob) => {
    // Show loading state while parsing
    setFiles([])
    setFetchStatus('Parsing volume...')
    // Get filename from store metadata first (needed for format detection)
    let filename = ''
    if (opfsStore) {
      const volumes = await opfsStore.list()
      const vol = volumes.find(v => v.id === _id)
      if (vol) filename = vol.filename
    }
    const data = await parseBlob(blob, filename)
    setFiles([{ data, filename }])
    setIsoLevel(computeDefaultIsoLevel(data.grid.data))
    setSliceIndex(Math.floor(data.grid.dims[sliceAxis] / 2))
    setCurrentVolumeId(_id)
    setMaterialId(extractMpId(filename) ?? '')
    setFetchStatus(null)
  }, [setCurrentVolumeId])

  const handleUrlSubmit = useCallback(async (url: string) => {
    setUrlLoading(true)
    setFetchStatus(null)
    setFiles([])
    try {
      const isJsonGz = url.toLowerCase().endsWith('.json.gz')

      const onProgress = (p: FetchProgress) => {
        if (p.phase === 'head') setFetchStatus('Checking file...')
        else if (p.phase === 'header' && p.header) {
          const dims = p.header.gridDims.join('\u00d7')
          const elems = p.header.elements.join('-')
          const size = p.contentLength ? ` (${(p.contentLength / (1024 * 1024)).toFixed(1)} MB)` : ''
          setFetchStatus(`${elems} ${dims}${size} \u2014 downloading...`)
        } else if (p.phase === 'downloading') {
          const size = p.contentLength ? ` (${(p.contentLength / (1024 * 1024)).toFixed(1)} MB)` : ''
          setFetchStatus(`Downloading${size}...`)
        }
      }

      // Resolve fetch URL: s3:// → anonymous HTTPS for .json.gz, credentialed for others
      let fetchUrl = url
      if (url.startsWith('s3://')) {
        if (isJsonGz) {
          fetchUrl = s3UriToHttps(url)
        } else {
          if (!awsCreds) {
            setAwsModalOpen(true)
            setUrlLoading(false)
            return
          }
          const result = await fetchVolumeFromS3(url, awsCreds, onProgress)
          const text = await result.blob.text()
          const data = parseCHGCAR(text)
          handleLoad(data, result.filename, result.blob)
          setFetchStatus(null)
          return
        }
      }

      if (isJsonGz) {
        const { blob, json, filename } = await fetchVolumeJsonGz(fetchUrl, onProgress)
        const data = parsePymatgenChgcar(json, filename)
        handleLoad(data, filename, blob)
      } else {
        const result = await fetchVolumeFromUrl(fetchUrl, onProgress)
        const text = await result.blob.text()
        const data = parseCHGCAR(text)
        handleLoad(data, result.filename, result.blob)
      }
      setFetchStatus(null)
    } catch (e) {
      // For s3:// .json.gz, if anonymous fetch fails (403), fall back to credentialed path
      if (url.startsWith('s3://') && url.toLowerCase().endsWith('.json.gz')) {
        if (awsCreds) {
          try {
            setFetchStatus('Anonymous access failed, trying with credentials...')
            const onProgress = (p: FetchProgress) => {
              if (p.phase === 'downloading') setFetchStatus('Downloading with credentials...')
            }
            const result = await fetchVolumeFromS3(url, awsCreds, onProgress)
            const data = await parseBlob(result.blob, result.filename)
            handleLoad(data, result.filename, result.blob)
            setFetchStatus(null)
            return
          } catch (e2) {
            setFetchStatus(`Error: ${e2 instanceof Error ? e2.message : String(e2)}`)
            return
          }
        }
      }
      setFetchStatus(`Error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setUrlLoading(false)
    }
  }, [awsCreds, handleLoad])

  const handleSizeConfirm = useCallback(async (cache: boolean) => {
    if (!sizeConfirm || !opfsStore) return
    if (cache) {
      const stored = await opfsStore.store(sizeConfirm.blob, sizeConfirm.filename, sizeConfirm.meta)
      setCurrentVolumeId(stored.id)
      setGalleryRefreshKey(k => k + 1)
    }
    setSizeConfirm(null)
  }, [sizeConfirm, setCurrentVolumeId])

  const handleAddFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const data = await parseBlob(file, file.name)
    setFiles(prev => [...prev, { data, filename: file.name }])
    e.target.value = ''
  }, [])

  const maxDensity = useMemo(() => {
    if (!primaryFile) return 1
    let max = 0
    for (let i = 0; i < primaryFile.data.grid.data.length; i++) {
      if (primaryFile.data.grid.data[i] > max) max = primaryFile.data.grid.data[i]
    }
    return max
  }, [primaryFile])

  const defaultIsoLevel = useMemo(() => {
    if (!primaryFile) return 0
    return computeDefaultIsoLevel(primaryFile.data.grid.data)
  }, [primaryFile])

  const effectiveIsoLevel = useMemo(
    () => Math.max(0, Math.min(isoLevel ?? 0, maxDensity)),
    [isoLevel, maxDensity],
  )

  const isoLevelRef = useRef(isoLevel ?? 0)
  isoLevelRef.current = isoLevel ?? 0
  const maxDensityRef = useRef(maxDensity)
  maxDensityRef.current = maxDensity
  const defaultIsoLevelRef = useRef(defaultIsoLevel)
  defaultIsoLevelRef.current = defaultIsoLevel

  const exampleLinks = (
    <details
      open={examplesOpen}
      onToggle={e => {
        const open = (e.target as HTMLDetailsElement).open
        setExamplesOpen(open)
        sessionStorage.setItem('elvis-examples-open', String(open))
      }}
      style={{ padding: '4px 16px', fontSize: 12, color: '#999' }}
    >
      <summary style={{ cursor: 'pointer', userSelect: 'none' }}>Examples</summary>
      <ul style={{ margin: '4px 0 0', paddingLeft: 20, lineHeight: 1.8 }}>
        {EXAMPLES.map(ex => {
          const cached = cachedMpIds.has(ex.mpId)
          return (
            <li key={`${ex.source ?? 'chgcars'}-${ex.mpId}`}>
              <a
                href="#"
                onClick={(e) => { e.preventDefault(); handleUrlSubmit(mpS3Uri(ex.mpId, ex.source)) }}
                style={{ color: cached ? '#6a8' : '#8ab', textDecoration: 'none' }}
              >
                {ex.mpId}
              </a>
              <span style={{ color: '#888' }}>
                {' \u2014 '}{ex.label}
                {cached && <span style={{ color: '#6a8', marginLeft: 4 }}>{'\u2713'}</span>}
              </span>
            </li>
          )
        })}
      </ul>
    </details>
  )

  const isComparison = files.length > 1
  const isLoading = !primaryFile && (initialQuery.isPending || initialQuery.isFetching || urlLoading || fetchStatus)

  return (
    <div className={styles.app}>
      <div className={styles.viewer}>
        <ErrorBoundary label="Viewer" resetKey={`${materialId}:${effectiveIsoLevel}`}>
          {primaryFile ? (
            <>
              {isComparison ? (
                <ComparisonView
                  volumes={files.map(f => ({ data: f.data, label: f.filename }))}
                  isoLevel={effectiveIsoLevel}
                  opacity={opacity}
                  showAtoms={showAtoms}
                  showAtomLabels={showAtomLabels}
                  showAbcCell={showAbcCell}
                  showXyzBox={showXyzBox}
                  showWorldAxes={showWorldAxes}
                  dashedLines={dashedLines}
                  activeMovements={activeMovements}
                  tilePadding={tilePadding}
                  tileFade={tileFade}
                />
              ) : (
                <DensityViewer
                  volume={primaryFile.data}
                  isoLevel={effectiveIsoLevel}
                  opacity={opacity}
                  showAtoms={showAtoms}
                  showAtomLabels={showAtomLabels}
                  showAbcCell={showAbcCell}
                  showXyzBox={showXyzBox}
                  showWorldAxes={showWorldAxes}
                  dashedLines={dashedLines}
                  lineWidth={lineWidth}
                  activeMovements={activeMovements}
                  cameraSnap={cameraSnap}
                  animationDuration={animDuration || settings.animationDuration}
                  onCameraChange={handleCameraChange}
                  initialCamera={initialCamera}
                  initialTargetOffset={initialTargetOffset}
                  showSlice={showSlice}
                  sliceAxis={sliceAxis}
                  sliceIndex={sliceIndex ?? 0}
                  tilePadding={tilePadding}
                  tileFade={tileFade}
                  abcIsXyz={abcIsXyz}
                />
              )}
              {showSlice && (() => {
                const d = primaryFile.data.grid.dims
                const [sw, sh] = sliceAxis === 0 ? [d[1], d[2]] : sliceAxis === 1 ? [d[0], d[2]] : [d[0], d[1]]
                const maxDim = 200
                const scale = maxDim / Math.max(sw, sh)
                return (
                  <div className={styles.slicePanel} style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    width: Math.round(sw * scale),
                    height: Math.round(sh * scale),
                    background: '#1a1a2e',
                    borderTop: '1px solid #333',
                    borderRight: '1px solid #333',
                  }}>
                    <SliceViewer
                      volume={primaryFile.data}
                      axis={sliceAxis}
                      sliceIndex={sliceIndex ?? 0}
                    />
                  </div>
                )
              })()}
            </>
          ) : (
            <div className={styles.loadingViewer}>
              {isLoading && (
                <>
                  <div className={styles.spinner} />
                  <div className={styles.loadingText}>
                    {fetchStatus && !fetchStatus.startsWith('Error') ? fetchStatus : 'Loading density data...'}
                  </div>
                </>
              )}
              {initialQuery.error && !fetchStatus && (
                <div className={styles.errorText}>
                  {initialQuery.error instanceof Error ? initialQuery.error.message : 'Failed to load'}
                </div>
              )}
            </div>
          )}
        </ErrorBoundary>
      </div>
      <div className={styles.sidebar}>
        {opfsStore && (
          <VolumeGallery
            store={opfsStore}
            currentVolumeId={currentVolumeId}
            onSelect={handleGallerySelect}
            refreshKey={galleryRefreshKey}
          />
        )}
        <Settings
          settings={settings}
          onUpdate={updateSettings}
          showCacheToggle={!!opfsStore}
          lineWidth={lineWidth}
          onLineWidthChange={setLineWidth}
        />
        <URLInput onSubmit={handleUrlSubmit} loading={urlLoading} />
        {fetchStatus && (
          <div style={{
            padding: '4px 16px',
            fontSize: 12,
            color: fetchStatus.startsWith('Error') ? '#ff4444' : '#aaa',
          }}>
            {fetchStatus}
          </div>
        )}
        {exampleLinks}
        {primaryFile && (
          <ErrorBoundary label="Controls" resetKey={`${materialId}:${effectiveIsoLevel}`}>
            <Controls
              isoLevel={effectiveIsoLevel}
              defaultIsoLevel={defaultIsoLevel}
              maxDensity={maxDensity}
              onIsoLevelChange={setIsoLevel}
              opacity={opacity}
              onOpacityChange={setOpacity}
              showAtoms={showAtoms}
              onShowAtomsChange={setShowAtoms}
              showAtomLabels={showAtomLabels}
              onShowAtomLabelsChange={setShowAtomLabels}
              showAbcCell={showAbcCell}
              onShowAbcCellChange={setShowAbcCell}
              showXyzBox={showXyzBox}
              onShowXyzBoxChange={setShowXyzBox}
              showWorldAxes={showWorldAxes}
              onShowWorldAxesChange={setShowWorldAxes}
              dashedLines={dashedLines}
              onDashedLinesChange={setDashedLines}
              orbitDeg={orbitDeg}
              onOrbitDegChange={setOrbitDeg}
              zoomPct={zoomPct}
              onZoomPctChange={setZoomPct}
              panStep={panStep}
              onPanStepChange={setPanStep}
              showSlice={showSlice}
              onShowSliceChange={setShowSlice}
              sliceAxis={sliceAxis}
              onSliceAxisChange={handleSliceAxisChange}
              sliceIndex={sliceIndex ?? 0}
              maxSliceIndex={maxSliceIndex}
              onSliceIndexChange={setSliceIndex}
              animDuration={animDuration}
              onAnimDurationChange={setAnimDuration}
              sweepMode={sweepMode as 'd' | 'r'}
              sweepDuration={sweepDuration}
              onSweepDurationChange={setSweepDuration}
              onSweepModeToggle={handleSweepModeToggle}
              sliceSpeed={sliceSpeed}
              onSliceSpeedChange={setSliceSpeed}
              cam={cam}
              filename={primaryFile.filename}
              elements={primaryFile.data.structure.elements}
              counts={primaryFile.data.structure.counts}
              abcIsXyz={abcIsXyz}
              tilePadding={tilePadding}
              onTilePaddingChange={setTilePadding}
              tileFade={tileFade}
              onTileFadeChange={setTileFade}
            />
          </ErrorBoundary>
        )}
        <input
          ref={addFileInputRef}
          type="file"
          accept=".CHGCAR,.ELFCAR,.npy,.json.gz"
          onChange={handleAddFile}
          style={{ display: 'none' }}
        />
        <button
          className={styles.addFileBtn}
          onClick={() => addFileInputRef.current?.click()}
        >
          + Add file for comparison
        </button>
      </div>

      <AWSCredentialsModal
        open={awsModalOpen}
        onClose={() => setAwsModalOpen(false)}
        onSave={(creds) => { saveCredentials(creds); setAwsCreds(creds) }}
        currentCreds={awsCreds}
        ssoContent={
          <SSOAuthFlow
            onSave={(creds) => { saveCredentials(creds); setAwsCreds(creds) }}
            onClose={() => setAwsModalOpen(false)}
          />
        }
      />

      <SizeConfirmModal
        open={!!sizeConfirm}
        filename={sizeConfirm?.filename ?? ''}
        fileSizeMB={sizeConfirm?.fileSizeMB ?? 0}
        maxSizeMB={settings.maxUploadSizeMB}
        storageUsage={null}
        onConfirm={handleSizeConfirm}
        onCancel={() => setSizeConfirm(null)}
      />

      <ShortcutsModal editable arrowIcon="move" TooltipComponent={MuiTooltip} />
      <Omnibar />
      <SequenceModal />
      <LookupModal />
      <SpeedDial actions={speedDialActions} />
      <ModeIndicator />
    </div>
  )
}
