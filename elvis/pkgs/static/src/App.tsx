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
  parseCHGCAR,
  parsePymatgenChgcar,
  useSettings,
  fracToCart,
} from '@elvis/core'
import { ShortcutsModal, Omnibar, SequenceModal, SpeedDial, useAction } from 'use-kbd'
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
  const [showSlice, setShowSlice] = useUrlState('sl', boolParam)
  const [sliceAxis, setSliceAxis] = useUrlState('sa', intParam(2)) as [0 | 1 | 2, (v: 0 | 1 | 2) => void]
  const [sliceIndex, setSliceIndex] = useUrlState('si', optIntParam, { debounce: 300 })
  const [orbitDeg, setOrbitDeg] = useUrlState('od', intParam(30))
  const [zoomPct, setZoomPct] = useUrlState('zd', intParam(0))
  const [panStep, setPanStep] = useUrlState('pd', floatParam({ default: 0, encoding: 'string', decimals: 1 }))
  const [animDuration, setAnimDuration] = useUrlState('a', floatParam({ default: 0.5, encoding: 'string', decimals: 1 }))
  const [sliceSpeed, setSliceSpeed] = useUrlState('ss', intParam(120))
  const [lineWidth, setLineWidth] = useUrlState('lw', floatParam({ default: 1, encoding: 'string', decimals: 1 }))
  const [tilePadding, setTilePadding] = useUrlState('tp', floatParam({ default: 0, encoding: 'string', decimals: 1 }), { debounce: 300 })
  const [tileFade, setTileFade] = useUrlState('nf', boolTrueParam)
  const [cam, setCam] = useUrlState('c', camParam)
  const [materialId, setMaterialId] = useUrlState('m', stringParam(DEFAULT_MP_ID))
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

  const startMovement = useCallback((dir: string) => {
    activeMovements.current.add(dir)
  }, [])

  const handleCameraChange = useCallback((theta: number, phi: number, zoom: number, roll: number) => {
    setCam([theta, phi, zoom, roll])
  }, [setCam])

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

  // View toggles (t _ chords)
  // Group toggles: if either member is on, turn both off; if both off, turn both on
  useAction('view:toggle-xyz', {
    label: 'Toggle XYZ (axes + box)',
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
    group: 'View',
    defaultBindings: ['t shift+x'],
    handler: () => setShowWorldAxes(!showWorldAxes),
  })
  useAction('view:toggle-abc', {
    label: 'Toggle abc (cell + atoms)',
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
    group: 'View',
    defaultBindings: ['t shift+a'],
    handler: () => setShowAtoms(!showAtoms),
  })
  useAction('view:toggle-labels', {
    label: 'Toggle atom labels',
    group: 'View',
    defaultBindings: ['t l'],
    handler: () => setShowAtomLabels(!showAtomLabels),
  })
  useAction('view:toggle-dashed', {
    label: 'Toggle dashed outlines',
    group: 'View',
    defaultBindings: ['t d'],
    handler: () => setDashedLines(!dashedLines),
  })
  useAction('view:toggle-slice', {
    label: 'Toggle 2D slice',
    group: 'View',
    defaultBindings: ['t s'],
    handler: () => setShowSlice(!showSlice),
  })
  useAction('view:toggle-tiling', {
    label: 'Toggle tiling',
    group: 'View',
    defaultBindings: ['t t'],
    handler: () => setTilePadding(tilePadding > 0 ? 0 : 1),
  })
  useAction('view:set-tile-padding', {
    label: 'Set tile padding',
    keywords: ['tiling', 'padding', 'periodic'],
    group: 'View',
    defaultBindings: ['\\f t'],
    handler: (_e, captures) => setTilePadding(captures?.[0] ?? 1),
  })
  useAction('view:toggle-tile-fade', {
    label: 'Toggle tile fade',
    keywords: ['tiling', 'fade', 'opacity'],
    group: 'View',
    defaultBindings: ['t f'],
    handler: () => setTileFade(!tileFade),
  })
  useAction('view:set-orbit-deg', {
    label: 'Set orbit step',
    keywords: ['deg', '90deg', '90', 'discrete', 'step', 'angle'],
    group: 'View',
    defaultBindings: ['\\d+ o'],
    handler: (_e, captures) => {
      setOrbitDeg(captures?.[0] ?? 90)
    },
  })
  useAction('view:toggle-orbit', {
    label: 'Toggle orbit step',
    keywords: ['deg', 'discrete', 'step'],
    group: 'View',
    defaultBindings: ['t o'],
    handler: () => setOrbitDeg(orbitDeg > 0 ? 0 : 90),
  })
  useAction('view:set-zoom-step', {
    label: 'Set zoom step %',
    keywords: ['discrete', 'step', 'zoom'],
    group: 'View',
    defaultBindings: ['\\d+ z'],
    handler: (_e, captures) => setZoomPct(captures?.[0] ?? 20),
  })
  useAction('view:toggle-zoom-step', {
    label: 'Toggle zoom step',
    keywords: ['discrete', 'step', 'zoom'],
    group: 'View',
    defaultBindings: ['t z'],
    handler: () => setZoomPct(zoomPct > 0 ? 0 : 20),
  })
  useAction('view:set-pan-step', {
    label: 'Set pan step',
    keywords: ['discrete', 'step', 'pan'],
    group: 'View',
    defaultBindings: ['\\d+ p'],
    handler: (_e, captures) => setPanStep(captures?.[0] ?? 1),
  })
  useAction('view:toggle-pan-step', {
    label: 'Toggle pan step',
    keywords: ['discrete', 'step', 'pan'],
    group: 'View',
    defaultBindings: ['t p'],
    handler: () => setPanStep(panStep > 0 ? 0 : 1),
  })
  useAction('view:set-anim-speed', {
    label: 'Set animation speed (seconds)',
    keywords: ['duration', 'animation', 'speed'],
    group: 'View',
    defaultBindings: ['\\f s'],
    handler: (_e, captures) => setAnimDuration(captures?.[0] ?? 0.5),
  })
  useAction('view:set-slice-speed', {
    label: 'Set slice speed (slices/sec)',
    keywords: ['slice', 'animation', 'speed', 'sweep'],
    group: 'View',
    defaultBindings: ['\\d+ shift+s'],
    handler: (_e, captures) => setSliceSpeed(captures?.[0] ?? 120),
  })
  useAction('slice:axis-x', {
    label: 'Slice along X',
    group: 'Slice',
    defaultBindings: ['1'],
    handler: () => { setShowSlice(true); setSliceAxis(0) },
  })
  useAction('slice:axis-y', {
    label: 'Slice along Y',
    group: 'Slice',
    defaultBindings: ['2'],
    handler: () => { setShowSlice(true); setSliceAxis(1) },
  })
  useAction('slice:axis-z', {
    label: 'Slice along Z',
    group: 'Slice',
    defaultBindings: ['3'],
    handler: () => { setShowSlice(true); setSliceAxis(2) },
  })

  // Slice animation: sweep sliceIndex to start or end
  const sliceAnimRef = useRef<{ target: number; raf: number } | null>(null)
  const cancelSliceAnim = useCallback(() => {
    if (sliceAnimRef.current) {
      cancelAnimationFrame(sliceAnimRef.current.raf)
      sliceAnimRef.current = null
    }
  }, [])

  const animateSliceTo = useCallback((target: number) => {
    cancelSliceAnim()
    let last = performance.now()
    let current = sliceIndex ?? 0
    let fractional = 0
    const step = target > current ? 1 : -1
    const tick = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      fractional += sliceSpeed * dt
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
  }, [sliceIndex, sliceSpeed, cancelSliceAnim, setSliceIndex])

  useAction('slice:animate-start', {
    label: 'Animate slice to start',
    group: 'Slice',
    defaultBindings: ['meta+arrowleft'],
    handler: () => { setShowSlice(true); animateSliceTo(0) },
  })
  useAction('slice:animate-end', {
    label: 'Animate slice to end',
    group: 'Slice',
    defaultBindings: ['meta+arrowright'],
    handler: () => { setShowSlice(true); animateSliceTo(maxSliceIndex) },
  })
  useAction('slice:jump-start', {
    label: 'Jump slice to start',
    group: 'Slice',
    defaultBindings: ['meta+shift+arrowleft'],
    handler: () => { cancelSliceAnim(); setShowSlice(true); setSliceIndex(0) },
  })
  useAction('slice:jump-end', {
    label: 'Jump slice to end',
    group: 'Slice',
    defaultBindings: ['meta+shift+arrowright'],
    handler: () => { cancelSliceAnim(); setShowSlice(true); setSliceIndex(maxSliceIndex) },
  })

  // Camera axis-snap (look down lattice vectors or world axes)
  useAction('cam:snap-a', {
    label: 'Look down a',
    group: 'Camera',
    defaultBindings: ['a'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.a }) },
  })
  useAction('cam:align-a', {
    label: 'Align a up',
    group: 'Camera',
    defaultBindings: ['shift+a'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.a }) },
  })
  useAction('cam:snap-b', {
    label: 'Look down b',
    group: 'Camera',
    defaultBindings: ['b'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.b }) },
  })
  useAction('cam:align-b', {
    label: 'Align b up',
    group: 'Camera',
    defaultBindings: ['shift+b'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.b }) },
  })
  useAction('cam:snap-c', {
    label: 'Look down c',
    group: 'Camera',
    defaultBindings: ['c'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.c }) },
  })
  useAction('cam:align-c', {
    label: 'Align c up',
    group: 'Camera',
    defaultBindings: ['shift+c'],
    enabled: !abcIsXyz,
    handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.c }) },
  })
  useAction('cam:snap-x', {
    label: 'Look down X',
    group: 'Camera',
    defaultBindings: ['x'],
    handler: () => snapCamera({ type: 'look-down', direction: [1, 0, 0] }),
  })
  useAction('cam:align-x', {
    label: 'Align X up',
    group: 'Camera',
    defaultBindings: ['shift+x'],
    handler: () => snapCamera({ type: 'align-up', axis: [1, 0, 0] }),
  })
  useAction('cam:snap-y', {
    label: 'Look down Y',
    group: 'Camera',
    defaultBindings: ['y'],
    handler: () => snapCamera({ type: 'look-down', direction: [0, 1, 0] }),
  })
  useAction('cam:align-y', {
    label: 'Align Y up',
    group: 'Camera',
    defaultBindings: ['shift+y'],
    handler: () => snapCamera({ type: 'align-up', axis: [0, 1, 0] }),
  })
  useAction('cam:snap-z', {
    label: 'Look down Z',
    group: 'Camera',
    defaultBindings: ['z'],
    handler: () => snapCamera({ type: 'look-down', direction: [0, 0, 1] }),
  })
  useAction('cam:align-z', {
    label: 'Align Z up',
    group: 'Camera',
    defaultBindings: ['shift+z'],
    handler: () => snapCamera({ type: 'align-up', axis: [0, 0, 1] }),
  })

  // Camera navigation (orbit: continuous or discrete step snaps)
  // In discrete mode, startMovement tracks held state so CameraController can chain snaps
  useAction('nav:orbit-left', {
    label: 'Orbit left',
    group: 'Camera',
    defaultBindings: ['arrowleft'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('orbit-left')
      if (orbitDeg > 0) snapCamera({ type: 'orbit-step', direction: 'left', degrees: orbitDeg })
    },
  })
  useAction('nav:orbit-right', {
    label: 'Orbit right',
    group: 'Camera',
    defaultBindings: ['arrowright'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('orbit-right')
      if (orbitDeg > 0) snapCamera({ type: 'orbit-step', direction: 'right', degrees: orbitDeg })
    },
  })
  useAction('nav:orbit-up', {
    label: 'Orbit up',
    group: 'Camera',
    defaultBindings: ['arrowup'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('orbit-up')
      if (orbitDeg > 0) snapCamera({ type: 'orbit-step', direction: 'up', degrees: orbitDeg })
    },
  })
  useAction('nav:orbit-down', {
    label: 'Orbit down',
    group: 'Camera',
    defaultBindings: ['arrowdown'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('orbit-down')
      if (orbitDeg > 0) snapCamera({ type: 'orbit-step', direction: 'down', degrees: orbitDeg })
    },
  })
  useAction('nav:pan-left', {
    label: 'Pan left',
    group: 'Camera',
    defaultBindings: ['shift+arrowleft'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('pan-left')
      if (panStep > 0) snapCamera({ type: 'pan-step', direction: 'left', distance: panStep })
    },
  })
  useAction('nav:pan-right', {
    label: 'Pan right',
    group: 'Camera',
    defaultBindings: ['shift+arrowright'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('pan-right')
      if (panStep > 0) snapCamera({ type: 'pan-step', direction: 'right', distance: panStep })
    },
  })
  useAction('nav:pan-up', {
    label: 'Pan up',
    group: 'Camera',
    defaultBindings: ['shift+arrowup'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('pan-up')
      if (panStep > 0) snapCamera({ type: 'pan-step', direction: 'up', distance: panStep })
    },
  })
  useAction('nav:pan-down', {
    label: 'Pan down',
    group: 'Camera',
    defaultBindings: ['shift+arrowdown'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('pan-down')
      if (panStep > 0) snapCamera({ type: 'pan-step', direction: 'down', distance: panStep })
    },
  })
  useAction('nav:zoom-in', {
    label: 'Zoom in',
    group: 'Camera',
    defaultBindings: ['='],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('zoom-in')
      if (zoomPct > 0) snapCamera({ type: 'zoom-step', direction: 'in', factor: 1 + zoomPct / 100 })
    },
  })
  useAction('nav:zoom-out', {
    label: 'Zoom out',
    group: 'Camera',
    defaultBindings: ['-'],
    handler: (e) => {
      if (e?.repeat) return
      startMovement('zoom-out')
      if (zoomPct > 0) snapCamera({ type: 'zoom-step', direction: 'out', factor: 1 + zoomPct / 100 })
    },
  })
  useAction('nav:roll-ccw', {
    label: 'Roll CCW',
    group: 'Camera',
    defaultBindings: ['['],
    handler: (e) => { if (e?.repeat) return; startMovement('roll-ccw') },
  })
  useAction('nav:roll-cw', {
    label: 'Roll CW',
    group: 'Camera',
    defaultBindings: [']'],
    handler: (e) => { if (e?.repeat) return; startMovement('roll-cw') },
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
      if (sliceIndex === null) setSliceIndex(Math.floor(result.data.grid.dims[2] / 2))
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
    setSliceIndex(Math.floor(data.grid.dims[2] / 2))

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
    setSliceIndex(Math.floor(data.grid.dims[2] / 2))
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

  const maxSliceIndex = useMemo(() => {
    if (!primaryFile) return 0
    return primaryFile.data.grid.dims[sliceAxis] - 1
  }, [primaryFile, sliceAxis])

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
        {primaryFile ? (
          <>
            {isComparison ? (
              <ComparisonView
                volumes={files.map(f => ({ data: f.data, label: f.filename }))}
                isoLevel={isoLevel ?? 0}
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
                isoLevel={isoLevel ?? 0}
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
                showSlice={showSlice}
                sliceAxis={sliceAxis}
                sliceIndex={sliceIndex ?? 0}
                tilePadding={tilePadding}
                tileFade={tileFade}
                abcIsXyz={abcIsXyz}
              />
            )}
            {showSlice && (
              <div className={styles.slicePanel} style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                width: 280,
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
            )}
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
          <Controls
            isoLevel={isoLevel ?? 0}
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
            onSliceAxisChange={setSliceAxis}
            sliceIndex={sliceIndex ?? 0}
            maxSliceIndex={maxSliceIndex}
            onSliceIndexChange={setSliceIndex}
            animDuration={animDuration}
            onAnimDurationChange={setAnimDuration}
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

      <ShortcutsModal editable />
      <Omnibar />
      <SequenceModal />
      <SpeedDial actions={speedDialActions} />
    </div>
  )
}
