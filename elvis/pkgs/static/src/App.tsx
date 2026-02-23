import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import type { VolumeData, StoredVolume, CameraSnapTarget } from '@elvis/core'
import {
  FileDropZone,
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
import { useUrlState, floatParam, boolParam, intParam, stringParam } from 'use-prms'
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

const MP_S3_PREFIX = 's3://materialsproject-parsed/chgcars/'

/** Extract mp-XXXXXX ID from a filename or S3 URI, if present. */
function extractMpId(s: string): string | undefined {
  const m = s.match(/(mp-\d+)/)
  return m?.[1]
}

/** Build the canonical MP S3 URI for a material ID. */
function mpS3Uri(mpId: string): string {
  return `${MP_S3_PREFIX}${mpId}.json.gz`
}

const EXAMPLES = [
  { mpId: 'mp-1000020', label: 'Fe\u2082Cu\u2082O\u2084 (8 MB)' },
  { mpId: 'mp-1828986', label: 'Na\u2082Al\u2082Si\u2084O\u2081\u2082 (10 MB)' },
  { mpId: 'mp-1000005', label: '17 MB' },
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
    href: 'https://github.com/Quantum-Accelerators/electrai',
  },
]

export default function App() {
  const [files, setFiles] = useState<LoadedFile[]>([])
  const [isoLevel, setIsoLevel] = useUrlState('iso', floatParam({ default: 0, encoding: 'string', decimals: 1 }), { debounce: 300 })
  const [opacity, setOpacity] = useUrlState('op', floatParam({ default: 0.6, encoding: 'string', decimals: 2 }), { debounce: 300 })
  const [showAtoms, setShowAtoms] = useUrlState('ha', boolTrueParam)
  const [showAbcCell, setShowAbcCell] = useUrlState('hc', boolTrueParam)
  const [showXyzBox, setShowXyzBox] = useUrlState('xb', boolParam)
  const [showWorldAxes, setShowWorldAxes] = useUrlState('xa', boolParam)
  const [showSlice, setShowSlice] = useUrlState('sl', boolParam)
  const [sliceAxis, setSliceAxis] = useUrlState('sa', intParam(2)) as [0 | 1 | 2, (v: 0 | 1 | 2) => void]
  const [sliceIndex, setSliceIndex] = useUrlState('si', intParam(0), { debounce: 300 })
  const [discreteOrbit, setDiscreteOrbit] = useUrlState('do', boolParam)
  const [cam, setCam] = useUrlState('c', camParam)
  const [materialId, setMaterialId] = useUrlState('m', stringParam(''))
  const [currentVolumeId, setCurrentVolumeIdRaw] = useState<string | null>(
    () => sessionStorage.getItem('elvis-active-volume'),
  )
  const setCurrentVolumeId = useCallback((id: string | null) => {
    setCurrentVolumeIdRaw(id)
    if (id) sessionStorage.setItem('elvis-active-volume', id)
    else sessionStorage.removeItem('elvis-active-volume')
  }, [])
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0)
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

  const snapCamera = useCallback((snap: CameraSnapTarget) => {
    cameraSnap.current = snap
  }, [])

  // View toggles (t _ chords)
  useAction('view:toggle-atoms', {
    label: 'Toggle atoms',
    group: 'View',
    defaultBindings: ['t a'],
    handler: () => setShowAtoms(!showAtoms),
  })
  useAction('view:toggle-abc-cell', {
    label: 'Toggle abc cell',
    group: 'View',
    defaultBindings: ['t c'],
    handler: () => setShowAbcCell(!showAbcCell),
  })
  useAction('view:toggle-xyz-box', {
    label: 'Toggle XYZ box',
    group: 'View',
    defaultBindings: ['t b'],
    handler: () => setShowXyzBox(!showXyzBox),
  })
  useAction('view:toggle-world-axes', {
    label: 'Toggle XYZ axes',
    group: 'View',
    defaultBindings: ['t x'],
    handler: () => setShowWorldAxes(!showWorldAxes),
  })
  useAction('view:toggle-slice', {
    label: 'Toggle 2D slice',
    group: 'View',
    defaultBindings: ['t s'],
    handler: () => setShowSlice(!showSlice),
  })
  useAction('view:toggle-90-orbit', {
    label: 'Toggle 90° orbit',
    keywords: ['deg', '90deg', '90', 'discrete', 'step'],
    group: 'View',
    defaultBindings: ['t o'],
    handler: () => setDiscreteOrbit(!discreteOrbit),
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

  // Camera axis-snap (look down lattice vectors or world axes)
  useAction('cam:snap-a', {
    label: 'Look down a',
    group: 'Camera',
    defaultBindings: ['a'],
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.a }) },
  })
  useAction('cam:align-a', {
    label: 'Align a up',
    group: 'Camera',
    defaultBindings: ['shift+a'],
    handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.a }) },
  })
  useAction('cam:snap-b', {
    label: 'Look down b',
    group: 'Camera',
    defaultBindings: ['b'],
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.b }) },
  })
  useAction('cam:align-b', {
    label: 'Align b up',
    group: 'Camera',
    defaultBindings: ['shift+b'],
    handler: () => { if (latticeDirections) snapCamera({ type: 'align-up', axis: latticeDirections.b }) },
  })
  useAction('cam:snap-c', {
    label: 'Look down c',
    group: 'Camera',
    defaultBindings: ['c'],
    handler: () => { if (latticeDirections) snapCamera({ type: 'look-down', direction: latticeDirections.c }) },
  })
  useAction('cam:align-c', {
    label: 'Align c up',
    group: 'Camera',
    defaultBindings: ['shift+c'],
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

  // Camera navigation (orbit: continuous or discrete 90-degree snaps)
  useAction('nav:orbit-left', {
    label: 'Orbit left',
    group: 'Camera',
    defaultBindings: ['arrowleft'],
    handler: (e) => {
      if (discreteOrbit) { snapCamera({ type: 'orbit-step', direction: 'left' }) }
      else { if (e?.repeat) return; startMovement('orbit-left') }
    },
  })
  useAction('nav:orbit-right', {
    label: 'Orbit right',
    group: 'Camera',
    defaultBindings: ['arrowright'],
    handler: (e) => {
      if (discreteOrbit) { snapCamera({ type: 'orbit-step', direction: 'right' }) }
      else { if (e?.repeat) return; startMovement('orbit-right') }
    },
  })
  useAction('nav:orbit-up', {
    label: 'Orbit up',
    group: 'Camera',
    defaultBindings: ['arrowup'],
    handler: (e) => {
      if (discreteOrbit) { snapCamera({ type: 'orbit-step', direction: 'up' }) }
      else { if (e?.repeat) return; startMovement('orbit-up') }
    },
  })
  useAction('nav:orbit-down', {
    label: 'Orbit down',
    group: 'Camera',
    defaultBindings: ['arrowdown'],
    handler: (e) => {
      if (discreteOrbit) { snapCamera({ type: 'orbit-step', direction: 'down' }) }
      else { if (e?.repeat) return; startMovement('orbit-down') }
    },
  })
  useAction('nav:pan-left', {
    label: 'Pan left',
    group: 'Camera',
    defaultBindings: ['shift+arrowleft'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-left') },
  })
  useAction('nav:pan-right', {
    label: 'Pan right',
    group: 'Camera',
    defaultBindings: ['shift+arrowright'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-right') },
  })
  useAction('nav:pan-up', {
    label: 'Pan up',
    group: 'Camera',
    defaultBindings: ['shift+arrowup'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-up') },
  })
  useAction('nav:pan-down', {
    label: 'Pan down',
    group: 'Camera',
    defaultBindings: ['shift+arrowdown'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-down') },
  })
  useAction('nav:zoom-in', {
    label: 'Zoom in',
    group: 'Camera',
    defaultBindings: ['='],
    handler: (e) => { if (e?.repeat) return; startMovement('zoom-in') },
  })
  useAction('nav:zoom-out', {
    label: 'Zoom out',
    group: 'Camera',
    defaultBindings: ['-'],
    handler: (e) => { if (e?.repeat) return; startMovement('zoom-out') },
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
  const materialIdRef = useRef(materialId)
  useEffect(() => {
    if (files.length > 0) return
    let cancelled = false
    ;(async () => {
      const m = materialIdRef.current
      if (opfsStore) {
        const volumes = await opfsStore.list()
        if (cancelled) return
        // If ?m= is set, try to find a matching cached file first
        if (m) {
          const cachedFilename = `${m}.json.gz`
          const cached = volumes.find(v => v.filename === cachedFilename)
          if (cached) {
            const blob = await opfsStore.get(cached.id)
            if (blob && !cancelled) {
              const data = await parseBlob(blob, cached.filename)
              if (cancelled) return
              setFiles([{ data, filename: cached.filename }])
              setIsoLevel(computeDefaultIsoLevel(data.grid.data))
              setSliceIndex(Math.floor(data.grid.dims[2] / 2))
              setCurrentVolumeId(cached.id)
              return
            }
          }
        }
        // No ?m= or not cached — try restoring last active volume
        if (!m && currentVolumeId) {
          const vol = volumes.find(v => v.id === currentVolumeId)
          if (vol) {
            const blob = await opfsStore.get(vol.id)
            if (blob && !cancelled) {
              const data = await parseBlob(blob, vol.filename)
              if (cancelled) return
              setFiles([{ data, filename: vol.filename }])
              setIsoLevel(computeDefaultIsoLevel(data.grid.data))
              setSliceIndex(Math.floor(data.grid.dims[2] / 2))
              return
            }
          }
        }
      }
      // ?m= set but not in cache — fetch from S3
      if (m && !cancelled) {
        handleUrlSubmit(mpS3Uri(m))
      }
    })()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleLoad = useCallback(async (data: VolumeData, filename: string, blob?: Blob) => {
    // Best-effort set ?m= param from MP material ID in filename
    const mpId = extractMpId(filename)
    setMaterialId(mpId ?? '')

    setFiles(prev => {
      const next = [...prev, { data, filename }]
      if (prev.length === 0) {
        setIsoLevel(computeDefaultIsoLevel(data.grid.data))
        setSliceIndex(Math.floor(data.grid.dims[2] / 2))
      }
      return next
    })

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

  const handleFileLoad = useCallback(async (data: VolumeData, filename: string, file: File) => {
    handleLoad(data, filename, file)
  }, [handleLoad])

  const handleGallerySelect = useCallback(async (_id: string, blob: Blob) => {
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
  }, [setCurrentVolumeId])

  const handleUrlSubmit = useCallback(async (url: string) => {
    setUrlLoading(true)
    setFetchStatus(null)
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

  if (files.length === 0) {
    return (
      <div className={styles.dropZone}>
        <div style={{ width: '100%', maxWidth: 600 }}>
          <FileDropZone onLoad={handleFileLoad} />
          <URLInput onSubmit={handleUrlSubmit} loading={urlLoading} />
          {fetchStatus && (
            <div style={{
              padding: '8px 16px',
              fontSize: 12,
              color: fetchStatus.startsWith('Error') ? '#ff4444' : '#aaa',
            }}>
              {fetchStatus}
            </div>
          )}
          <div style={{ padding: '4px 16px', fontSize: 12, color: '#666' }}>
            Examples:
            <ul style={{ margin: '4px 0 0', paddingLeft: 20, lineHeight: 1.8 }}>
              {EXAMPLES.map(ex => (
                <li key={ex.mpId}>
                  <a
                    href="#"
                    onClick={(e) => { e.preventDefault(); handleUrlSubmit(mpS3Uri(ex.mpId)) }}
                    style={{ color: '#8ab', textDecoration: 'none' }}
                  >
                    {ex.mpId}
                  </a>
                  <span style={{ color: '#555' }}>{' \u2014 '}{ex.label}</span>
                </li>
              ))}
            </ul>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginTop: 12 }}>
            <button
              onClick={() => setAwsModalOpen(true)}
              style={{
                padding: '4px 12px',
                background: 'transparent',
                border: '1px solid #444',
                borderRadius: 4,
                color: awsCreds ? '#8bc' : '#888',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              {awsCreds ? 'AWS \u2713' : 'AWS Credentials'}
            </button>
          </div>
        </div>
        {opfsStore && (
          <VolumeGallery
            store={opfsStore}
            currentVolumeId={currentVolumeId}
            onSelect={handleGallerySelect}
            refreshKey={galleryRefreshKey}
          />
        )}
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
        <SpeedDial actions={speedDialActions} />
      </div>
    )
  }

  const isComparison = files.length > 1

  return (
    <div className={styles.app}>
      <div className={styles.viewer}>
        {isComparison ? (
          <ComparisonView
            volumes={files.map(f => ({ data: f.data, label: f.filename }))}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showAbcCell={showAbcCell}
            showXyzBox={showXyzBox}
            showWorldAxes={showWorldAxes}
            activeMovements={activeMovements}
          />
        ) : (
          <DensityViewer
            volume={primaryFile.data}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showAbcCell={showAbcCell}
            showXyzBox={showXyzBox}
            showWorldAxes={showWorldAxes}
            activeMovements={activeMovements}
            cameraSnap={cameraSnap}
            animationDuration={settings.animationDuration}
            onCameraChange={handleCameraChange}
            initialCamera={initialCamera}
            showSlice={showSlice}
            sliceAxis={sliceAxis}
            sliceIndex={sliceIndex}
          />
        )}
        {showSlice && primaryFile && (
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
              sliceIndex={sliceIndex}
            />
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
        <Controls
          isoLevel={isoLevel}
          defaultIsoLevel={defaultIsoLevel}
          maxDensity={maxDensity}
          onIsoLevelChange={setIsoLevel}
          opacity={opacity}
          onOpacityChange={setOpacity}
          showAtoms={showAtoms}
          onShowAtomsChange={setShowAtoms}
          showAbcCell={showAbcCell}
          onShowAbcCellChange={setShowAbcCell}
          showXyzBox={showXyzBox}
          onShowXyzBoxChange={setShowXyzBox}
          showWorldAxes={showWorldAxes}
          onShowWorldAxesChange={setShowWorldAxes}
          discreteOrbit={discreteOrbit}
          onDiscreteOrbitChange={setDiscreteOrbit}
          showSlice={showSlice}
          onShowSliceChange={setShowSlice}
          sliceAxis={sliceAxis}
          onSliceAxisChange={setSliceAxis}
          sliceIndex={sliceIndex}
          maxSliceIndex={maxSliceIndex}
          onSliceIndexChange={setSliceIndex}
          filename={primaryFile.filename}
          elements={primaryFile.data.structure.elements}
        />
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
