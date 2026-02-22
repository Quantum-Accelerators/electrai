import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import type { VolumeData, StoredVolume } from '@elvis/core'
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
  useSettings,
} from '@elvis/core'
import { ShortcutsModal, Omnibar, SequenceModal, SpeedDial, useAction } from 'use-kbd'
import type { SpeedDialAction } from 'use-kbd'
import 'use-kbd/styles.css'
import { OpfsVolumeStore, isOPFSSupported } from './storage/OpfsVolumeStore.ts'
import { loadCredentials, saveCredentials } from './utils/aws-credentials.ts'
import { fetchVolumeFromUrl, fetchVolumeFromS3 } from './utils/fetch-volume.ts'
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

const opfsStore = isOPFSSupported() ? new OpfsVolumeStore() : null

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
  const [isoLevel, setIsoLevel] = useState(0)
  const [opacity, setOpacity] = useState(0.6)
  const [showAtoms, setShowAtoms] = useState(true)
  const [showUnitCell, setShowUnitCell] = useState(true)
  const [showWorldAxes, setShowWorldAxes] = useState(false)
  const [showSlice, setShowSlice] = useState(false)
  const [sliceAxis, setSliceAxis] = useState<0 | 1 | 2>(2)
  const [sliceIndex, setSliceIndex] = useState(0)
  const [currentVolumeId, setCurrentVolumeIdRaw] = useState<string | null>(
    () => sessionStorage.getItem('elvis-active-volume'),
  )
  const setCurrentVolumeId = useCallback((id: string | null) => {
    setCurrentVolumeIdRaw(id)
    if (id) sessionStorage.setItem('elvis-active-volume', id)
    else sessionStorage.removeItem('elvis-active-volume')
  }, [])
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

  // Camera movement state (shared with CameraController inside Canvas)
  const activeMovements = useRef(new Set<string>())

  const startMovement = useCallback((dir: string) => {
    activeMovements.current.add(dir)
  }, [])

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

  // View toggles
  useAction('view:toggle-atoms', {
    label: 'Toggle atoms',
    group: 'View',
    defaultBindings: ['a'],
    handler: () => setShowAtoms(v => !v),
  })
  useAction('view:toggle-unit-cell', {
    label: 'Toggle unit cell',
    group: 'View',
    defaultBindings: ['u'],
    handler: () => setShowUnitCell(v => !v),
  })
  useAction('view:toggle-world-axes', {
    label: 'Toggle XYZ axes',
    group: 'View',
    defaultBindings: ['x'],
    handler: () => setShowWorldAxes(v => !v),
  })
  useAction('view:toggle-slice', {
    label: 'Toggle 2D slice',
    group: 'View',
    defaultBindings: ['s'],
    handler: () => setShowSlice(v => !v),
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

  // Camera navigation
  useAction('nav:pan-left', {
    label: 'Pan left',
    group: 'Camera',
    defaultBindings: ['arrowleft'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-left') },
  })
  useAction('nav:pan-right', {
    label: 'Pan right',
    group: 'Camera',
    defaultBindings: ['arrowright'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-right') },
  })
  useAction('nav:pan-up', {
    label: 'Pan up',
    group: 'Camera',
    defaultBindings: ['arrowup'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-up') },
  })
  useAction('nav:pan-down', {
    label: 'Pan down',
    group: 'Camera',
    defaultBindings: ['arrowdown'],
    handler: (e) => { if (e?.repeat) return; startMovement('pan-down') },
  })
  useAction('nav:orbit-left', {
    label: 'Orbit left',
    group: 'Camera',
    defaultBindings: ['shift+arrowleft'],
    handler: (e) => { if (e?.repeat) return; startMovement('orbit-left') },
  })
  useAction('nav:orbit-right', {
    label: 'Orbit right',
    group: 'Camera',
    defaultBindings: ['shift+arrowright'],
    handler: (e) => { if (e?.repeat) return; startMovement('orbit-right') },
  })
  useAction('nav:orbit-up', {
    label: 'Orbit up',
    group: 'Camera',
    defaultBindings: ['shift+arrowup'],
    handler: (e) => { if (e?.repeat) return; startMovement('orbit-up') },
  })
  useAction('nav:orbit-down', {
    label: 'Orbit down',
    group: 'Camera',
    defaultBindings: ['shift+arrowdown'],
    handler: (e) => { if (e?.repeat) return; startMovement('orbit-down') },
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

  // Auto-restore last active volume from OPFS on mount
  useEffect(() => {
    if (!opfsStore || !currentVolumeId || files.length > 0) return
    let cancelled = false
    ;(async () => {
      const blob = await opfsStore.get(currentVolumeId)
      if (!blob || cancelled) return
      const volumes = await opfsStore.list()
      const vol = volumes.find(v => v.id === currentVolumeId)
      const text = await blob.text()
      const data = parseCHGCAR(text)
      if (cancelled) return
      const filename = vol?.filename ?? 'CHGCAR'
      setFiles([{ data, filename }])
      setIsoLevel(computeDefaultIsoLevel(data.grid.data))
      setSliceIndex(Math.floor(data.grid.dims[2] / 2))
    })()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleLoad = useCallback(async (data: VolumeData, filename: string, blob?: Blob) => {
    setFiles(prev => {
      const next = [...prev, { data, filename }]
      if (prev.length === 0) {
        setIsoLevel(computeDefaultIsoLevel(data.grid.data))
        setSliceIndex(Math.floor(data.grid.dims[2] / 2))
      }
      return next
    })

    // Cache in OPFS if enabled
    if (opfsStore && settings.cacheInOPFS && blob) {
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
      }
    }
  }, [settings, setCurrentVolumeId])

  const handleFileLoad = useCallback(async (data: VolumeData, filename: string, file: File) => {
    handleLoad(data, filename, file)
  }, [handleLoad])

  const handleGallerySelect = useCallback(async (_id: string, blob: Blob) => {
    const text = await blob.text()
    const data = parseCHGCAR(text)
    setFiles([{ data, filename: '' }])
    setIsoLevel(computeDefaultIsoLevel(data.grid.data))
    setSliceIndex(Math.floor(data.grid.dims[2] / 2))
    setCurrentVolumeId(_id)
    // Update filename from store metadata
    if (opfsStore) {
      const volumes = await opfsStore.list()
      const vol = volumes.find(v => v.id === _id)
      if (vol) {
        setFiles([{ data, filename: vol.filename }])
      }
    }
  }, [setCurrentVolumeId])

  const handleUrlSubmit = useCallback(async (url: string) => {
    setUrlLoading(true)
    setFetchStatus(null)
    try {
      const onProgress = (p: FetchProgress) => {
        if (p.phase === 'head') setFetchStatus('Checking file...')
        else if (p.phase === 'header' && p.header) {
          const dims = p.header.gridDims.join('\u00d7')
          const elems = p.header.elements.join('-')
          const size = p.contentLength ? ` (${(p.contentLength / (1024 * 1024)).toFixed(1)} MB)` : ''
          setFetchStatus(`${elems} ${dims}${size} \u2014 downloading...`)
        } else if (p.phase === 'downloading') setFetchStatus('Downloading...')
      }

      let blob: Blob, filename: string
      if (url.startsWith('s3://')) {
        if (!awsCreds) {
          setAwsModalOpen(true)
          setUrlLoading(false)
          return
        }
        const result = await fetchVolumeFromS3(url, awsCreds, onProgress)
        blob = result.blob
        filename = result.filename
      } else {
        const result = await fetchVolumeFromUrl(url, onProgress)
        blob = result.blob
        filename = result.filename
      }

      const text = await blob.text()
      const data = parseCHGCAR(text)
      handleLoad(data, filename, blob)
      setFetchStatus(null)
    } catch (e) {
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
    }
    setSizeConfirm(null)
  }, [sizeConfirm, setCurrentVolumeId])

  const handleAddFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    const data = parseCHGCAR(text)
    setFiles(prev => [...prev, { data, filename: file.name }])
    e.target.value = ''
  }, [])

  const primaryFile = files[0] ?? null
  const maxDensity = useMemo(() => {
    if (!primaryFile) return 1
    let max = 0
    for (let i = 0; i < primaryFile.data.grid.data.length; i++) {
      if (primaryFile.data.grid.data[i] > max) max = primaryFile.data.grid.data[i]
    }
    return max
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
            showUnitCell={showUnitCell}
            showWorldAxes={showWorldAxes}
            activeMovements={activeMovements}
          />
        ) : (
          <DensityViewer
            volume={primaryFile.data}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showUnitCell={showUnitCell}
            showWorldAxes={showWorldAxes}
            activeMovements={activeMovements}
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
          maxDensity={maxDensity}
          onIsoLevelChange={setIsoLevel}
          opacity={opacity}
          onOpacityChange={setOpacity}
          showAtoms={showAtoms}
          onShowAtomsChange={setShowAtoms}
          showUnitCell={showUnitCell}
          onShowUnitCellChange={setShowUnitCell}
          showWorldAxes={showWorldAxes}
          onShowWorldAxesChange={setShowWorldAxes}
          showSlice={showSlice}
          onShowSliceChange={setShowSlice}
          sliceAxis={sliceAxis}
          onSliceAxisChange={setSliceAxis}
          sliceIndex={sliceIndex}
          maxSliceIndex={maxSliceIndex}
          onSliceIndexChange={setSliceIndex}
          filename={primaryFile.filename}
        />
        <input
          ref={addFileInputRef}
          type="file"
          accept=".CHGCAR,.ELFCAR,.npy"
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
