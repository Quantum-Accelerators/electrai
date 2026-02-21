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
import { OpfsVolumeStore, isOPFSSupported } from './storage/OpfsVolumeStore.ts'
import { loadCredentials, saveCredentials } from './utils/aws-credentials.ts'
import { fetchVolumeFromUrl, fetchVolumeFromS3 } from './utils/fetch-volume.ts'
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

export default function App() {
  const [files, setFiles] = useState<LoadedFile[]>([])
  const [isoLevel, setIsoLevel] = useState(0)
  const [opacity, setOpacity] = useState(0.6)
  const [showAtoms, setShowAtoms] = useState(true)
  const [showUnitCell, setShowUnitCell] = useState(true)
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
        />
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
          />
        ) : (
          <DensityViewer
            volume={primaryFile.data}
            isoLevel={isoLevel}
            opacity={opacity}
            showAtoms={showAtoms}
            showUnitCell={showUnitCell}
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
    </div>
  )
}
