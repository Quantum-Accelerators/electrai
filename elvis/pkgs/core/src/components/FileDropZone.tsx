import { useCallback, useRef, useState } from 'react'
import type { DragEvent, ChangeEvent } from 'react'
import { parseCHGCAR } from '../parsers/chgcar.ts'
import { parseNpy } from '../parsers/npy.ts'
import type { VolumeData } from '../types.ts'

interface FileDropZoneProps {
  onLoad: (data: VolumeData, filename: string, file: File) => void
}

export function FileDropZone({ onLoad }: FileDropZoneProps) {
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(async (file: File) => {
    setError(null)
    setLoading(true)
    try {
      const name = file.name.toLowerCase()
      if (name.endsWith('.npy')) {
        const buf = await file.arrayBuffer()
        const { shape, data } = parseNpy(buf)
        if (shape.length !== 3) throw new Error(`Expected 3D array, got ${shape.length}D`)
        const volumeData: VolumeData = {
          title: file.name,
          scaleFactor: 1.0,
          lattice: [
            shape[0], 0, 0,
            0, shape[1], 0,
            0, 0, shape[2],
          ],
          structure: { elements: [], counts: [], atoms: [] },
          grid: {
            dims: [shape[0], shape[1], shape[2]],
            data,
          },
        }
        onLoad(volumeData, file.name, file)
      } else {
        // Assume CHGCAR/ELFCAR text format
        const text = await file.text()
        const volumeData = parseCHGCAR(text)
        onLoad(volumeData, file.name, file)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [onLoad])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processFile(file)
  }, [processFile])

  const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }, [processFile])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        border: `2px dashed ${dragging ? '#4a9eff' : '#666'}`,
        borderRadius: 12,
        padding: 40,
        cursor: 'pointer',
        background: dragging ? 'rgba(74, 158, 255, 0.05)' : 'transparent',
        transition: 'all 0.2s',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".CHGCAR,.ELFCAR,.npy"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      {loading ? (
        <p style={{ color: '#aaa', fontSize: 18 }}>Parsing file…</p>
      ) : (
        <>
          <p style={{ color: '#ccc', fontSize: 18, margin: 0 }}>
            Drop a CHGCAR, ELFCAR, or .npy file here
          </p>
          <p style={{ color: '#888', fontSize: 14, marginTop: 8 }}>
            or click to browse
          </p>
        </>
      )}
      {error && (
        <p style={{ color: '#ff4444', fontSize: 14, marginTop: 12 }}>{error}</p>
      )}
    </div>
  )
}
