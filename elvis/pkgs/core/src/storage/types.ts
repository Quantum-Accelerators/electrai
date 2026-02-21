export interface StoredVolume {
  id: string
  filename: string
  elements: string[]
  atomCount: number
  gridDims: [number, number, number]
  fileSize: number
  addedAt: number
}

export interface VolumeStore {
  list(): Promise<StoredVolume[]>
  get(id: string): Promise<Blob | null>
  store(
    file: File | Blob,
    filename: string,
    meta: Omit<StoredVolume, 'id' | 'addedAt' | 'fileSize'>,
  ): Promise<StoredVolume>
  delete(id: string): Promise<void>
  usage(): Promise<{ count: number; totalBytes: number }>
}
