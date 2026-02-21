import type { StoredVolume, VolumeStore } from '@elvis/core'

const METADATA_FILE = 'metadata.json'
const FILES_DIR = 'files'

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

async function getStorageRoot(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle('elvis', { create: true })
}

async function getSubdir(parent: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle> {
  return parent.getDirectoryHandle(name, { create: true })
}

interface Metadata {
  volumes: StoredVolume[]
}

async function readMetadata(root: FileSystemDirectoryHandle): Promise<Metadata> {
  try {
    const fileHandle = await root.getFileHandle(METADATA_FILE)
    const file = await fileHandle.getFile()
    const text = await file.text()
    return JSON.parse(text)
  } catch {
    return { volumes: [] }
  }
}

async function writeMetadata(root: FileSystemDirectoryHandle, metadata: Metadata): Promise<void> {
  const fileHandle = await root.getFileHandle(METADATA_FILE, { create: true })
  const writable = await fileHandle.createWritable()
  await writable.write(JSON.stringify(metadata, null, 2))
  await writable.close()
}

export function isOPFSSupported(): boolean {
  return 'storage' in navigator && 'getDirectory' in navigator.storage
}

export class OpfsVolumeStore implements VolumeStore {
  async list(): Promise<StoredVolume[]> {
    const root = await getStorageRoot()
    const metadata = await readMetadata(root)
    return metadata.volumes
  }

  async get(id: string): Promise<Blob | null> {
    try {
      const root = await getStorageRoot()
      const filesDir = await getSubdir(root, FILES_DIR)
      const fileHandle = await filesDir.getFileHandle(id)
      return await fileHandle.getFile()
    } catch {
      return null
    }
  }

  async store(
    file: File | Blob,
    filename: string,
    meta: Omit<StoredVolume, 'id' | 'addedAt' | 'fileSize'>,
  ): Promise<StoredVolume> {
    const root = await getStorageRoot()
    const filesDir = await getSubdir(root, FILES_DIR)

    const id = generateId()

    const fileHandle = await filesDir.getFileHandle(id, { create: true })
    const writable = await fileHandle.createWritable()
    await writable.write(file)
    await writable.close()

    const metadata = await readMetadata(root)
    const stored: StoredVolume = {
      ...meta,
      id,
      fileSize: file.size,
      addedAt: Date.now(),
    }
    metadata.volumes.unshift(stored)
    // Also store filename explicitly (meta may not have it at the right key)
    stored.filename = filename
    await writeMetadata(root, metadata)

    return stored
  }

  async delete(id: string): Promise<void> {
    const root = await getStorageRoot()
    const filesDir = await getSubdir(root, FILES_DIR)

    try {
      await filesDir.removeEntry(id)
    } catch {
      // File may not exist
    }

    const metadata = await readMetadata(root)
    metadata.volumes = metadata.volumes.filter(v => v.id !== id)
    await writeMetadata(root, metadata)
  }

  async usage(): Promise<{ count: number; totalBytes: number }> {
    const metadata = await readMetadata(await getStorageRoot())
    const totalBytes = metadata.volumes.reduce((sum, v) => sum + v.fileSize, 0)
    return { count: metadata.volumes.length, totalBytes }
  }
}
