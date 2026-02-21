import type { StoredVolume, VolumeStore } from '@elvis/core'

/**
 * Filesystem-based VolumeStore implementation (stub).
 * Future: reads/writes server filesystem, exposes via API.
 */
export class FsVolumeStore implements VolumeStore {
  async list(): Promise<StoredVolume[]> {
    throw new Error('FsVolumeStore not yet implemented')
  }

  async get(_id: string): Promise<Blob | null> {
    throw new Error('FsVolumeStore not yet implemented')
  }

  async store(
    _file: File | Blob,
    _filename: string,
    _meta: Omit<StoredVolume, 'id' | 'addedAt' | 'fileSize'>,
  ): Promise<StoredVolume> {
    throw new Error('FsVolumeStore not yet implemented')
  }

  async delete(_id: string): Promise<void> {
    throw new Error('FsVolumeStore not yet implemented')
  }

  async usage(): Promise<{ count: number; totalBytes: number }> {
    throw new Error('FsVolumeStore not yet implemented')
  }
}
