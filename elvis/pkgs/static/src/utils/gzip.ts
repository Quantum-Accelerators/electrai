/** Decompress a gzipped Blob using the browser-native DecompressionStream API. */
export async function decompressGzip(blob: Blob): Promise<ArrayBuffer> {
  const ds = new DecompressionStream('gzip')
  const reader = blob.stream().pipeThrough(ds).getReader()
  const chunks: Uint8Array[] = []
  let totalLength = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    totalLength += value.byteLength
  }
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.byteLength
  }
  return result.buffer
}
