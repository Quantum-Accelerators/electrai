/**
 * Parse a NumPy .npy file (v1.0 format, float32/float64, C-contiguous).
 * Returns the shape and a Float32Array of the data.
 */
export function parseNpy(buffer: ArrayBuffer): { shape: number[]; data: Float32Array } {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)

  // Magic: \x93NUMPY
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e || bytes[2] !== 0x55 ||
      bytes[3] !== 0x4d || bytes[4] !== 0x50 || bytes[5] !== 0x59) {
    throw new Error('Not a valid .npy file (bad magic)')
  }

  const major = bytes[6]
  const headerLen = major >= 2
    ? view.getUint32(8, true)
    : view.getUint16(8, true)
  const headerOffset = major >= 2 ? 12 : 10

  const headerStr = new TextDecoder().decode(bytes.slice(headerOffset, headerOffset + headerLen))
  const dataOffset = headerOffset + headerLen

  // Parse the Python dict header for dtype and shape
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/)
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/)
  const orderMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/)

  if (!descrMatch || !shapeMatch) {
    throw new Error(`Failed to parse .npy header: ${headerStr}`)
  }

  if (orderMatch && orderMatch[1] === 'True') {
    throw new Error('Fortran-order .npy files not supported')
  }

  const descr = descrMatch[1]
  const shape = shapeMatch[1]
    .split(',')
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(Number)

  const rawData = buffer.slice(dataOffset)

  // Support little-endian float32 and float64
  if (descr === '<f4' || descr === '|f4') {
    return { shape, data: new Float32Array(rawData) }
  } else if (descr === '<f8' || descr === '|f8') {
    const f64 = new Float64Array(rawData)
    const f32 = new Float32Array(f64.length)
    for (let i = 0; i < f64.length; i++) {
      f32[i] = f64[i]
    }
    return { shape, data: f32 }
  } else {
    throw new Error(`Unsupported dtype: ${descr}`)
  }
}
