import { S3Client, HeadObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3'
import { parseCHGCARHeader } from '@elvis/core'
import type { CHGCARHeader } from '@elvis/core'
import type { AWSCredentials } from './aws-credentials.ts'
import { decompressGzip } from './gzip.ts'

export interface FetchProgress {
  phase: 'head' | 'header' | 'downloading' | 'done'
  header?: CHGCARHeader
  contentLength?: number
}

function parseS3Uri(uri: string): { bucket: string; key: string } {
  const match = uri.match(/^s3:\/\/([^/]+)\/(.+)$/)
  if (!match) throw new Error(`Invalid S3 URI: ${uri}`)
  return { bucket: match[1], key: match[2] }
}

/** Convert s3://bucket/key → https://bucket.s3.amazonaws.com/key for anonymous access. */
export function s3UriToHttps(uri: string): string {
  const { bucket, key } = parseS3Uri(uri)
  return `https://${bucket}.s3.amazonaws.com/${key}`
}

/** Fetch a .json.gz file, decompress, and parse JSON. Returns the blob (for caching), parsed JSON, and filename. */
export async function fetchVolumeJsonGz(
  url: string,
  onProgress?: (progress: FetchProgress) => void,
): Promise<{ blob: Blob; json: unknown; filename: string }> {
  onProgress?.({ phase: 'head' })

  // HEAD for content-length (no range-request preview for gzip)
  const headResp = await fetch(url, { method: 'HEAD' })
  const contentLength = headResp.ok
    ? parseInt(headResp.headers.get('content-length') ?? '0', 10)
    : undefined
  onProgress?.({ phase: 'downloading', contentLength })

  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status} ${resp.statusText}`)
  const blob = await resp.blob()

  onProgress?.({ phase: 'done', contentLength })
  const buf = await decompressGzip(blob)
  const json = JSON.parse(new TextDecoder().decode(buf))
  const filename = url.split('/').pop() ?? 'chgcar.json.gz'
  return { blob, json, filename }
}

export async function fetchVolumeFromUrl(
  url: string,
  onProgress?: (progress: FetchProgress) => void,
): Promise<{ blob: Blob; header: CHGCARHeader; filename: string }> {
  onProgress?.({ phase: 'head' })

  // Parallel: HEAD for content-length + Range for header metadata
  const [headResp, rangeResp] = await Promise.all([
    fetch(url, { method: 'HEAD' }),
    fetch(url, { headers: { Range: 'bytes=0-4095' } }),
  ])

  const contentLength = headResp.ok
    ? parseInt(headResp.headers.get('content-length') ?? '0', 10)
    : undefined

  const headerText = await rangeResp.text()
  const header = parseCHGCARHeader(headerText)
  onProgress?.({ phase: 'header', header, contentLength })

  onProgress?.({ phase: 'downloading', header, contentLength })
  const fullResp = await fetch(url)
  if (!fullResp.ok) throw new Error(`Failed to fetch: ${fullResp.status} ${fullResp.statusText}`)
  const blob = await fullResp.blob()

  const filename = url.split('/').pop() ?? 'CHGCAR'
  onProgress?.({ phase: 'done', header, contentLength })

  return { blob, header, filename }
}

export async function fetchVolumeFromS3(
  uri: string,
  creds: AWSCredentials,
  onProgress?: (progress: FetchProgress) => void,
): Promise<{ blob: Blob; header: CHGCARHeader; filename: string }> {
  const { bucket, key } = parseS3Uri(uri)

  const client = new S3Client({
    region: creds.region ?? 'us-east-1',
    credentials: {
      accessKeyId: creds.accessKeyId,
      secretAccessKey: creds.secretAccessKey,
      sessionToken: creds.sessionToken,
    },
  })

  onProgress?.({ phase: 'head' })

  // Head request for content-length
  const headResult = await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key }))
  const contentLength = headResult.ContentLength

  // Range request for header
  const rangeResult = await client.send(
    new GetObjectCommand({ Bucket: bucket, Key: key, Range: 'bytes=0-4095' }),
  )
  const headerBytes = await rangeResult.Body!.transformToByteArray()
  const headerText = new TextDecoder().decode(headerBytes)
  const header = parseCHGCARHeader(headerText)
  onProgress?.({ phase: 'header', header, contentLength })

  // Full download
  onProgress?.({ phase: 'downloading', header, contentLength })
  const fullResult = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }))
  const fullBytes = await fullResult.Body!.transformToByteArray()
  const blob = new Blob([fullBytes])

  const filename = key.split('/').pop() ?? 'CHGCAR'
  onProgress?.({ phase: 'done', header, contentLength })

  return { blob, header, filename }
}
