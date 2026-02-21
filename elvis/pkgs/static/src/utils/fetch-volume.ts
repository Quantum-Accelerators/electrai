import { S3Client, HeadObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3'
import { parseCHGCARHeader } from '@elvis/core'
import type { CHGCARHeader } from '@elvis/core'
import type { AWSCredentials } from './aws-credentials.ts'

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
    region: 'us-east-1',
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
