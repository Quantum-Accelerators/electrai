export interface AWSCredentials {
  accessKeyId: string
  secretAccessKey: string
  sessionToken?: string
  expiration?: string
  region?: string
}

const STORAGE_KEY = 'elvis-aws-credentials'

export function loadCredentials(): AWSCredentials | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const creds: AWSCredentials = JSON.parse(raw)
    if (creds.expiration) {
      const exp = new Date(creds.expiration)
      if (exp.getTime() < Date.now()) return null
    }
    return creds
  } catch {
    return null
  }
}

export function saveCredentials(creds: AWSCredentials) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(creds))
}

export function clearCredentials() {
  localStorage.removeItem(STORAGE_KEY)
}
