import { test, expect, type Page, type Route } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const FIXTURE_PATH = join(__dirname, 'fixtures', 'mp-test.json.gz')
const fixtureBytes = readFileSync(FIXTURE_PATH)

/** Intercept S3 requests and serve the local fixture. */
async function interceptS3(page: Page) {
  await page.route('**/materialsproject-parsed.s3.amazonaws.com/**', (route: Route) => {
    const method = route.request().method()
    if (method === 'HEAD') {
      return route.fulfill({
        status: 200,
        headers: {
          'content-length': String(fixtureBytes.length),
          'content-type': 'application/gzip',
        },
      })
    }
    return route.fulfill({
      status: 200,
      headers: { 'content-type': 'application/gzip' },
      body: Buffer.from(fixtureBytes),
    })
  })
}

/** Parse URL search params from current page URL. */
function urlParams(page: Page): URLSearchParams {
  return new URL(page.url()).searchParams
}

/** Wait for a URL param to satisfy a predicate (handles debounced updates). */
async function waitForParam(
  page: Page,
  key: string,
  predicate: (value: string | null) => boolean,
  { timeout = 5000 } = {},
) {
  await expect(async () => {
    const val = urlParams(page).get(key)
    expect(predicate(val)).toBe(true)
  }).toPass({ timeout })
}

/** Wait for material to finish loading. */
async function waitForLoad(page: Page) {
  await expect(page.getByText('mp-1000020.json.gz').first()).toBeVisible({ timeout: 15000 })
  // Give the app a moment to settle after rendering (URL debounces, state propagation)
  await page.waitForTimeout(500)
}

test.describe('Elvis E2E', () => {
  test('load default material', async ({ page }) => {
    await interceptS3(page)
    await page.goto('/')

    await waitForLoad(page)

    // Canvas should be visible (first = three.js, second = slice preview)
    await expect(page.locator('canvas').first()).toBeVisible()

    // Element badges: Fe and O
    await expect(page.getByText('Fe').first()).toBeVisible()
    await expect(page.getByText('O').first()).toBeVisible()

    // Default material → no ?m= param in URL (it's the default)
    expect(urlParams(page).get('m')).toBeNull()
  })

  test('orbit camera', async ({ page }) => {
    await interceptS3(page)
    // od=90 (large discrete step), a=0.1 (fast animation; a=0 causes NaN in snap interpolation)
    await page.goto('/?od=90&a=0.1')

    await waitForLoad(page)

    // Record whether ?c= exists before interaction
    const hadCamBefore = urlParams(page).get('c')

    // Press ArrowRight to orbit right by 90° — dispatch directly on document body
    // to match how use-kbd listens for keyboard events
    await page.locator('body').press('ArrowRight')

    // Wait for ?c= to appear (first interaction sets it)
    await waitForParam(page, 'c', v => {
      if (!v) return false
      const parts = v.split(/[\s+]+/).map(Number)
      if (parts.length < 3) return false
      // If we had a previous value, check theta changed
      if (hadCamBefore) {
        const prevTheta = parseFloat(hadCamBefore.split(/[\s+]+/)[0])
        const curTheta = parts[0]
        return Math.abs(Math.abs(curTheta - prevTheta) - 90) < 5
      }
      // Otherwise just verify it appeared with valid numbers
      return parts.every(isFinite)
    }, { timeout: 5000 })
  })

  test('pan camera', async ({ page }) => {
    await interceptS3(page)
    // pd=1 (discrete pan step), a=0.1 (fast animation; a=0 causes NaN in snap interpolation)
    await page.goto('/?pd=1&a=0.1')

    await waitForLoad(page)

    // No ct= initially (null by default)
    expect(urlParams(page).get('ct')).toBeNull()

    // Shift+ArrowRight triggers pan — dispatch on document body
    await page.locator('body').press('Shift+ArrowRight')

    // Wait for ct= to appear with a valid 3-tuple
    await waitForParam(page, 'ct', v => {
      if (!v) return false
      const parts = v.trim().split(/[\s+]+/).map(Number)
      return parts.length === 3 && parts.every(isFinite)
    }, { timeout: 5000 })
  })

  test('slice mode: step by 1', async ({ page }) => {
    await interceptS3(page)
    // Pre-set si=16 to avoid relying on auto-set timing
    await page.goto('/?si=16&a=0')

    await waitForLoad(page)

    // Confirm si=16
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })
    const initialSi = 16

    // Press 's' to enter slice mode
    await page.locator('body').press('s')

    // Mode indicator should show "Slice"
    await expect(page.locator('.kbd-mode-indicator-label')).toHaveText('Slice')

    // Press ArrowRight to step +1
    await page.locator('body').press('ArrowRight')

    // si should change by exactly 1 (direction depends on sliceStepSign,
    // but for a cubic cell at default camera, sign=1 → si increases)
    await waitForParam(page, 'si', v => {
      if (v === null) return false
      const si = parseInt(v)
      return si === initialSi + 1 || si === initialSi - 1
    })
  })

  test('slice mode: step by 10', async ({ page }) => {
    await interceptS3(page)
    await page.goto('/?si=16&a=0')

    await waitForLoad(page)

    // Confirm si=16 from URL
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    // Enter slice mode
    await page.locator('body').press('s')
    await expect(page.locator('.kbd-mode-indicator-label')).toHaveText('Slice')

    // Shift+ArrowRight to step by 10
    await page.locator('body').press('Shift+ArrowRight')

    // si should change by 10 (or clamp to 0/31)
    await waitForParam(page, 'si', v => {
      if (v === null) return false
      const si = parseInt(v)
      // Could be 26 (16+10) or 6 (16-10), depending on sign
      return si === 26 || si === 6
    })
  })
})
