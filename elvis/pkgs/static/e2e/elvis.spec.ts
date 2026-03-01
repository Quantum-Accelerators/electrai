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

/** Wait for material to finish loading and three.js scene to initialize. */
async function waitForLoad(page: Page) {
  await expect(page.getByText('mp-1000020.json.gz').first()).toBeVisible({ timeout: 15000 })
  // Wait for the slice info to appear (proves grid data is parsed and maxSliceIndex is set)
  await expect(page.getByText(/Slice: \d+ \/ 31/)).toBeVisible({ timeout: 10000 })
}

/** Enter slice mode and wait for the keymap to be active. */
async function enterSliceMode(page: Page) {
  await page.locator('body').press('s')
  await expect(page.locator('.kbd-mode-indicator-label')).toHaveText('Slice')
  // Allow one animation frame for the effective keymap to recompute after mode change
  await page.waitForTimeout(100)
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

    // Press ArrowRight to orbit right by 90°
    await page.locator('body').press('ArrowRight')

    // Wait for ?c= to appear (first interaction sets it)
    await waitForParam(page, 'c', v => {
      if (!v) return false
      const parts = v.split(/[\s+]+/).map(Number)
      if (parts.length < 3) return false
      if (hadCamBefore) {
        const prevTheta = parseFloat(hadCamBefore.split(/[\s+]+/)[0])
        const curTheta = parts[0]
        return Math.abs(Math.abs(curTheta - prevTheta) - 90) < 5
      }
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

    // Shift+ArrowRight triggers pan
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
    await page.goto('/?si=16&a=0')

    await waitForLoad(page)
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    await enterSliceMode(page)

    // Press ArrowRight to step slice index
    await page.locator('body').press('ArrowRight')

    // si should change by exactly 1 (sign depends on camera angle;
    // for cubic cell at default camera with cam=null, sign=1 → si increases)
    await waitForParam(page, 'si', v => {
      if (v === null) return false
      const si = parseInt(v)
      return si === 17 || si === 15
    })
  })

  test('slice mode: step by 10', async ({ page }) => {
    await interceptS3(page)
    await page.goto('/?si=16&a=0')

    await waitForLoad(page)
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    await enterSliceMode(page)

    // Shift+ArrowRight to step by 10
    await page.locator('body').press('Shift+ArrowRight')

    // si should change by 10 (sign=1 → 26, or sign=-1 → 6)
    await waitForParam(page, 'si', v => {
      if (v === null) return false
      const si = parseInt(v)
      return si === 26 || si === 6
    })
  })

  // Direction-specific tests: verify sliceStepSign maps ArrowRight to the correct
  // screen direction by fixing the camera angle. The test fixture has an axis-aligned
  // 4Å cubic lattice, so the c-vector (sa=2) is (0,0,4).
  //
  // Zero-roll cases (roll=0):
  // At theta=−45° (c=-45+90+15+0): right=(cos−45,0,−sin−45)=(0.707,0,0.707)
  //   dot(c,right) = 4·0.707 = +2.83 > 0 → sign=1 → ArrowRight increases si
  // At theta=+45° (c=45+90+15+0): right=(cos45,0,−sin45)=(0.707,0,−0.707)
  //   dot(c,right) = 4·(−0.707) = −2.83 < 0 → sign=−1 → ArrowRight decreases si
  //
  // Non-zero roll case (roll=90°, catches the ±sin(ρ) sign in rolled-right formula):
  // At c=0+45+15+90 (theta=0, phi=45°, roll=90°):
  //   right_noroll = (1, 0, 0), up_noroll = (0, sin45, -cos45·0) ≈ (0, 0.707, 0)
  //   rolled_right = right·cos90 − up·sin90 = (0,0,0) − (0,0.707,0) = (0,−0.707,0)
  //   dot(c, rolled_right) = (0,0,4)·(0,−0.707,0) = 0 … but with non-zero phi:
  //   up_noroll = (0, sin45, -cos0·cos45) = (0, 0.707, -0.707)
  //   rolled_right = (1,0,0)·0 − (0,0.707,-0.707)·1 = (0, -0.707, 0.707)
  //   dot = 4·0.707 = +2.83 > 0 → sign=1 → ArrowRight increases si
  //   (Buggy +sin(ρ) formula would give (0,+0.707,−0.707), dot=−2.83, wrong sign)

  test('slice direction: ArrowRight increases si when dot > 0', async ({ page }) => {
    await interceptS3(page)
    // theta=-45 → c-axis projects onto screen-right → ArrowRight = +si
    await page.goto('/?si=16&a=0&c=-45+90+15+0')

    await waitForLoad(page)
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    await enterSliceMode(page)
    await page.locator('body').press('ArrowRight')

    await waitForParam(page, 'si', v => v === '17')
  })

  test('slice direction: ArrowRight decreases si when dot < 0', async ({ page }) => {
    await interceptS3(page)
    // theta=+45 → c-axis projects opposite to screen-right → ArrowRight = -si
    await page.goto('/?si=16&a=0&c=45+90+15+0')

    await waitForLoad(page)
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    await enterSliceMode(page)
    await page.locator('body').press('ArrowRight')

    await waitForParam(page, 'si', v => v === '15')
  })

  test('slice direction: non-zero roll (catches ±sin sign bug)', async ({ page }) => {
    await interceptS3(page)
    // theta=0, phi=45, roll=90 → rolled right has y<0 component from −up·sin(ρ)
    // c-axis = (0,0,4), rolled_right = (0,−0.707,0.707) → dot = +2.83 → sign=1
    // Buggy +sin(ρ) would give rolled_right = (0,+0.707,−0.707) → dot = −2.83 → wrong sign
    await page.goto('/?si=16&a=0&c=0+45+15+90')

    await waitForLoad(page)
    await waitForParam(page, 'si', v => v === '16', { timeout: 10000 })

    await enterSliceMode(page)
    await page.locator('body').press('ArrowRight')

    await waitForParam(page, 'si', v => v === '17')
  })
})
