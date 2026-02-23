import type { ScreenshotsMap, ScreencastConfig } from 'scrns'

// Run: pnpm scrns [-i <name>]
const screens: ScreenshotsMap = {
  'og-image': {
    query: '?m=mp-1000020&iso=571.4&si=48&c=178.8+31.3+11.5&xb&xa&lw=1.5',
    width: 1200,
    height: 630,
    selector: 'canvas',
    preScreenshotSleep: 5000,
  },
  'orbit-updown': {
    query: '?m=mp-1000020&iso=571.4&si=48&c=90+91.9+14.1+-180&a=6.0&xb&xa&do',
    width: 800,
    height: 600,
    selector: 'canvas',
    preScreenshotSleep: 5000,
    path: 'orbit-updown.gif',
    fps: 30,
    gifQuality: 10,
    actions: [
      { type: 'wait', duration: 500 },
      { type: 'key', key: 'ArrowUp', duration: 100 },
      { type: 'wait', duration: 7000 },
      { type: 'key', key: 'ArrowDown', duration: 100 },
      { type: 'wait', duration: 7000 },
    ],
  } satisfies ScreencastConfig,
}

export default screens
