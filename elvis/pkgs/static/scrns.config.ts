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
  'slice-sweep': {
    query: '?c=28+59.1+13.4+1.9&iso=0.3&m=mp-1523390&si=35&sa=0&ss=20&xa&xb&sl&hc',
    width: 800,
    height: 600,
    selector: 'canvas',
    preScreenshotSleep: 5000,
    path: 'slice-sweep.gif',
    fps: 30,
    gifQuality: 10,
    actions: [
      { type: 'wait', duration: 500 },
      { type: 'key', key: 'Meta+Shift+ArrowLeft', duration: 100 },
      { type: 'wait', duration: 500 },
      { type: 'key', key: 'Meta+ArrowRight', duration: 100 },
      { type: 'wait', duration: 4000 },
      { type: 'key', key: 'Meta+ArrowLeft', duration: 100 },
      { type: 'wait', duration: 4000 },
    ],
  } satisfies ScreencastConfig,
  'tiling-slice-sweep': {
    query: '?iso=571.4&c=70.1+54.6+13.5+-88&od=30&si=0&a=0.5&tp=0.5&ss=20&ct=0.426+-0.56+-0.00378&sl',
    width: 800,
    height: 600,
    selector: 'canvas',
    preScreenshotSleep: 5000,
    path: 'tiling-slice-sweep.gif',
    fps: 30,
    gifQuality: 10,
    actions: [
      { type: 'wait', duration: 500 },
      { type: 'key', key: 'Meta+Shift+ArrowLeft', duration: 100 },
      { type: 'wait', duration: 500 },
      { type: 'key', key: 'Meta+ArrowRight', duration: 100 },
      { type: 'wait', duration: 6000 },
      { type: 'key', key: 'Meta+ArrowLeft', duration: 100 },
      { type: 'wait', duration: 6000 },
    ],
  } satisfies ScreencastConfig,
}

export default screens
