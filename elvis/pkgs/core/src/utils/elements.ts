/**
 * CPK colors and covalent radii for common elements.
 * Colors are hex values; radii are in Angstroms.
 */
export interface ElementData {
  color: number
  radius: number
}

export const ELEMENTS: Record<string, ElementData> = {
  H:  { color: 0xffffff, radius: 0.31 },
  He: { color: 0xd9ffff, radius: 0.28 },
  Li: { color: 0xcc80ff, radius: 1.28 },
  Be: { color: 0xc2ff00, radius: 0.96 },
  B:  { color: 0xffb5b5, radius: 0.84 },
  C:  { color: 0x909090, radius: 0.76 },
  N:  { color: 0x3050f8, radius: 0.71 },
  O:  { color: 0xff0d0d, radius: 0.66 },
  F:  { color: 0x90e050, radius: 0.57 },
  Na: { color: 0xab5cf2, radius: 1.66 },
  Mg: { color: 0x8aff00, radius: 1.41 },
  Al: { color: 0xbfa6a6, radius: 1.21 },
  Si: { color: 0xf0c8a0, radius: 1.11 },
  P:  { color: 0xff8000, radius: 1.07 },
  S:  { color: 0xffff30, radius: 1.05 },
  Cl: { color: 0x1ff01f, radius: 1.02 },
  K:  { color: 0x8f40d4, radius: 2.03 },
  Ca: { color: 0x3dff00, radius: 1.76 },
  Ti: { color: 0xbfc2c7, radius: 1.60 },
  V:  { color: 0xa6a6ab, radius: 1.53 },
  Cr: { color: 0x8a99c7, radius: 1.39 },
  Mn: { color: 0x9c7ac7, radius: 1.39 },
  Fe: { color: 0xe06633, radius: 1.32 },
  Co: { color: 0xf090a0, radius: 1.26 },
  Ni: { color: 0x50d050, radius: 1.24 },
  Cu: { color: 0xc88033, radius: 1.32 },
  Zn: { color: 0x7d80b0, radius: 1.22 },
  Ga: { color: 0xc28f8f, radius: 1.22 },
  Ge: { color: 0x668f8f, radius: 1.20 },
  As: { color: 0xbd80e3, radius: 1.19 },
  Se: { color: 0xffa100, radius: 1.20 },
  Br: { color: 0xa62929, radius: 1.20 },
  Sr: { color: 0x00ff00, radius: 1.95 },
  Zr: { color: 0x94e0e0, radius: 1.75 },
  Mo: { color: 0x54b5b5, radius: 1.54 },
  Ag: { color: 0xc0c0c0, radius: 1.45 },
  Sn: { color: 0x668080, radius: 1.39 },
  Ba: { color: 0x00c900, radius: 2.15 },
  W:  { color: 0x2194d6, radius: 1.62 },
  Pt: { color: 0xd0d0e0, radius: 1.36 },
  Au: { color: 0xffd123, radius: 1.36 },
  Pb: { color: 0x575961, radius: 1.46 },
}

const DEFAULT_ELEMENT: ElementData = { color: 0xff1493, radius: 1.0 }

export function getElement(symbol: string): ElementData {
  return ELEMENTS[symbol] ?? DEFAULT_ELEMENT
}
