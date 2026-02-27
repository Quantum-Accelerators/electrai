import { Matrix4 } from 'three'
import type { LatticeMatrix } from '../types.ts'
import { latticeToMatrix4 } from './lattice.ts'

type ShaderObj = {
  uniforms: Record<string, { value: unknown }>
  vertexShader: string
  fragmentShader: string
}

/**
 * Returns an onBeforeCompile callback that injects per-fragment distance-based
 * fade into any Three.js material shader (MeshStandard, MeshBasic, LineDashed, etc.)
 *
 * Fragments beyond `padding` distance (in fractional lattice coords) from the
 * primary cell [0,1]³ are discarded. When fade=true, opacity fades linearly
 * from 1 at the cell boundary to 0 at the padding edge.
 */
export function tileFadeCompile(
  lattice: LatticeMatrix,
  padding: number,
  fade: boolean,
): (shader: ShaderObj) => void {
  const invLat = new Matrix4().copy(latticeToMatrix4(lattice)).invert()

  return (shader: ShaderObj) => {
    shader.uniforms.uTFInvLat = { value: invLat }
    shader.uniforms.uTFPad = { value: padding }
    shader.uniforms.uTFFade = { value: fade ? 1.0 : 0.0 }

    // Vertex: compute world position and pass to fragment
    shader.vertexShader = shader.vertexShader.replace(
      '#include <common>',
      '#include <common>\nvarying vec3 vTFWP;',
    )
    shader.vertexShader = shader.vertexShader.replace(
      '#include <project_vertex>',
      `#include <project_vertex>
{vec4 _w=vec4(transformed,1.0);
#ifdef USE_INSTANCING
_w=instanceMatrix*_w;
#endif
vTFWP=(modelMatrix*_w).xyz;}`,
    )

    // Fragment: compute fractional distance from primary cell, fade + discard
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <common>',
      `#include <common>
varying vec3 vTFWP;
uniform mat4 uTFInvLat;
uniform float uTFPad;
uniform float uTFFade;`,
    )
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <premultiplied_alpha_fragment>',
      `#include <premultiplied_alpha_fragment>
{vec3 _f=(uTFInvLat*vec4(vTFWP,1.0)).xyz;
float _d=max(max(0.0,max(-_f.x,_f.x-1.0)),max(max(0.0,max(-_f.y,_f.y-1.0)),max(0.0,max(-_f.z,_f.z-1.0))));
if(_d>=uTFPad)discard;
if(uTFFade>0.5)gl_FragColor.a*=1.0-_d/uTFPad;}`,
    )
  }
}
