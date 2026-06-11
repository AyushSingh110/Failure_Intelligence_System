// ── CrystalScene — full-page scroll-scrubbed WebGL scene ─────────────────────
//
//  One persistent fixed canvas behind the whole landing page. A faceted glass
//  crystal travels through a keyframed timeline as the user scrolls:
//
//    hero      → floats in the hero's right column
//    features  → recedes deep behind the capability cards
//    arch      → drifts to the left of the architecture diagram
//    pipeline  → swings to the right of the pipeline band
//    bench     → sinks far back behind the benchmark table
//    cta       → returns huge and bright behind the final CTA
//
//  Keyframe stops are measured from real DOM section offsets (by element id),
//  so the timeline stays accurate regardless of content height. All motion is
//  frame-damped — scrolling fast spins the crystal harder; everything settles
//  softly. The camera eases toward the cursor for parallax.
//
//  Rendering pauses for prefers-reduced-motion users.
// ─────────────────────────────────────────────────────────────────────────────

import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { MeshTransmissionMaterial, Environment, Lightformer, Sparkles, Float } from '@react-three/drei'
import * as THREE from 'three'

// ── Scroll timeline ───────────────────────────────────────────────────────────
// xF = horizontal position as a fraction of the half-viewport (world units)
const POSES = [
  { id: 'hero',     xF:  0.52, y:  0.05, z:  0.0, s: 1.30, glow: 1.00, spin: 0.14, color: '#00d4ff' },
  { id: 'features', xF:  0.00, y:  0.30, z: -2.6, s: 0.78, glow: 0.50, spin: 0.30, color: '#7c8cff' },
  { id: 'arch',     xF: -0.56, y:  0.00, z: -0.9, s: 1.05, glow: 0.78, spin: 0.20, color: '#a78bfa' },
  { id: 'pipeline', xF:  0.56, y:  0.00, z: -1.4, s: 0.88, glow: 0.62, spin: 0.26, color: '#00d4ff' },
  { id: 'bench',    xF:  0.00, y:  0.18, z: -2.8, s: 0.70, glow: 0.42, spin: 0.36, color: '#67e8f9' },
  { id: 'cta',      xF:  0.00, y: -0.05, z:  0.9, s: 1.60, glow: 1.30, spin: 0.55, color: '#00ff88' },
]

const _colA = new THREE.Color()
const _colB = new THREE.Color()

function lerpPose(a, b, t) {
  const e = t * t * (3 - 2 * t) // smoothstep
  return {
    xF:   a.xF   + (b.xF   - a.xF)   * e,
    y:    a.y    + (b.y    - a.y)    * e,
    z:    a.z    + (b.z    - a.z)    * e,
    s:    a.s    + (b.s    - a.s)    * e,
    glow: a.glow + (b.glow - a.glow) * e,
    spin: a.spin + (b.spin - a.spin) * e,
    color: _colA.set(a.color).lerp(_colB.set(b.color), e),
  }
}

// Measures section anchor offsets so keyframes track the real layout.
function useSectionStops() {
  const stops = useRef(null)
  useEffect(() => {
    const measure = () => {
      const vh = window.innerHeight
      stops.current = POSES.map(p => {
        const el = document.getElementById(`s3d-${p.id}`)
        return el ? Math.max(el.offsetTop - vh * 0.45, 0) : null
      })
    }
    measure()
    window.addEventListener('resize', measure)
    const iv = setInterval(measure, 2500) // re-measure as lazy content settles
    return () => { window.removeEventListener('resize', measure); clearInterval(iv) }
  }, [])
  return stops
}

function currentPose(stops, scrollYpx) {
  if (!stops) return { ...POSES[0], color: _colA.set(POSES[0].color) }
  // collect valid (offset, pose) pairs in order
  const pts = []
  for (let i = 0; i < POSES.length; i++) if (stops[i] != null) pts.push([stops[i], POSES[i]])
  if (pts.length === 0) return { ...POSES[0], color: _colA.set(POSES[0].color) }
  if (scrollYpx <= pts[0][0]) return { ...pts[0][1], color: _colA.set(pts[0][1].color) }
  for (let i = 0; i < pts.length - 1; i++) {
    const [y0, p0] = pts[i]
    const [y1, p1] = pts[i + 1]
    if (scrollYpx < y1) return lerpPose(p0, p1, (scrollYpx - y0) / Math.max(y1 - y0, 1))
  }
  const last = pts[pts.length - 1][1]
  return { ...last, color: _colA.set(last.color) }
}

// ── Crystal ───────────────────────────────────────────────────────────────────
function Crystal({ stops }) {
  const group = useRef(null)      // timeline position/scale
  const spinner = useRef(null)    // continuous rotation
  const coreMat = useRef(null)
  const coreLight = useRef(null)
  const edgesMat = useRef(null)
  const { viewport } = useThree()

  const vel = useRef(0)
  const lastY = useRef(typeof window !== 'undefined' ? window.scrollY : 0)

  const gemGeo = useMemo(() => {
    const g = new THREE.IcosahedronGeometry(1, 0)
    g.scale(1, 1.38, 1) // elongate into a gem
    return g
  }, [])
  const edgesGeo = useMemo(() => new THREE.EdgesGeometry(gemGeo), [gemGeo])

  useFrame(({ camera, pointer, clock }, dt) => {
    const g = group.current
    if (!g) return
    const t = clock.elapsedTime
    const yPx = window.scrollY

    // scroll velocity (px/s), damped — drives extra spin + tilt
    const rawVel = (yPx - lastY.current) / Math.max(dt, 1e-4)
    lastY.current = yPx
    vel.current = THREE.MathUtils.damp(vel.current, rawVel, 3, dt)

    const pose = currentPose(stops.current, yPx)

    // narrow screens: keep the crystal centred and further back
    const narrow = viewport.aspect < 1
    const halfW = viewport.width / 2
    const targetX = (narrow ? pose.xF * 0.25 : pose.xF) * halfW
    const targetZ = pose.z - (narrow ? 1.2 : 0)

    g.position.x = THREE.MathUtils.damp(g.position.x, targetX, 3.2, dt)
    g.position.y = THREE.MathUtils.damp(g.position.y, pose.y, 3.2, dt)
    g.position.z = THREE.MathUtils.damp(g.position.z, targetZ, 3.2, dt)
    const k = THREE.MathUtils.damp(g.scale.x, pose.s, 3.2, dt)
    g.scale.setScalar(k)

    // continuous spin + velocity kick + slight pointer-follow tilt
    const sp = spinner.current
    if (sp) {
      sp.rotation.y += dt * (pose.spin + Math.min(Math.abs(vel.current) * 0.0004, 0.9))
      sp.rotation.x = THREE.MathUtils.damp(sp.rotation.x, Math.sin(t * 0.23) * 0.18 + pointer.y * 0.12, 2, dt)
      sp.rotation.z = THREE.MathUtils.damp(sp.rotation.z, THREE.MathUtils.clamp(-vel.current * 0.00012, -0.22, 0.22), 2, dt)
    }

    // glow color + intensity follow the timeline
    if (coreMat.current) {
      coreMat.current.color.copy(pose.color)
      coreMat.current.opacity = 0.5 + 0.25 * Math.sin(t * 1.8) * pose.glow
    }
    if (coreLight.current) {
      coreLight.current.color.copy(pose.color)
      coreLight.current.intensity = 26 * pose.glow
    }
    if (edgesMat.current) edgesMat.current.opacity = 0.1 + 0.1 * pose.glow

    // camera parallax toward the cursor
    camera.position.x = THREE.MathUtils.damp(camera.position.x, pointer.x * 0.45, 2.5, dt)
    camera.position.y = THREE.MathUtils.damp(camera.position.y, pointer.y * 0.3, 2.5, dt)
    camera.lookAt(0, 0, 0)
  })

  return (
    <group ref={group}>
      <Float speed={1.3} rotationIntensity={0.18} floatIntensity={0.5}>
        <group ref={spinner}>
          {/* glass gem */}
          <mesh geometry={gemGeo}>
            <MeshTransmissionMaterial
              transmission={1}
              thickness={1.4}
              roughness={0.08}
              ior={1.45}
              chromaticAberration={0.35}
              anisotropicBlur={0.25}
              distortion={0.18}
              distortionScale={0.4}
              temporalDistortion={0.12}
              attenuationColor="#a78bfa"
              attenuationDistance={2.2}
              samples={4}
              resolution={384}
              flatShading
            />
          </mesh>
          {/* facet edges */}
          <lineSegments geometry={edgesGeo} scale={1.002}>
            <lineBasicMaterial ref={edgesMat} color="#dceaff" transparent opacity={0.16} />
          </lineSegments>
          {/* inner energy core */}
          <mesh scale={0.42}>
            <icosahedronGeometry args={[1, 1]} />
            <meshBasicMaterial ref={coreMat} color="#00d4ff" transparent opacity={0.6} blending={THREE.AdditiveBlending} depthWrite={false} />
          </mesh>
          <pointLight ref={coreLight} intensity={26} color="#00d4ff" distance={8} decay={2} />
        </group>
        <Sparkles count={48} scale={4.4} size={2.2} speed={0.35} color="#bfd9ff" opacity={0.5} />
      </Float>
    </group>
  )
}

// ── Depth starfield with scroll parallax ─────────────────────────────────────
// Positions generated once at module load — stable per session.
function _genShell(count, rMin, rMax, ySquash) {
  const arr = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    const r = rMin + Math.pow(Math.random(), 0.7) * (rMax - rMin)
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta)
    arr[i * 3 + 1] = r * Math.cos(phi) * ySquash
    arr[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta)
  }
  return arr
}
const STARS_NEAR = _genShell(500, 4, 9, 1)
const STARS_FAR  = _genShell(800, 9, 18, 1)

function StarLayer({ positions, size, opacity, drift, parallax }) {
  const ref = useRef(null)
  useFrame((_, dt) => {
    const p = ref.current
    if (!p) return
    p.rotation.y += dt * drift
    // scroll parallax — layers rise at different speeds
    const prog = window.scrollY / Math.max(document.documentElement.scrollHeight - window.innerHeight, 1)
    p.position.y = THREE.MathUtils.damp(p.position.y, prog * parallax, 2.5, dt)
  })
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={size} color="#9ecbff" transparent opacity={opacity} sizeAttenuation blending={THREE.AdditiveBlending} depthWrite={false} />
    </points>
  )
}

function Scene() {
  const stops = useSectionStops()
  return (
    <>
      <ambientLight intensity={0.35} />
      {/* local lightformer env — no network fetch, drives the glass reflections */}
      <Environment resolution={128}>
        <Lightformer intensity={2.2} position={[4, 3, 5]} scale={[6, 2, 1]} color="#00d4ff" />
        <Lightformer intensity={1.6} position={[-5, -2, -3]} scale={[5, 2, 1]} color="#a78bfa" />
        <Lightformer intensity={1.2} position={[0, 5, -4]} scale={[8, 1.5, 1]} color="#ffffff" />
        <Lightformer intensity={0.8} position={[2, -4, 2]} scale={[4, 1.5, 1]} color="#00ff88" />
      </Environment>
      <Crystal stops={stops} />
      <StarLayer positions={STARS_NEAR} size={0.035} opacity={0.5}  drift={0.012} parallax={5} />
      <StarLayer positions={STARS_FAR}  size={0.025} opacity={0.32} drift={0.006} parallax={2.4} />
    </>
  )
}

export default function CrystalScene() {
  const [reducedMotion] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
  // skip WebGL entirely for reduced-motion users — the page stands on its own
  if (reducedMotion) return null

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none' }} aria-hidden="true">
      <Canvas
        camera={{ position: [0, 0, 7], fov: 38 }}
        dpr={[1, 1.6]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
      >
        <Scene />
      </Canvas>
    </div>
  )
}
