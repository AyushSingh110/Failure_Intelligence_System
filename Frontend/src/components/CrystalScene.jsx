// ── CrystalScene — full-page scroll-scrubbed WebGL scene ─────────────────────
//
//  One persistent fixed canvas behind the whole landing page. The centerpiece
//  is the GUARDIAN CORE — a glowing AI core wrapped in two counter-rotating
//  geodesic shield shells. Incoming attack streaks fly in from off-screen and
//  are BLOCKED at the shield with a flash + shield pulse: the FIE story told
//  in one object.
//
//  The assembly travels a keyframed timeline as the user scrolls:
//
//    hero      → floats in the hero's right column
//    features  → tiny distant glint behind the capability cards
//    arch      → drifts to the left of the architecture diagram
//    pipeline  → swings to the right of the pipeline band
//    bench     → sinks far back behind the benchmark table
//    cta       → returns big and bright behind the final CTA
//
//  Keyframe stops are measured from real DOM section offsets (by element id),
//  so the timeline stays accurate regardless of content height. All motion is
//  frame-damped; the camera eases toward the cursor for parallax.
//
//  Rendering is skipped entirely for prefers-reduced-motion users.
// ─────────────────────────────────────────────────────────────────────────────

import { useRef, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Environment, Lightformer, Sparkles, Float } from '@react-three/drei'
import * as THREE from 'three'

// ── Scroll timeline ───────────────────────────────────────────────────────────
// xF = horizontal position as a fraction of the half-viewport (world units)
const POSES = [
  { id: 'hero',     xF:  0.52, y:  0.05, z:  0.0, s: 1.15, glow: 1.00, spin: 0.14, color: '#00d4ff' },
  // during the card deck the guardian retreats far into the distance —
  // a small glint between the cards, never on top of them
  { id: 'features', xF:  0.00, y: -0.05, z: -6.0, s: 0.45, glow: 0.25, spin: 0.30, color: '#7c8cff' },
  { id: 'arch',     xF: -0.70, y:  0.05, z: -2.8, s: 0.65, glow: 0.55, spin: 0.20, color: '#a78bfa' },
  { id: 'pipeline', xF:  0.60, y:  0.05, z: -2.2, s: 0.70, glow: 0.55, spin: 0.26, color: '#00d4ff' },
  { id: 'bench',    xF:  0.00, y:  0.18, z: -2.8, s: 0.62, glow: 0.42, spin: 0.36, color: '#67e8f9' },
  { id: 'cta',      xF:  0.00, y: -0.05, z:  0.9, s: 1.35, glow: 1.30, spin: 0.55, color: '#00ff88' },
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

// ── Attack interceptors ───────────────────────────────────────────────────────
// A small pool of hostile streaks that fly in from off-screen and die at the
// shield radius with a flash. Each impact pulses the shield via onImpact().
const ATTACK_N = 5
const SPAWN_R  = 6.2

function _spawnAttack(initial = false) {
  const dir = new THREE.Vector3().randomDirection()
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
  return {
    dir, quat,
    t: initial ? -Math.random() * 2.2 : -(0.5 + Math.random() * 2.0), // negative t = waiting to launch
    speed: 0.5 + Math.random() * 0.45,
  }
}

function Interceptors({ shieldR = 1.62, onImpact }) {
  const heads   = useRef([])
  const flashes = useRef([])
  const attacks = useRef(Array.from({ length: ATTACK_N }, () => _spawnAttack(true)))
  const flashLife = useRef(new Array(ATTACK_N).fill(0))

  useFrame((_, dt) => {
    attacks.current.forEach((a, i) => {
      const head = heads.current[i]
      const flash = flashes.current[i]
      if (!head || !flash) return

      a.t += dt * a.speed
      if (a.t >= 1) {
        // impact — flash at the shield, pulse it, respawn the attacker
        flashLife.current[i] = 1
        flash.position.copy(a.dir).multiplyScalar(shieldR)
        onImpact?.()
        attacks.current[i] = _spawnAttack()
      }

      if (a.t > 0 && a.t < 1) {
        const r = SPAWN_R + (shieldR - SPAWN_R) * a.t
        head.visible = true
        head.position.copy(a.dir).multiplyScalar(r)
        head.quaternion.copy(a.quat)
        // streak brightens as it closes in
        head.children.forEach(c => { c.material.opacity = 0.25 + a.t * 0.7 })
      } else {
        head.visible = false
      }

      // flash decay
      const l = flashLife.current[i]
      if (l > 0) {
        flashLife.current[i] = Math.max(l - dt * 2.6, 0)
        flash.visible = true
        flash.scale.setScalar(0.12 + (1 - l) * 0.55)
        flash.material.opacity = l * 0.85
      } else {
        flash.visible = false
      }
    })
  })

  return (
    <>
      {Array.from({ length: ATTACK_N }, (_, i) => (
        <group key={`a${i}`}>
          {/* streak: head + tail, oriented along the flight path */}
          <group ref={el => { heads.current[i] = el }} visible={false}>
            <mesh>
              <sphereGeometry args={[0.035, 10, 10]} />
              <meshBasicMaterial color="#ff4466" transparent opacity={0.9} blending={THREE.AdditiveBlending} depthWrite={false} />
            </mesh>
            <mesh position={[0, 0.3, 0]}>
              <cylinderGeometry args={[0.004, 0.016, 0.55, 6]} />
              <meshBasicMaterial color="#ff7755" transparent opacity={0.5} blending={THREE.AdditiveBlending} depthWrite={false} />
            </mesh>
          </group>
          {/* impact flash */}
          <mesh ref={el => { flashes.current[i] = el }} visible={false}>
            <sphereGeometry args={[1, 12, 12]} />
            <meshBasicMaterial color="#ff5566" transparent opacity={0} blending={THREE.AdditiveBlending} depthWrite={false} />
          </mesh>
        </group>
      ))}
    </>
  )
}

// ── Guardian Core — glowing model core inside counter-rotating shield shells ──
function GuardianCore({ stops }) {
  const group = useRef(null)      // timeline position/scale
  const spinner = useRef(null)    // continuous rotation
  const shieldA = useRef(null)
  const shieldB = useRef(null)
  const shieldAMat = useRef(null)
  const shieldBMat = useRef(null)
  const coreMat = useRef(null)
  const haloMat = useRef(null)
  const coreLight = useRef(null)
  const pulse = useRef(0)         // shield-hit feedback, decays each frame
  const { viewport } = useThree()

  const vel = useRef(0)
  const lastY = useRef(typeof window !== 'undefined' ? window.scrollY : 0)

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

    // narrow screens: keep the guardian centred and further back
    const narrow = viewport.aspect < 1
    const halfW = viewport.width / 2
    const targetX = (narrow ? pose.xF * 0.25 : pose.xF) * halfW
    const targetZ = pose.z - (narrow ? 1.2 : 0)

    g.position.x = THREE.MathUtils.damp(g.position.x, targetX, 3.2, dt)
    g.position.y = THREE.MathUtils.damp(g.position.y, pose.y, 3.2, dt)
    g.position.z = THREE.MathUtils.damp(g.position.z, targetZ, 3.2, dt)
    const k = THREE.MathUtils.damp(g.scale.x, pose.s, 3.2, dt)
    g.scale.setScalar(k)

    // assembly spin + velocity kick + slight pointer-follow tilt
    const sp = spinner.current
    if (sp) {
      sp.rotation.y += dt * (pose.spin + Math.min(Math.abs(vel.current) * 0.0004, 0.9))
      sp.rotation.x = THREE.MathUtils.damp(sp.rotation.x, Math.sin(t * 0.2) * 0.14 + pointer.y * 0.1, 2, dt)
      sp.rotation.z = THREE.MathUtils.damp(sp.rotation.z, THREE.MathUtils.clamp(-vel.current * 0.0001, -0.18, 0.18), 2, dt)
    }

    // shields counter-rotate independently of the assembly
    if (shieldA.current) { shieldA.current.rotation.y += dt * 0.22; shieldA.current.rotation.x += dt * 0.05 }
    if (shieldB.current) { shieldB.current.rotation.y -= dt * 0.15; shieldB.current.rotation.z += dt * 0.04 }

    // shield-hit pulse decays; brightens shield + core for a beat
    pulse.current = THREE.MathUtils.damp(pulse.current, 0, 4, dt)
    const p = pulse.current
    if (shieldAMat.current) shieldAMat.current.opacity = (0.22 + p * 0.5) * Math.min(pose.glow + 0.25, 1)
    if (shieldBMat.current) shieldBMat.current.opacity = (0.09 + p * 0.2) * Math.min(pose.glow + 0.25, 1)

    // glow color + intensity follow the timeline
    if (coreMat.current) {
      coreMat.current.emissive.copy(pose.color)
      coreMat.current.emissiveIntensity = 1.2 * pose.glow + p * 1.4 + Math.sin(t * 1.7) * 0.18
    }
    if (haloMat.current) {
      haloMat.current.color.copy(pose.color)
      haloMat.current.opacity = 0.13 * pose.glow + p * 0.2
    }
    if (coreLight.current) {
      coreLight.current.color.copy(pose.color)
      coreLight.current.intensity = 24 * pose.glow + p * 30
    }

    // camera parallax toward the cursor
    camera.position.x = THREE.MathUtils.damp(camera.position.x, pointer.x * 0.45, 2.5, dt)
    camera.position.y = THREE.MathUtils.damp(camera.position.y, pointer.y * 0.3, 2.5, dt)
    camera.lookAt(0, 0, 0)
  })

  return (
    <group ref={group}>
      <Float speed={1.2} rotationIntensity={0.14} floatIntensity={0.45}>
        <group ref={spinner}>
          {/* the protected model — glowing core */}
          <mesh>
            <icosahedronGeometry args={[0.58, 3]} />
            <meshStandardMaterial ref={coreMat} color="#0a1626" emissive="#00d4ff" emissiveIntensity={1.2} roughness={0.3} metalness={0.55} />
          </mesh>
          {/* soft halo around the core */}
          <mesh scale={1.16}>
            <icosahedronGeometry args={[0.58, 2]} />
            <meshBasicMaterial ref={haloMat} color="#00d4ff" transparent opacity={0.13} blending={THREE.AdditiveBlending} depthWrite={false} />
          </mesh>
          {/* inner shield — primary detection mesh */}
          <mesh ref={shieldA}>
            <icosahedronGeometry args={[1.45, 1]} />
            <meshBasicMaterial ref={shieldAMat} color="#00d4ff" wireframe transparent opacity={0.22} />
          </mesh>
          {/* outer shield — fine secondary lattice, counter-rotating */}
          <mesh ref={shieldB}>
            <icosahedronGeometry args={[1.62, 2]} />
            <meshBasicMaterial ref={shieldBMat} color="#a78bfa" wireframe transparent opacity={0.09} />
          </mesh>
          {/* faint energy skin between the shells */}
          <mesh>
            <sphereGeometry args={[1.53, 32, 32]} />
            <meshBasicMaterial color="#00d4ff" transparent opacity={0.025} blending={THREE.AdditiveBlending} depthWrite={false} />
          </mesh>
          <pointLight ref={coreLight} intensity={24} color="#00d4ff" distance={9} decay={2} />
        </group>
        <Interceptors shieldR={1.6} onImpact={() => { pulse.current = 1 }} />
        <Sparkles count={36} scale={4} size={2} speed={0.3} color="#9ecbff" opacity={0.4} />
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
      {/* local lightformer env — no network fetch, gives the metals something to reflect */}
      <Environment resolution={128}>
        <Lightformer intensity={2.2} position={[4, 3, 5]} scale={[6, 2, 1]} color="#00d4ff" />
        <Lightformer intensity={1.6} position={[-5, -2, -3]} scale={[5, 2, 1]} color="#a78bfa" />
        <Lightformer intensity={1.2} position={[0, 5, -4]} scale={[8, 1.5, 1]} color="#ffffff" />
      </Environment>
      <GuardianCore stops={stops} />
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
