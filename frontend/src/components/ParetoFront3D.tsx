"use client";

import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { bounds, formatSI, ticks, toScene, type Bounds } from "@/lib/pareto";
import { useRunStore } from "@/lib/telemetry/store";

const CAP_STEP = 4096;
type Klass = "front" | "population" | "archive" | "infeasible";
const STYLE: Record<Klass, { color: string; scale: number }> = {
  front: { color: "#ffffff", scale: 0.042 },
  population: { color: "#8f8f8f", scale: 0.028 },
  archive: { color: "#474747", scale: 0.016 },
  infeasible: { color: "#262626", scale: 0.014 },
};

function useKeysAndBounds() {
  const version = useRunStore((s) => s.version);
  const items = useRunStore((s) => s.items);
  return useMemo(() => {
    const keys = Array.from(items.keys());
    return { keys, b: bounds(items.values()) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, items]);
}

function Points({ keys, b }: { keys: string[]; b: Bounds }) {
  const items = useRunStore((s) => s.items);
  const front = useRunStore((s) => s.front);
  const population = useRunStore((s) => s.population);
  const version = useRunStore((s) => s.version);
  const select = useRunStore((s) => s.select);
  const ref = useRef<THREE.InstancedMesh>(null);
  const capacity = Math.max(CAP_STEP, Math.ceil(keys.length / CAP_STEP) * CAP_STEP);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color();
    keys.forEach((k, i) => {
      const it = items.get(k)!;
      const klass: Klass = !it.feasible ? "infeasible" : front.has(k) ? "front" : population.has(k) ? "population" : "archive";
      const st = STYLE[klass];
      pos.set(...toScene(it, b));
      scl.setScalar(st.scale);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, col.set(st.color));
    });
    mesh.count = keys.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [keys, b, items, front, population, version]);

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (e.instanceId !== undefined && keys[e.instanceId]) select(keys[e.instanceId]!);
  };

  return (
    <instancedMesh key={capacity} ref={ref} args={[undefined, undefined, capacity]} onClick={onClick}>
      <sphereGeometry args={[1, 10, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function Marker({ k, b, radius, dashed }: { k: string | null; b: Bounds; radius: number; dashed?: boolean }) {
  const it = useRunStore((s) => (k ? s.items.get(k) : undefined));
  useRunStore((s) => s.version);
  if (!it) return null;
  return (
    <mesh position={toScene(it, b)}>
      <sphereGeometry args={[radius, 16, 12]} />
      <meshBasicMaterial color={dashed ? "#bdbdbd" : "#ffffff"} wireframe />
    </mesh>
  );
}

function Axes({ b }: { b: Bounds }) {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), []);
  const e = ticks(b.lo[0], b.hi[0], true);
  const l = ticks(b.lo[1], b.hi[1], true);
  const a = ticks(b.lo[2], b.hi[2], false);
  const label = "pointer-events-none whitespace-nowrap font-mono text-[10px] text-neutral-400";
  return (
    <group>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color="#303030" />
      </lineSegments>
      {e.map((t) => (
        <Html key={`e${t.at}`} position={[t.at, -1.08, 1.08]} center className={label}>{formatSI(t.label, "J")}</Html>
      ))}
      {l.map((t) => (
        <Html key={`l${t.at}`} position={[1.1, -1.08, t.at]} center className={label}>{formatSI(t.label, "s")}</Html>
      ))}
      {a.map((t) => (
        <Html key={`a${t.at}`} position={[-1.12, t.at, 1.08]} center className={label}>{t.label.toFixed(1)}%</Html>
      ))}
      <Html position={[0, -1.3, 1.25]} center className={`${label} text-neutral-200`}>energy / inference (log)</Html>
      <Html position={[1.35, -1.3, 0]} center className={`${label} text-neutral-200`}>latency (log)</Html>
      <Html position={[-1.3, 1.2, 1.1]} center className={`${label} text-neutral-200`}>accuracy</Html>
    </group>
  );
}

export default function ParetoFront3D() {
  const { keys, b } = useKeysAndBounds();
  const selected = useRunStore((s) => s.selectedKey);
  const recommended = useRunStore((s) => s.recommendedKey);
  return (
    <div className="relative h-full w-full bg-black">
      <Canvas camera={{ position: [3.1, 2.1, 3.3], fov: 40 }} dpr={[1, 2]} onPointerMissed={() => undefined}>
        {b && (
          <>
            <Axes b={b} />
            <Points keys={keys} b={b} />
            <Marker k={recommended} b={b} radius={0.09} dashed />
            <Marker k={selected} b={b} radius={0.065} />
          </>
        )}
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
      </Canvas>
      <div className="pointer-events-none absolute bottom-3 left-3 space-y-1 font-mono text-[10px] text-neutral-400">
        {(Object.keys(STYLE) as Klass[]).map((k) => (
          <div key={k} className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: STYLE[k].color, outline: "1px solid #555" }} />
            {k}
          </div>
        ))}
        <div className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full border border-dashed border-neutral-300" /> recommended (ASF)
        </div>
        <div className="pt-1 text-neutral-500">{keys.length.toLocaleString()} candidates · drag to orbit · click to inspect</div>
      </div>
    </div>
  );
}
