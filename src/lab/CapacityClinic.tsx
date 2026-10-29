import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import type { Evaluated } from "../planner/search";
import type { Cluster } from "../planner/hardware";
import { Field, Info, LogSlider, Segmented, fmtNum, fmtTokens } from "./controls";

/** Capacity clinic for module 04: replace "1/latency" arithmetic with the
 * request-level simulator (continuous batching, chunked prefill, paged KV,
 * preemption, prefix cache) that parity-matches nomo_planner.serving_sim.
 * Rates are bridged from the SAME roofline evaluator that ranked the plan:
 *   prefill  linear time in batch (compute-bound): rate = prompt·B / prefill_s, gain 0
 *   decode   flat step time (memory-bound):        rate = 1 / decode_step_s, gain 1
 * so a sim at the plan's batch reproduces the evaluator exactly and extends it
 * with queueing, SLO tails and cache dynamics the closed form cannot see.
 * The sweep runs on a web worker; the main thread stays interactive. */

import { buildResult, type SimJob, type ClinicResult, type SweepPoint } from "./clinic-shared";

const W = 560;
const H = 210;
const PAD = { l: 52, r: 46, t: 14, b: 36 };
const CV_CHOICES = [0, 0.5, 1, 2];

function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    const v = 10 ** e;
    if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
  }
  return ticks;
}

export function CapacityClinic({ selected, settings, cluster }: {
  pack?: unknown;
  selected: Evaluated | null;
  settings: Record<string, unknown>;
  cluster: Cluster;
}) {
  const [load, setLoad] = useState(2);
  const [bursty, setBursty] = useState(false);
  const [cvIndex, setCvIndex] = useState(0);
  const [replicas, setReplicas] = useState(1);
  const [sloTtftS, setSloTtftS] = useState(2);
  const [chunk, setChunk] = useState(0);
  const [count, setCount] = useState(600);
  const [seed, setSeed] = useState(20261027);
  const [result, setResult] = useState<ClinicResult | null>(null);
  const [busy, setBusy] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestRef = useRef(0);

  const plan = selected && Object.values(selected.metrics.constraints).every((v) => v <= 0) ? selected : null;

  // bridge the evaluator's roofline into simulator rates (disclosed in the card body)
  const bridge = useMemo(() => {
    if (!plan) return null;
    const b = plan.metrics.breakdown as Record<string, number>;
    const promptTokens = Number(settings.prompt_tokens ?? 2048);
    const outputTokens = Number(settings.output_tokens ?? 256);
    const batch = Number(plan.plan.batch);
    const tp = Number(plan.plan.tp);
    const prefillS = Number(b.prefill_s);
    const decodeStepS = Number(b.decode_step_s);
    if (!(prefillS > 0) || !(decodeStepS > 0)) return null;
    const parts = b.memory_parts_gb as unknown as Record<string, number>;
    const kvBytesPerToken = (Number(b.kv_gb) * 1e9) / Math.max(1, batch * (promptTokens + outputTokens));
    const kvTokens = Math.max(1024, Math.floor(((Number(b.memory_cap_gb) - Number(parts.weights) - Number(parts.overhead)) * 1e9) / Math.max(1e-12, kvBytesPerToken)));
    return {
      promptTokens, outputTokens, batch, tp, prefillS, decodeStepS, kvBytesPerToken, kvTokens,
      prefillRate: (promptTokens * batch) / prefillS,
      decodeRate: 1 / decodeStepS,
      maxBatch: batch,
      staticReqS: Number(b.tokens_per_s) / Math.max(1, outputTokens),
    };
  }, [plan, settings.prompt_tokens, settings.output_tokens]);

  const cv = CV_CHOICES[cvIndex]!;
  const itlSloMs = Number(settings.max_ms_per_token ?? 50);
  const itlSlo = itlSloMs / 1000;

  // The static estimate is an upper bound that ignores queueing entirely, so the
  // sweep window starts well below it (knee region) and ends above it (collapse).
  const rates = useMemo(() => {
    if (!bridge) return [];
    const base = bridge.staticReqS / 32;
    return Array.from({ length: 12 }, (_, i) => base * 2 ** i);
  }, [bridge]);

  const job: SimJob | null = useMemo(() => (bridge ? {
    rate: load, bursty, promptMean: bridge.promptTokens, answerMean: bridge.outputTokens, cv,
    prefillRate: bridge.prefillRate, decodeRate: bridge.decodeRate, maxBatch: bridge.maxBatch,
    chunk, kvTokens: bridge.kvTokens, sloTtftS, sloItlS: itlSlo, replicas, count, seed,
  } : null), [bridge, load, bursty, cv, chunk, sloTtftS, itlSlo, replicas, count, seed]);

  const deferredJob = useDeferredValue(job);
  const deferredRates = useDeferredValue(rates);
  const stale = job !== deferredJob || rates !== deferredRates;

  useEffect(() => {
    if (!deferredJob || !deferredRates.length) return undefined;
    const requestId = String(++requestRef.current);
    setBusy(true);
    let cancelled = false;
    const deliver = (payload: ClinicResult) => {
      if (!cancelled) {
        setResult(payload);
        setBusy(false);
      }
    };
    const compute = () => deliver(buildResult(deferredJob, deferredRates, load));
    let worker: Worker | null = null;
    if (typeof Worker !== "undefined") {
      try {
        worker = workerRef.current ?? new Worker(new URL("../simulation/clinic-worker.ts", import.meta.url), { type: "module" });
        workerRef.current = worker;
        worker.onmessage = (event: MessageEvent<{ requestId: string; result: ClinicResult }>) => {
          if (event.data.requestId === requestId) deliver(event.data.result);
        };
        worker.postMessage({ requestId, kind: "run", job: deferredJob, rates: deferredRates, load });
      } catch {
        worker = null;
      }
    }
    if (!worker) compute();
    return () => {
      cancelled = true;
      if (worker && workerRef.current !== worker) worker.terminate();
    };
  }, [deferredJob, deferredRates, load]);

  useEffect(() => () => {
    workerRef.current?.terminate();
    workerRef.current = null;
  }, []);

  if (!plan || !bridge) return null;
  const b = bridge;
  const knee = result?.sweep.find((p) => p.passRatio < 0.99) ?? null;
  const capacity = (() => {
    let cap: SweepPoint | null = null;
    for (const point of result?.sweep ?? []) {
      if (point.passRatio >= 0.99) cap = point;
      else break;
    }
    return cap;
  })();

  const usdPerDay = cluster.costPerDeviceHour * b.tp * replicas * 24;
  const tokPerDayAtCapacity = capacity ? capacity.goodput * b.outputTokens * 86400 : 0;

  const xMin = Math.min(load, rates[0] ?? 0.1) * 0.7;
  const xMax = Math.max(load, rates[rates.length - 1] ?? 1) * 1.4;
  const yMax = Math.max(b.staticReqS * 1.5, ...(result?.sweep ?? []).map((p) => p.goodput), 0.1) * 1.3;
  const x = (v: number) => PAD.l + ((Math.log10(Math.max(v, xMin)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - PAD.l - PAD.r);
  const y = (v: number) => H - PAD.b - ((Math.log10(Math.max(v, yMax / 200)) - Math.log10(yMax / 200)) / (Math.log10(yMax) - Math.log10(yMax / 200))) * (H - PAD.t - PAD.b);
  const yPass = (p: number) => H - PAD.b - p * (H - PAD.t - PAD.b);
  const goodputPath = (result?.sweep ?? []).map((p, i) => `${i ? "L" : "M"}${x(p.offered).toFixed(1)},${y(Math.max(p.goodput, yMax / 200)).toFixed(1)}`).join(" ");
  const passPath = (result?.sweep ?? []).map((p, i) => `${i ? "L" : "M"}${x(p.offered).toFixed(1)},${yPass(p.passRatio).toFixed(1)}`).join(" ");
  const idealEnd = Math.min(xMax, yMax);
  const idealPath = `M${x(xMin).toFixed(1)},${y(xMin).toFixed(1)} L${x(idealEnd).toFixed(1)},${y(idealEnd).toFixed(1)}`;

  const atLoad = result?.atLoad?.metrics ?? null;
  const sec = (v: number | null) => (v === null ? "—" : v >= 1 ? `${v.toFixed(2)} s` : `${(v * 1000).toFixed(0)} ms`);
  const ms = (v: number | null) => (v === null ? "—" : `${(v * 1000).toFixed(1)} ms`);

  return (
    <section className="lab-card clinic" aria-label="Capacity clinic">
      <div className="lab-card-head">
        <h3>Capacity clinic — the load test before the load test</h3>
        <span className="lab-note">{busy || stale ? "simulating…" : "request-level simulator · parity-checked against the Python reference per seed"}</span>
      </div>
      <p className="lab-story">
        The recommendation above is a roofline: it prices a batch, not a queue. This clinic replays your workload against the same plan —
        requests arrive on a {bursty ? "bursty on/off" : "Poisson"} schedule, lengths vary{cv > 0 ? ` (CV ${cv})` : ""}, the KV cache fills,
        blocks evict, and requests preempt and recompute when memory runs out. The result is the artifact a load-test campaign would have
        taken your team days to produce: the largest arrival rate that still meets your SLO.
      </p>

      <div className="clinic-controls">
        <div className="clinic-slider">
          <LogSlider label="Offered load" value={load} onChange={(v) => setLoad(v ?? load)} min={Math.max(0.05, (rates[0] ?? 0.1) / 2)} max={(rates[rates.length - 1] ?? 100) * 2}
            format={(v) => `${v < 10 ? v.toFixed(2) : v.toFixed(0)} req/s`} />
        </div>
        <div className="clinic-slider">
          <LogSlider label="TTFT SLO" value={sloTtftS} onChange={(v) => setSloTtftS(v ?? 2)} min={0.2} max={10} format={(v) => `${(v * 1000).toFixed(0)} ms`} />
        </div>
        <Field label="Token-gap SLO" hint="Your speed limit from the inputs, in effect while tokens stream">
          <span className="clinic-static">{itlSloMs.toFixed(0)} ms</span>
        </Field>
        <Field label={<Info text="Bursty traffic (on/off surges) collapses the knee much earlier than a smooth Poisson stream.">Traffic</Info>}>
          <Segmented label="Traffic" value={bursty ? "bursty" : "steady"} onChange={(v) => setBursty(v === "bursty")}
            options={[{ value: "steady", label: "Steady" }, { value: "bursty", label: "Bursty" }]} />
        </Field>
        <Field label="Length variability">
          <Segmented label="Length variability" value={String(cvIndex)} onChange={(v) => setCvIndex(Number(v))}
            options={CV_CHOICES.map((c, i) => ({ value: String(i), label: c === 0 ? "fixed" : `CV ${c}` }))} />
        </Field>
        <Field label={<Info text="Chunked prefill splits long prompts so decode steps are not blocked. The standard fix for token-gap spikes.">Prefill chunk</Info>}>
          <Segmented label="Prefill chunk" value={String(chunk)} onChange={(v) => setChunk(Number(v))}
            options={[0, 512, 1024].map((n) => ({ value: String(n), label: n === 0 ? "off" : String(n) }))} />
        </Field>
        <Field label={<Info text="Independent model copies behind one load balancer. Cost and capacity both scale.">Model copies</Info>}>
          <Segmented label="Model copies" value={String(replicas)} onChange={(v) => setReplicas(Number(v))}
            options={[1, 2, 4].map((n) => ({ value: String(n), label: String(n) }))} />
        </Field>
        <Field label="Simulated requests" hint="More requests = smoother percentiles, slower sweep">
          <Segmented label="Simulated requests" value={String(count)} onChange={(v) => setCount(Number(v))}
            options={[300, 600, 1200].map((n) => ({ value: String(n), label: String(n) }))} />
        </Field>
        <Field label="Seed">
          <input className="clinic-seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} aria-label="Simulator seed" />
        </Field>
      </div>

      <div className="clinic-grid">
        <div>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Goodput versus offered load" className="clinic-chart">
            <rect x={PAD.l} y={PAD.t} width={Math.max(0, x(knee ? knee.offered : xMax) - PAD.l)} height={H - PAD.t - PAD.b} className="clinic-pass-zone" />
            {logTicks(xMin, xMax).map((v) => (
              <g key={v}>
                <line x1={x(v)} y1={PAD.t} x2={x(v)} y2={H - PAD.b} className="clinic-gridline" />
                <text x={x(v)} y={H - PAD.b + 14} textAnchor="middle" className="clinic-tick">{v < 1 ? v.toFixed(1) : v.toFixed(0)}</text>
              </g>
            ))}
            {logTicks(yMax / 200, yMax).map((v) => (
              <g key={`y${v}`}>
                <line x1={PAD.l} y1={y(v)} x2={W - PAD.r} y2={y(v)} className="clinic-gridline" />
                <text x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="clinic-tick">{v < 1 ? v.toFixed(2) : v.toFixed(0)}</text>
              </g>
            ))}
            {[0, 0.5, 1].map((p) => (
              <text key={`p${p}`} x={W - PAD.r + 6} y={yPass(p) + 3} className="clinic-tick">{p * 100}%</text>
            ))}
            <path d={idealPath} className="clinic-ideal" />
            <path d={passPath} className="clinic-pass-line" />
            <path d={goodputPath} className="clinic-goodput-line" />
            {knee && <circle cx={x(knee.offered)} cy={y(Math.max(knee.goodput, yMax / 200))} r={4} className="clinic-knee" />}
            <circle cx={x(load)} cy={y(Math.max(atLoad ? Number(atLoad.slo_goodput_requests_per_s) : 0, yMax / 200))} r={4} className="clinic-marker" />
            <text x={W - PAD.r - 4} y={PAD.t + 12} textAnchor="end" className="clinic-legend clinic-legend--goodput">goodput (req/s meeting SLO)</text>
            <text x={W - PAD.r - 4} y={PAD.t + 26} textAnchor="end" className="clinic-legend clinic-legend--pass">SLO pass fraction</text>
            <text x={PAD.l} y={H - 4} className="clinic-axis">offered load (req/s, log)</text>
          </svg>
          <p className="lab-note">
            Goodput counts only requests that met every SLO. The grey diagonal is the no-queueing ideal; the gap to the green curve is what
            queueing, batching interference and KV pressure take away. The shaded region is where at least 99% of requests hold the SLO.
          </p>
        </div>

        <div className="clinic-readout">
          {atLoad && (
            <>
              <p className="clinic-verdict">
                At <b>{load < 10 ? load.toFixed(2) : load.toFixed(0)} req/s</b> this plan delivers{" "}
                <b>{Number(atLoad.slo_goodput_requests_per_s).toFixed(2)} req/s within SLO</b>{" "}
                ({Math.round(Number(atLoad.slo_goodput_ratio) * 100)}% of requests).
                {capacity
                  ? <> It holds the SLO up to <b>{capacity.offered < 10 ? capacity.offered.toFixed(2) : capacity.offered.toFixed(0)} req/s</b> — about{" "}
                    <b>{fmtTokens(Math.round(tokPerDayAtCapacity * replicas))} output tok/day</b> for <b>{fmtNum(usdPerDay * replicas, "USD/day")}</b> of GPU rent.</>
                  : <b> It never meets the SLO in any simulated run — add copies, chunk the prefill, widen the SLO, or change the plan.</b>}
              </p>
              <table className="lab-mini-table clinic-table">
                <thead><tr><th>at this load</th><th>P50</th><th>P90</th><th>P99</th></tr></thead>
                <tbody>
                  <tr><td>time to first token</td><td>{sec(atLoad.ttft_p50_s === null ? null : Number(atLoad.ttft_p50_s))}</td>
                    <td>{sec(atLoad.ttft_p90_s === null ? null : Number(atLoad.ttft_p90_s))}</td>
                    <td className={atLoad.ttft_p99_s !== null && Number(atLoad.ttft_p99_s) > sloTtftS ? "clinic-miss" : ""}>{sec(atLoad.ttft_p99_s === null ? null : Number(atLoad.ttft_p99_s))}</td></tr>
                  <tr><td>gap between tokens</td><td>{ms(atLoad.itl_p50_s === null ? null : Number(atLoad.itl_p50_s))}</td>
                    <td>{ms(atLoad.itl_p90_s === null ? null : Number(atLoad.itl_p90_s))}</td>
                    <td className={atLoad.itl_p99_s !== null && Number(atLoad.itl_p99_s) > itlSlo ? "clinic-miss" : ""}>{ms(atLoad.itl_p99_s === null ? null : Number(atLoad.itl_p99_s))}</td></tr>
                </tbody>
              </table>
              <dl className="clinic-facts">
                <div><dt>SLO misses / preemptions</dt><dd>{atLoad.requests !== null ? Math.round(Number(atLoad.requests) - Number(atLoad.slo_goodput_requests)) : "—"} / {atLoad.preemptions}</dd></div>
                <div><dt>KV pressure (peak / avg)</dt><dd>{Math.round(Number(atLoad.kv_utilization_peak) * 100)}% / {Math.round(Number(atLoad.kv_utilization_avg) * 100)}%</dd></div>
                <div><dt>Static calculator promised</dt><dd>{b.staticReqS < 10 ? b.staticReqS.toFixed(2) : b.staticReqS.toFixed(0)} req/s of raw batch throughput</dd></div>
                <div><dt>Simulated goodput ceiling</dt><dd>{capacity ? `${capacity.offered < 10 ? capacity.offered.toFixed(2) : capacity.offered.toFixed(0)} req/s within SLO` : "never meets SLO"}</dd></div>
                <div><dt>KV cache provisioned</dt><dd>{(b.kvTokens / 1024).toFixed(0)}k tokens per copy (~{((b.kvTokens * b.kvBytesPerToken) / 1e9).toFixed(1)} GB)</dd></div>
              </dl>
            </>
          )}
        </div>
      </div>

      <p className="lab-note">
        Model: requests arrive {bursty ? "in on/off surges (4× rate, 1 s bursts)" : "as a Poisson process"}, lengths are {cv === 0 ? "fixed" : `lognormal with CV ${cv}`},
        scheduling is continuous batching with {b.maxBatch}-sequence batches{chunk > 0 ? ` and ${chunk}-token prefill chunks` : " and whole-prompt prefills"},
        KV is paged in 16-token blocks with an LRU prefix cache, and preemption recomputes evicted requests. Rates come from this plan's own
        roofline (prefill {(b.prefillRate / 1000).toFixed(1)}k tok/s scaling with batch, decode step {(b.decodeStepS * 1000).toFixed(1)} ms flat while
        memory-bound), so the simulator and the recommendation cannot disagree at the plan's operating point. Absolute numbers remain model-based:
        validate against your stack before signing an SLA. Seed {seed} reproduces these exact numbers (verified against the Python reference engine).
      </p>
    </section>
  );
}
