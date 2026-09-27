"use client";

import { useRunStore } from "@/lib/telemetry/store";

import { Term } from "./ui";

function Sparkline({ values, height = 40 }: { values: number[]; height?: number }) {
  if (values.length < 2) return <div style={{ height }} className="text-2xs text-ink-faint">Starts after round 2</div>;
  const w = 220;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${height - 2 - ((v - lo) / span) * (height - 4)}`);
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="block" aria-hidden>
      <polyline points={pts.join(" ")} fill="none" stroke="#2F5BEA" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const OP_NAMES: Record<string, string> = {
  segment_aligned: "combine whole sections", precision_uniform: "combine precisions", domain_flip: "change a layer's style",
  boundary_shift: "move a style boundary", precision_step: "change one precision", precision_cascade: "change a run of precisions",
  timestep: "change spike steps", coding_swap: "switch spike code", plasticity: "toggle on-chip learning",
  guard_impl: "move the safety guard",
};

/** Search progress in plain words, with the technical view one click away. */
export default function RunHUD() {
  const run = useRunStore((s) => s.run);
  const status = useRunStore((s) => s.status);
  const conn = useRunStore((s) => s.connection);
  const lastGen = useRunStore((s) => s.lastGen);
  const hv = useRunStore((s) => s.hv);
  const error = useRunStore((s) => s.error);
  const nFront = useRunStore((s) => s.front.size);
  const ops = Object.entries(lastGen?.operators ?? {}).sort((a, b) => b[1] - a[1]);
  const statusWord = { pending: "Starting", running: "Searching", completed: "Finished", stopped: "Stopped", failed: "Failed" }[status];
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="font-bold">{statusWord}</h2>
        {conn !== "open" && status === "running" && <span className="text-2xs text-ink-muted">reconnecting…</span>}
      </div>
      {error && <p role="alert" className="rounded-md bg-cross-tint p-2 text-sm">{error}</p>}
      <p className="text-sm text-ink-soft">
        Round {lastGen?.gen ?? 0} of {run?.generations ?? "?"}. {(lastGen?.unique ?? 0).toLocaleString()} different designs tried,
        {" "}{nFront} best trade-offs so far, {Math.round((lastGen?.feasible_fraction ?? 0) * 100)}% of the current round within your limits.
      </p>
      <div>
        <div className="mb-1 flex justify-between text-2xs text-ink-muted">
          <Term k="hypervolume">Search quality</Term><span>levels off when finished</span>
        </div>
        <Sparkline values={hv.map((p) => p.hv)} />
      </div>
      {ops.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-ink-muted">What the search is trying most</summary>
          <ul className="mt-2 space-y-1">
            {ops.slice(0, 6).map(([n, p]) => (
              <li key={n} className="flex items-center gap-2 text-2xs text-ink-muted">
                <span className="w-40 truncate">{OP_NAMES[n] ?? n}</span>
                <span className="h-1.5 flex-1 rounded bg-paper"><span className="block h-full rounded bg-ann" style={{ width: `${Math.min(100, p * 100)}%` }} /></span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
