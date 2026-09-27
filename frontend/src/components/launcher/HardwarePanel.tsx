"use client";

import type { HardwareIn } from "@/lib/telemetry/protocol";
import type { GLOSSARY } from "@/lib/glossary";

import { NumberField, Term } from "../ui";

const FIELDS: { key: keyof HardwareIn; label: string; unit: string; term?: keyof typeof GLOSSARY }[] = [
  { key: "mac_energy_pj", label: "Energy per multiply-add", unit: "pJ", term: "mac_energy" },
  { key: "sop_energy_pj", label: "Energy per spike event", unit: "pJ", term: "sop_energy" },
  { key: "neuron_energy_pj", label: "Energy per neuron update", unit: "pJ" },
  { key: "sram_kb_per_core", label: "Memory per core (SRAM)", unit: "KB", term: "sram" },
  { key: "n_cores", label: "Neuromorphic cores", unit: "cores", term: "cores" },
  { key: "neurons_per_core", label: "Neurons per core", unit: "neurons" },
  { key: "bus_bandwidth_gbs", label: "Bus bandwidth", unit: "GB/s", term: "bus_bandwidth" },
  { key: "routing_latency_us", label: "Spike routing latency", unit: "µs", term: "routing_latency" },
  { key: "timestep_us", label: "Spiking time step", unit: "µs" },
  { key: "static_power_mw", label: "Always-on power", unit: "mW", term: "static_power" },
  { key: "clock_mhz", label: "Clock frequency", unit: "MHz", term: "clock" },
];

function fmt(v: number | undefined): string {
  if (v === undefined) return "";
  return Math.abs(v) >= 1000 || Math.abs(v) < 0.01 ? v.toPrecision(3) : String(Number(v.toPrecision(4)));
}

/** Override any chip coefficient. Empty fields keep the built-in value shown as the placeholder. */
export default function HardwarePanel({ value, onChange, defaults }: {
  value: HardwareIn | null | undefined; onChange: (v: HardwareIn | null) => void; defaults: Record<string, number>;
}) {
  const v = value ?? {};
  const count = Object.values(v).filter((x) => x !== null && x !== undefined).length;
  const set = (k: keyof HardwareIn, x: number | null) => {
    const next = { ...v, [k]: x };
    if (x === null) delete next[k];
    onChange(Object.keys(next).length ? next : null);
  };
  return (
    <div>
      <p className="mb-4 text-sm text-ink-muted">
        Built-in values are <Term k="placeholder">placeholders</Term>. Enter your own chip&apos;s numbers to replace them;
        anything you enter is marked as yours in every report.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {FIELDS.map((f) => (
          <NumberField key={f.key} unit={f.unit} value={v[f.key] ?? null} onChange={(x) => set(f.key, x)}
            placeholder={fmt(defaults[f.key])} min={0}
            label={f.term ? <Term k={f.term}>{f.label}</Term> : f.label} />
        ))}
      </div>
      {count > 0 && (
        <button type="button" onClick={() => onChange(null)} className="mt-3 text-sm font-bold text-ann-ink hover:underline">
          Reset {count} value{count > 1 ? "s" : ""} to built-in
        </button>
      )}
    </div>
  );
}
