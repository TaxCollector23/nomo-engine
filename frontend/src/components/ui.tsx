"use client";

import { useId, useState } from "react";

import { GLOSSARY } from "@/lib/glossary";

/** A technical term with a plain-English explanation on hover or keyboard focus. */
export function Term({ k, children }: { k: keyof typeof GLOSSARY; children?: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const g = GLOSSARY[k];
  if (!g) return <>{children}</>;
  return (
    <span className="relative inline-block">
      <span tabIndex={0} aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
        onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        className="cursor-help border-b border-dotted border-ink-faint">
        {children ?? g.title}
      </span>
      {open && (
        <span role="tooltip" id={id}
          className="absolute left-0 top-full z-50 mt-1.5 block w-72 rounded-md border border-line bg-panel p-3 text-sm font-normal normal-case leading-snug tracking-normal text-ink-soft shadow-lg">
          <span className="mb-1 block font-bold text-ink">{g.title}</span>
          {g.body}
        </span>
      )}
    </span>
  );
}

export function Toggle({ checked, onChange, label, hint, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; label: React.ReactNode; hint?: string; disabled?: boolean;
}) {
  return (
    <label className={`flex items-start gap-3 ${disabled ? "opacity-50" : ""}`}>
      <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={`mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${checked ? "bg-ann" : "bg-line-strong"}`}>
        <span className={`h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-[18px]" : "translate-x-0.5"}`} />
      </button>
      <span className="text-sm">
        <span className="text-ink">{label}</span>
        {hint && <span className="block text-2xs text-ink-muted">{hint}</span>}
      </span>
    </label>
  );
}

export function NumberField({ label, value, onChange, unit, placeholder, step, min, max, help }: {
  label: React.ReactNode; value: number | null | undefined; onChange: (v: number | null) => void; unit?: string;
  placeholder?: string; step?: number; min?: number; max?: number; help?: string;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm text-ink-soft">{label}</label>
      <span className="flex items-center rounded-md border border-line bg-panel focus-within:border-ann">
        <input id={id} type="number" inputMode="decimal" step={step ?? "any"} min={min} max={max}
          value={value ?? ""} placeholder={placeholder} aria-describedby={help ? `${id}-help` : undefined}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
          className="w-full min-w-0 rounded-md bg-transparent px-2.5 py-1.5 text-sm text-ink outline-none placeholder:text-ink-faint" />
        {unit && <span className="pr-2.5 text-2xs text-ink-muted">{unit}</span>}
      </span>
      {help && <span id={`${id}-help`} className="mt-0.5 block text-2xs text-ink-muted">{help}</span>}
    </div>
  );
}

export function Slider({ label, value, onChange, min, max, step, format }: {
  label: React.ReactNode; value: number; onChange: (v: number) => void; min: number; max: number; step: number;
  format?: (v: number) => string;
}) {
  const id = useId();
  return (
    <div>
      <span className="mb-1 flex justify-between text-sm text-ink-soft">
        <label htmlFor={id}>{label}</label>
        <span className="font-bold text-ink">{format ? format(value) : value}</span>
      </span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        aria-valuetext={format ? format(value) : String(value)} className="w-full" />
    </div>
  );
}

export function Disclosure({ title, summary, children, defaultOpen }: {
  title: string; summary?: string; children: React.ReactNode; defaultOpen?: boolean;
}) {
  return (
    <details open={defaultOpen} className="group rounded-lg border border-line bg-panel">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3">
        <span>
          <span className="font-bold text-ink">{title}</span>
          {summary && <span className="ml-2 text-sm text-ink-muted">{summary}</span>}
        </span>
        <span aria-hidden className="text-ink-muted transition-transform group-open:rotate-90">›</span>
      </summary>
      <div className="border-t border-line px-4 py-4">{children}</div>
    </details>
  );
}

export const DOMAIN_META = {
  ANN: { word: "Continuous", cls: "bg-ann-tint text-ann-ink border-ann", dot: "bg-ann", term: "continuous" as const },
  SNN: { word: "Spiking", cls: "bg-snn-tint text-snn-ink border-snn", dot: "bg-snn", term: "spiking" as const },
  SYM: { word: "Physics formula", cls: "bg-sym-tint text-sym-ink border-sym", dot: "bg-sym", term: "symbolic" as const },
};

export function DomainChip({ d, small }: { d: "ANN" | "SNN" | "SYM"; small?: boolean }) {
  const m = DOMAIN_META[d];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 ${small ? "py-0 text-2xs" : "py-0.5 text-xs"} ${m.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} />{m.word}
    </span>
  );
}

export function Button({ children, onClick, kind = "primary", disabled, type = "button", title }: {
  children: React.ReactNode; onClick?: () => void; kind?: "primary" | "secondary" | "quiet"; disabled?: boolean;
  type?: "button" | "submit"; title?: string;
}) {
  const cls = {
    primary: "bg-ink text-white hover:bg-ink-soft",
    secondary: "border border-line-strong bg-panel text-ink hover:border-ink",
    quiet: "text-ann-ink hover:underline",
  }[kind];
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title}
      className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${cls}`}>
      {children}
    </button>
  );
}

export function formatPct(v: number, digits = 0): string {
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}%`;
}
