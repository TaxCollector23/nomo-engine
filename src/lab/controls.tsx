import { useId, useState, type ReactNode } from "react";

/** A term with a plain-English explanation on hover, focus or tap. */
export function Info({ text, children }: { text: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="lab-info">
      <span tabIndex={0} className="lab-info-term" aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onClick={() => setOpen((o) => !o)}>
        {children}
      </span>
      {open && <span role="tooltip" id={id} className="lab-info-pop">{text}</span>}
    </span>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: string; children: ReactNode }) {
  return (
    <div className="lab-field">
      <div className="lab-field-label">{label}</div>
      {children}
      {hint && <div className="lab-field-hint">{hint}</div>}
    </div>
  );
}

export function Select({ value, onChange, options, label }: {
  value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label: string;
}) {
  return (
    <select className="lab-select" aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

/** Slider over a log scale (for tokens, budgets): shows a formatted value, optional "off" at the left end. */
export function LogSlider({ value, onChange, min, max, format, label, allowOff }: {
  value: number | null; onChange: (v: number | null) => void; min: number; max: number; format: (v: number) => string;
  label: string; allowOff?: boolean;
}) {
  const L = Math.log10(min), H = Math.log10(max), steps = 400;
  const pos = value === null ? 0 : Math.round(((Math.log10(Math.max(value, min)) - L) / (H - L)) * steps) + (allowOff ? 1 : 0);
  const total = steps + (allowOff ? 1 : 0);
  return (
    <div className="lab-slider">
      <div className="lab-slider-value">{value === null ? "no limit" : format(value)}</div>
      <input type="range" min={0} max={total} value={pos} aria-label={label}
        aria-valuetext={value === null ? "no limit" : format(value)}
        onChange={(e) => {
          const p = Number(e.target.value) - (allowOff ? 1 : 0);
          if (allowOff && p < 0) { onChange(null); return; }
          const v = 10 ** (L + ((H - L) * p) / steps);
          const mag = 10 ** Math.floor(Math.log10(v));
          onChange(Math.round(v / mag * 10) / 10 * mag);   // two significant figures
        }} />
    </div>
  );
}

export function LinSlider({ value, onChange, min, max, step, format, label }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number; format: (v: number) => string; label: string;
}) {
  return (
    <div className="lab-slider">
      <div className="lab-slider-value">{format(value)}</div>
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} aria-valuetext={format(value)}
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Chips<T extends string | number>({ options, value, onChange, label, format }: {
  options: T[]; value: T[]; onChange: (v: T[]) => void; label: string; format?: (v: T) => string;
}) {
  return (
    <div className="lab-chips" role="group" aria-label={label}>
      {options.map((o) => {
        const on = value.includes(o);
        return (
          <button key={String(o)} type="button" aria-pressed={on} className={on ? "is-on" : ""}
            onClick={() => {
              const next = on ? value.filter((x) => x !== o) : [...value, o];
              if (next.length) onChange(options.filter((x) => next.includes(x)));
            }}>
            {format ? format(o) : String(o)}
          </button>
        );
      })}
    </div>
  );
}

export function Segmented<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string; hint?: string }[]; label: string;
}) {
  return (
    <div className="lab-segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} title={o.hint}
          className={value === o.value ? "is-on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="lab-toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="lab-toggle-track" aria-hidden="true"><span /></span>
      <span>{label}</span>
    </label>
  );
}

export function fmtNum(v: number, unit: string): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  if (unit === "USD") return v >= 1e9 ? `$${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}k` : `$${v.toFixed(2)}`;
  if (unit.startsWith("USD /")) return `$${v < 0.1 ? v.toFixed(3) : v.toFixed(2)}`;
  if (unit === "days") return v >= 10 ? `${v.toFixed(0)} days` : `${v.toFixed(1)} days`;
  if (unit === "ms") return `${v < 10 ? v.toFixed(2) : v.toFixed(1)} ms`;
  if (unit === "nats/token") return v.toFixed(4);
  if (unit === "points") return `${Number(v.toPrecision(3))} pt${v === 1 ? "" : "s"}`;
  if (unit === "%") return `${v.toFixed(2)}%`;
  return `${Number(v.toPrecision(3))} ${unit}`;
}

export function fmtTokens(v: number): string {
  if (v >= 1e12) return `${(v / 1e12).toPrecision(3)}T`;
  if (v >= 1e9) return `${(v / 1e9).toPrecision(3)}B`;
  if (v >= 1e6) return `${(v / 1e6).toPrecision(3)}M`;
  return v.toLocaleString("en-US");
}
