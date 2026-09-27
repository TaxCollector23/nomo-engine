"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import HardwarePanel from "@/components/launcher/HardwarePanel";
import LayerLockTable from "@/components/launcher/LayerLockTable";
import UploadDrop from "@/components/launcher/UploadDrop";
import CalibrationDrop from "@/components/launcher/CalibrationDrop";
import { Button, Disclosure, NumberField, Slider, Term, Toggle } from "@/components/ui";
import { api, waitForBackend } from "@/lib/api";
import { MODEL_BLURB } from "@/lib/models";
import { PRESET_ORDER, applyPreset, defaultConfig, startRun } from "@/lib/runConfig";
import { apiBase, type CalibrationSummary, type Catalog, type LayerRow, type Preset, type RunIn, type SearchIn, type UploadedModel } from "@/lib/telemetry/protocol";

interface RunSummary { run_id: string; status: string; config: RunIn; created_at: number }

function chipBlurb(d: Record<string, number>): string {
  const n = d.n_cores ?? 0;
  const kb = d.sram_kb_per_core ?? 0;
  const mem = kb >= 1024 * 1024 ? `${Math.round(kb / 1024 / 1024)} GB` : kb >= 1024 ? `${Math.round(kb / 1024)} MB` : `${Math.round(kb)} KB`;
  return n <= 1 ? `One processor, ${mem} memory` : `${n} cores, ${mem} memory per core`;
}

function Step({ n, title, children, aside }: { n: number; title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="grid gap-4 border-t border-line py-8 md:grid-cols-[220px_1fr]">
      <div>
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-ink text-sm font-bold text-white">{n}</div>
        <h2 className="mt-3 text-lg font-bold">{title}</h2>
        {aside && <div className="mt-1 text-sm text-ink-muted">{aside}</div>}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}

function Choice({ selected, onClick, title, body, badge }: {
  selected: boolean; onClick: () => void; title: string; body: string; badge?: string;
}) {
  return (
    <button type="button" onClick={onClick} aria-pressed={selected}
      className={`w-full rounded-lg border p-4 text-left transition-colors ${selected ? "border-ink bg-panel shadow-[inset_0_0_0_1px_#1D2433]" : "border-line bg-panel hover:border-line-strong"}`}>
      <span className="flex items-start justify-between gap-2">
        <span className="font-bold">{title}</span>
        {badge && <span className="shrink-0 rounded-full bg-paper px-2 py-0.5 text-2xs text-ink-muted">{badge}</span>}
      </span>
      <span className="mt-1 block text-sm text-ink-muted">{body}</span>
    </button>
  );
}

export default function Home() {
  const router = useRouter();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [presets, setPresets] = useState<Record<string, Preset> | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [uploads, setUploads] = useState<UploadedModel[]>([]);
  const [calibrations, setCalibrations] = useState<Record<string, CalibrationSummary>>({});
  const [cfg, setCfg] = useState<RunIn>(defaultConfig());
  const [waking, setWaking] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const up = await waitForBackend((s) => !cancelled && setWaking(s));
      if (cancelled) return;
      setWaking(null);
      if (!up) {
        setErr(`The Nomo server at ${apiBase()} did not respond. If you run it yourself, start it with "python -m nomo.cli serve".`);
        return;
      }
      try {
        const [c, p, r] = await Promise.all([api("/catalog"), api("/presets"), api("/runs")]);
        const cat = (await c.json()) as Catalog;
        const pre = (await p.json()) as Record<string, Preset>;
        setCatalog(cat);
        setPresets(pre);
        setRuns(((await r.json()) as RunSummary[]).sort((a, b) => b.created_at - a.created_at));
        if (pre.balanced_edge) setCfg((c0) => applyPreset(c0, "balanced_edge", pre.balanced_edge!));
      } catch {
        setErr(`Cannot reach the Nomo server at ${apiBase()}.`);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const upload = uploads.find((u) => u.model_id === cfg.model);
  const layers: LayerRow[] = useMemo(
    () => upload?.layer_table ?? catalog?.models[cfg.model]?.layer_table ?? [], [upload, catalog, cfg.model]);
  const hw = catalog?.hardware[cfg.hardware];
  const search = cfg.search as SearchIn;
  const setSearch = (patch: Partial<SearchIn>) => setCfg((c) => ({ ...c, preset: null, search: { ...(c.search as SearchIn), ...patch } }));
  const setBudget = (patch: Partial<RunIn["budgets"]>) => setCfg((c) => ({ ...c, preset: null, budgets: { ...c.budgets, ...patch } }));
  const toggleCoding = (c: "rate" | "ttfs", on: boolean) =>
    setSearch({ codings: on ? Array.from(new Set([...search.codings, c])) : search.codings.filter((x) => x !== c) });

  const nothingAllowed = !search.allow_continuous && !search.allow_spiking && !search.allow_symbolic;
  const noCoding = search.allow_spiking && search.codings.length === 0;
  const lockCount = Object.keys(cfg.pins ?? {}).length;
  const hwCount = Object.keys(cfg.hardware_overrides ?? {}).length;

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      const id = await startRun(cfg);
      router.push(`/runs/${id}`);
    } catch (x) {
      setErr(x instanceof TypeError ? `Cannot reach the Nomo server at ${apiBase()}.` : x instanceof Error ? x.message : String(x));
      setBusy(false);
    }
  };

  const weights = search.asf_weights ?? [1, 1, 1];
  const setWeight = (i: number, v: number) => {
    const w = [...weights] as [number, number, number];
    w[i] = v;
    setSearch({ asf_weights: w });
  };

  return (
    <main className="mx-auto max-w-6xl px-5 pb-32 pt-10">
      <header className="flex items-start justify-between gap-6">
        <div>
          <p className="text-sm font-bold text-ann-ink">Nomo</p>
          <h1 className="mt-2 max-w-2xl text-3xl font-bold leading-tight md:text-4xl">
            Find the most energy-efficient way to run your AI on your chip.
          </h1>
          <p className="mt-3 max-w-2xl text-ink-muted">
            Nomo tries thousands of combinations of standard, spiking and physics-based layers, then shows you the best
            trade-offs between energy, speed and accuracy, ready to export.
          </p>
        </div>
        <Link href="/admin" className="text-sm text-ink-muted hover:text-ink">Admin</Link>
      </header>

      {waking !== null && (
        <p className="mt-6 rounded-lg border border-line bg-panel p-4 text-sm">
          Waking the server ({waking}s). Free hosting sleeps when idle; this can take up to a minute.
        </p>
      )}

      <div className="mt-8">
        <Step n={1} title="Choose a model" aside="Start with an example, or upload your own network.">
          <div className="grid gap-3 sm:grid-cols-2">
            {Object.keys(catalog?.models ?? {}).map((m) => (
              <Choice key={m} selected={cfg.model === m} onClick={() => setCfg((c) => ({ ...c, model: m, pins: {} }))}
                title={MODEL_BLURB[m]?.title ?? m} body={MODEL_BLURB[m]?.body ?? `${catalog!.models[m]!.layers.length} layers`}
                badge={`${catalog!.models[m]!.base_accuracy}% accurate`} />
            ))}
            {uploads.map((u) => (
              <Choice key={u.model_id} selected={cfg.model === u.model_id}
                onClick={() => setCfg((c) => ({ ...c, model: u.model_id, pins: {} }))}
                title={u.name} badge="uploaded"
                body={`${u.report.layers} layers, ${(u.report.params / 1e6).toFixed(2)} M weights, from ${u.report.source_format.replace(/_/g, " ")}`} />
            ))}
            <button type="button" onClick={() => setShowUpload((s) => !s)} aria-expanded={showUpload}
              className="rounded-lg border border-dashed border-line-strong p-4 text-left hover:border-ink">
              <span className="font-bold">Upload your own model</span>
              <span className="mt-1 block text-sm text-ink-muted">ONNX, PyTorch weights or a JSON graph</span>
            </button>
          </div>
          {showUpload && (
            <div className="mt-4">
              <UploadDrop onUploaded={(u) => {
                setUploads((us) => [u, ...us.filter((x) => x.model_id !== u.model_id)]);
                setCfg((c) => ({ ...c, model: u.model_id, pins: {} }));
                setShowUpload(false);
              }} />
            </div>
          )}
          {upload && upload.report.assumptions.length > 0 && (
            <div className="mt-4 rounded-lg border border-line bg-panel p-4 text-sm">
              <p className="font-bold">What Nomo read from {upload.name}</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-muted">
                {upload.report.assumptions.map((a) => <li key={a}>{a}</li>)}
              </ul>
            </div>
          )}
          <CalibrationDrop modelId={cfg.model} current={upload?.calibration ?? calibrations[cfg.model]} onAttached={(calibration) => {
            setCalibrations((all) => ({ ...all, [cfg.model]: calibration }));
            if (upload) setUploads((us) => us.map((u) => u.model_id === upload.model_id ? { ...u, calibration } : u));
          }} />
        </Step>

        <Step n={2} title="Choose a chip" aside="Energy and speed are estimated for this hardware.">
          <div className="grid gap-3 sm:grid-cols-3">
            {Object.entries(catalog?.hardware ?? {}).map(([k, h]) => (
              <Choice key={k} selected={cfg.hardware === k}
                onClick={() => setCfg((c) => ({ ...c, hardware: k, hardware_overrides: null, pins: {} }))}
                title={h.name} body={chipBlurb(h.defaults)} />
            ))}
          </div>
          {hw && (
            <div className="mt-4">
              <Disclosure title="Chip parameters" summary={hwCount ? `${hwCount} custom value${hwCount > 1 ? "s" : ""}` : "using built-in values"}>
                <HardwarePanel value={cfg.hardware_overrides} defaults={hw.defaults}
                  onChange={(v) => setCfg((c) => ({ ...c, hardware_overrides: v }))} />
              </Disclosure>
            </div>
          )}
        </Step>

        <Step n={3} title="What matters most?" aside="Pick a goal. You can adjust everything it sets in step 4.">
          <div className="grid gap-3 sm:grid-cols-2">
            {presets && PRESET_ORDER.map((k) => presets[k] && (
              <Choice key={k} selected={cfg.preset === k} onClick={() => setCfg((c) => applyPreset(c, k, presets[k]!))}
                title={presets[k]!.label} body={presets[k]!.summary} />
            ))}
          </div>
          {cfg.preset && presets?.[cfg.preset]?.notes && (
            <p className="mt-3 text-sm text-ink-muted">{presets[cfg.preset]!.notes}</p>
          )}
          {!cfg.preset && <p className="mt-3 text-sm text-ink-muted">Custom settings.</p>}
          {catalog?.modes && <label className="mt-4 block max-w-xl">
            <span className="mb-1 block text-sm font-bold">Operational mode</span>
            <select value={cfg.mode ?? ""} onChange={(e) => setCfg((c) => ({ ...c, preset: null, mode: e.target.value || null }))}
              className="w-full rounded-md border border-line bg-panel px-3 py-2 text-sm">
              <option value="">General co-design</option>
              {Object.entries(catalog.modes).map(([id, mode]) => <option key={id} value={id}>{mode.title}</option>)}
            </select>
            {cfg.mode && catalog.modes[cfg.mode] && <span className="mt-1 block text-sm text-ink-muted">{catalog.modes[cfg.mode]!.summary}</span>}
          </label>}
        </Step>

        <Step n={4} title="Fine-tune" aside="Optional. The defaults work well.">
          <div className="space-y-3">
            <Disclosure title="Limits" summary={`accuracy may drop by up to ${cfg.budgets.accuracy_drop_max ?? 0} points`} defaultOpen>
              <div className="grid gap-5 md:grid-cols-3">
                <Slider label={<Term k="accuracy_drop">Allowed accuracy drop</Term>} min={0.5} max={10} step={0.5}
                  value={cfg.budgets.accuracy_drop_max ?? 3} format={(v) => `${v} points`}
                  onChange={(v) => setBudget({ accuracy_drop_max: v })} />
                <NumberField label="Energy limit per decision (optional)" unit="µJ" placeholder="no limit"
                  value={cfg.budgets.energy_j ? cfg.budgets.energy_j * 1e6 : null}
                  onChange={(v) => setBudget({ energy_j: v ? v * 1e-6 : null })} />
                <NumberField label="Response-time limit (optional)" unit="ms" placeholder="no limit"
                  value={cfg.budgets.latency_s ? cfg.budgets.latency_s * 1e3 : null}
                  onChange={(v) => setBudget({ latency_s: v ? v * 1e-3 : null })} />
              </div>
            </Disclosure>

            <Disclosure title="Computing styles"
              summary={[search.allow_continuous && "continuous", search.allow_spiking && "spiking", search.allow_symbolic && "physics formulas"].filter(Boolean).join(", ") || "none"}>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <Toggle checked={search.allow_continuous} onChange={(v) => setSearch({ allow_continuous: v })}
                    label={<Term k="continuous">Continuous layers</Term>} hint="Standard neural-network layers" />
                  <Toggle checked={search.allow_spiking} onChange={(v) => setSearch({ allow_spiking: v })}
                    label={<Term k="spiking">Spiking layers</Term>} hint="Brain-style, event-driven layers" />
                  <Toggle checked={search.allow_symbolic} onChange={(v) => setSearch({ allow_symbolic: v })}
                    label={<Term k="symbolic">Physics formulas</Term>} hint="Replace layers with exact equations where possible" />
                  <Toggle checked={Boolean(cfg.lock_symbolic)} onChange={(v) => setCfg((c) => ({ ...c, preset: null, lock_symbolic: v }))}
                    label="Always use physics formulas where available" hint="Locks every layer that has a formula to it" />
                </div>
                <div className="space-y-4">
                  <fieldset disabled={!search.allow_spiking} className="space-y-2 disabled:opacity-50">
                    <legend className="mb-1 text-sm text-ink-soft">Spike codes the search may use</legend>
                    {(["rate", "ttfs"] as const).map((c) => (
                      <label key={c} className="flex items-center gap-2 text-sm">
                        <input type="checkbox" checked={search.codings.includes(c)} onChange={(e) => toggleCoding(c, e.target.checked)}
                          className="h-4 w-4 accent-[#2F5BEA]" />
                        <Term k={c}>{c === "rate" ? "Rate coding" : "Time-to-first-spike"}</Term>
                      </label>
                    ))}
                    <label className="flex items-center gap-2 text-sm text-ink-faint">
                      <input type="checkbox" disabled className="h-4 w-4" />
                      <Term k="phase">Phase coding</Term><span className="text-2xs">(not modelled yet)</span>
                    </label>
                  </fieldset>
                  <Slider label={<Term k="crossing_penalty">Discourage style changes</Term>} min={0} max={1} step={0.05}
                    value={search.crossing_penalty} onChange={(v) => setSearch({ crossing_penalty: v })}
                    format={(v) => (v === 0 ? "off" : `+${Math.round(v * 100)}% per crossing`)} />
                  <Slider label={<Term k="min_saving">Mixed designs must save at least</Term>} min={0} max={60} step={5}
                    value={search.crossing_min_saving_pct} onChange={(v) => setSearch({ crossing_min_saving_pct: v })}
                    format={(v) => (v === 0 ? "off" : `${v}%`)} />
                </div>
              </div>
              {nothingAllowed && <p role="alert" className="mt-4 text-sm text-cross">Allow at least one computing style.</p>}
              {noCoding && <p role="alert" className="mt-4 text-sm text-cross">Pick at least one spike code, or switch spiking layers off.</p>}
            </Disclosure>

            <Disclosure title="Layer locks" summary={lockCount ? `${lockCount} locked` : "none: Nomo decides every layer"}>
              {layers.length && hw ? (
                <LayerLockTable layers={layers} pins={cfg.pins ?? {}} bits={hw.bits}
                  onChange={(pins) => setCfg((c) => ({ ...c, pins }))} />
              ) : <p className="text-sm text-ink-muted">Loading layers…</p>}
            </Disclosure>

            <Disclosure title="Search settings" summary={`${cfg.pop_size} designs per round, up to ${cfg.generations} rounds`}>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <NumberField label={<Term k="population">Designs per round</Term>} value={cfg.pop_size} min={8}
                  max={catalog?.limits.max_pop} onChange={(v) => setCfg((c) => ({ ...c, pop_size: v ?? 64 }))}
                  help={catalog ? `up to ${catalog.limits.max_pop} on this server` : undefined} />
                <NumberField label={<Term k="generations">Rounds</Term>} value={cfg.generations} min={1}
                  max={catalog?.limits.max_generations} onChange={(v) => setCfg((c) => ({ ...c, generations: v ?? 60 }))}
                  help={catalog ? `up to ${catalog.limits.max_generations} on this server` : undefined} />
                <NumberField label={<Term k="patience">Stop after no progress for</Term>} unit="rounds" value={search.patience}
                  min={2} onChange={(v) => setSearch({ patience: v ?? 12 })} />
                <NumberField label={<Term k="archive">Trade-offs to keep</Term>} value={search.archive_capacity} min={0}
                  onChange={(v) => setSearch({ archive_capacity: v ?? 0 })} help="0 keeps all of them" />
                <Slider label={<Term k="crossover">Combine two parents</Term>} min={0} max={1} step={0.05}
                  value={search.p_crossover} onChange={(v) => setSearch({ p_crossover: v })} format={(v) => `${Math.round(v * 100)}%`} />
                <Slider label={<Term k="mutation">Add a random change</Term>} min={0} max={1} step={0.05}
                  value={search.p_mutation} onChange={(v) => setSearch({ p_mutation: v })} format={(v) => `${Math.round(v * 100)}%`} />
                <NumberField label="Random seed" value={cfg.seed} min={0} onChange={(v) => setCfg((c) => ({ ...c, seed: v ?? 0 }))}
                  help="Same seed and settings give the same result" />
              </div>
              <div className="mt-5">
                <p className="mb-2 text-sm text-ink-soft">How the recommended design is picked</p>
                <div className="grid gap-4 sm:grid-cols-3">
                  {(["Low energy", "Fast response", "High accuracy"] as const).map((lbl, i) => (
                    <Slider key={lbl} label={lbl} min={0.5} max={5} step={0.5} value={weights[i]!}
                      onChange={(v) => setWeight(i, v)} format={(v) => `×${v}`} />
                  ))}
                </div>
              </div>
            </Disclosure>
          </div>
        </Step>
      </div>

      {runs.length > 0 && (
        <section className="border-t border-line py-8">
          <h2 className="text-lg font-bold">Recent searches on this server</h2>
          <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-panel">
            {runs.slice(0, 8).map((r) => (
              <li key={r.run_id}>
                <Link href={`/runs/${r.run_id}`} className="flex items-center justify-between px-4 py-3 text-sm hover:bg-paper">
                  <span>{MODEL_BLURB[r.config.model]?.title ?? r.config.model} on {catalog?.hardware[r.config.hardware]?.name ?? r.config.hardware}</span>
                  <span className="text-ink-muted">{r.status}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-panel/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3">
          <p className="min-w-0 truncate text-sm text-ink-muted">
            {err ? <span role="alert" className="text-cross">{err}</span> : (
              <>
                {upload?.name ?? MODEL_BLURB[cfg.model]?.title ?? cfg.model} on {hw?.name ?? cfg.hardware}
                {cfg.preset && presets?.[cfg.preset] ? `, ${presets[cfg.preset]!.label}` : ", custom goal"}
                {lockCount ? `, ${lockCount} locked layer${lockCount > 1 ? "s" : ""}` : ""}
                {hwCount ? `, ${hwCount} custom chip value${hwCount > 1 ? "s" : ""}` : ""}
              </>
            )}
          </p>
          <Button onClick={submit} disabled={busy || waking !== null || !catalog || nothingAllowed || noCoding}>
            {busy ? "Starting…" : "Find the best designs"}
          </Button>
        </div>
      </div>
    </main>
  );
}
