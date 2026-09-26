"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { apiBase, type Catalog, type RunIn } from "@/lib/telemetry/protocol";

interface RunSummary { run_id: string; status: string; config: RunIn; last_gen: { gen?: number; hv?: number } }

const num = (s: string): number | null => (s.trim() === "" ? null : Number(s));

export default function Home() {
  const router = useRouter();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    model: "perception_cnn", hardware: "akd1500", drop: "4", energy: "", latency: "",
    pop: "64", gens: "60", seed: "0", plastic: "0",
  });

  useEffect(() => {
    const load = async () => {
      try {
        const [c, r] = await Promise.all([fetch(`${apiBase()}/catalog`), fetch(`${apiBase()}/runs`)]);
        setCatalog((await c.json()) as Catalog);
        setRuns((await r.json()) as RunSummary[]);
      } catch {
        setErr(`cannot reach the Nomo server at ${apiBase()} — start it with \`nomo serve\``);
      }
    };
    void load();
  }, []);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    const body: RunIn = {
      model: form.model, hardware: form.hardware, pop_size: Number(form.pop), generations: Number(form.gens),
      seed: Number(form.seed),
      budgets: { accuracy_drop_max: num(form.drop), energy_j: num(form.energy), latency_s: num(form.latency),
        min_plastic_params: Number(form.plastic) || 0 },
    };
    try {
      const r = await fetch(`${apiBase()}/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error(await r.text());
      const { run_id } = (await r.json()) as { run_id: string };
      router.push(`/runs/${run_id}`);
    } catch (x) {
      setErr(String(x));
      setBusy(false);
    }
  };

  const field = "w-full border border-neutral-800 bg-black px-2 py-1.5 text-sm text-neutral-100 focus:border-neutral-300 focus:outline-none";
  const label = "mb-1 block text-[10px] uppercase tracking-wider text-neutral-500";

  return (
    <main className="mx-auto max-w-4xl p-8 font-mono">
      <h1 className="text-2xl text-neutral-50">nomo</h1>
      <p className="mb-8 text-xs text-neutral-500">tri-domain (ANN · SNN · symbolic) hardware-aware NSGA-II</p>
      {err && <div className="mb-6 border border-neutral-600 p-3 text-xs text-neutral-300">{err}</div>}

      <form onSubmit={submit} className="grid grid-cols-3 gap-4 border border-neutral-900 p-5">
        <div>
          <label className={label}>model</label>
          <select className={field} value={form.model} onChange={set("model")}>
            {Object.keys(catalog?.models ?? { [form.model]: 0 }).map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className={label}>hardware</label>
          <select className={field} value={form.hardware} onChange={set("hardware")}>
            {Object.entries(catalog?.hardware ?? { [form.hardware]: { name: form.hardware } }).map(([k, v]) => (
              <option key={k} value={k}>{v.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={label}>max accuracy drop (pp)</label>
          <input className={field} value={form.drop} onChange={set("drop")} inputMode="decimal" />
        </div>
        <div>
          <label className={label}>energy budget (J, optional)</label>
          <input className={field} value={form.energy} onChange={set("energy")} placeholder="e.g. 7e-4" />
        </div>
        <div>
          <label className={label}>latency budget (s, optional)</label>
          <input className={field} value={form.latency} onChange={set("latency")} placeholder="e.g. 2e-3" />
        </div>
        <div>
          <label className={label}>min plastic params</label>
          <input className={field} value={form.plastic} onChange={set("plastic")} inputMode="numeric" />
        </div>
        <div>
          <label className={label}>population</label>
          <input className={field} value={form.pop} onChange={set("pop")} inputMode="numeric" />
        </div>
        <div>
          <label className={label}>generations</label>
          <input className={field} value={form.gens} onChange={set("gens")} inputMode="numeric" />
        </div>
        <div>
          <label className={label}>seed</label>
          <input className={field} value={form.seed} onChange={set("seed")} inputMode="numeric" />
        </div>
        <div className="col-span-3 flex items-center justify-between pt-2">
          <span className="text-[10px] text-neutral-600">
            hardware coefficients are placeholders until calibrated via the measurement LUT (SPEC §0)
          </span>
          <button disabled={busy} className="bg-neutral-100 px-5 py-2 text-sm text-black hover:bg-white disabled:opacity-40">
            {busy ? "starting…" : "start search"}
          </button>
        </div>
      </form>

      {runs.length > 0 && (
        <table className="mt-8 w-full text-xs">
          <thead className="text-left text-[10px] uppercase tracking-wider text-neutral-500">
            <tr><th className="py-1">run</th><th>model</th><th>hardware</th><th>status</th><th>gen</th><th>hv</th></tr>
          </thead>
          <tbody>
            {runs.slice().reverse().map((r) => (
              <tr key={r.run_id} className="border-t border-neutral-900 text-neutral-300">
                <td className="py-1.5"><Link className="underline hover:text-white" href={`/runs/${r.run_id}`}>{r.run_id}</Link></td>
                <td>{r.config.model}</td><td>{r.config.hardware}</td><td>{r.status}</td>
                <td>{r.last_gen?.gen ?? 0}</td><td>{(r.last_gen?.hv ?? 0).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
