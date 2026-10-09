"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { api } from "@/lib/api";
import { apiBase } from "@/lib/telemetry/protocol";

const STREAMS = ["all", "backend", "access", "users", "runs", "telemetry", "errors"] as const;
type Stream = (typeof STREAMS)[number];
type Tab = "logs" | "users" | "runs" | "stats";
const TOKEN_KEY = "nomo.admin_token";
const TOKEN_EVENT = "nomo.admin_token_changed";

function subscribeAdminToken(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(TOKEN_EVENT, onChange);
  return () => window.removeEventListener(TOKEN_EVENT, onChange);
}

function readAdminToken(): string {
  try { return sessionStorage.getItem(TOKEN_KEY) ?? ""; } catch { return ""; }
}

function readServerAdminToken(): string {
  return "";
}

interface LogRecord { ts: number; time: string; level: string; stream: string; event: string; [k: string]: unknown }
type Json = Record<string, unknown>;

const LEVEL_STYLE: Record<string, string> = {
  DEBUG: "text-neutral-600", INFO: "text-neutral-300", WARNING: "text-neutral-100 underline decoration-dotted",
  ERROR: "bg-neutral-100 text-black px-1", CRITICAL: "bg-neutral-100 text-black px-1",
};
const CORE = new Set(["ts", "time", "level", "stream", "event"]);

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 23);
}

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export default function AdminPage() {
  const token = useSyncExternalStore(subscribeAdminToken, readAdminToken, readServerAdminToken);
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState<Tab>("logs");
  const [stream, setStream] = useState<Stream>("all");
  const [level, setLevel] = useState("");
  const [contains, setContains] = useState("");
  const [live, setLive] = useState(true);
  const [records, setRecords] = useState<LogRecord[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [data, setData] = useState<Json | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const lastTs = useRef<number | undefined>(undefined);

  const authed = useCallback(async (path: string): Promise<Response> => {
    const r = await api(path, { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
    if (r.status === 401 || r.status === 403) {
      const d = (await r.json().catch(() => ({}))) as { detail?: string };
      throw new Error(d.detail ?? `HTTP ${r.status}`);
    }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r;
  }, [token]);

  const logQuery = useMemo(() => {
    const q = new URLSearchParams({ stream, limit: "1000" });
    if (level) q.set("level", level);
    if (contains.trim()) q.set("contains", contains.trim());
    return q;
  }, [stream, level, contains]);

  // full reload when filters change
  useEffect(() => {
    if (!token || tab !== "logs") return;
    lastTs.current = undefined;
    authed(`/admin/logs?${logQuery}`)
      .then((r) => r.json())
      .then((d: { records: LogRecord[] }) => {
        setRecords(d.records);
        lastTs.current = d.records.at(-1)?.ts;
        setErr(null);
      })
      .catch((e: Error) => setErr(e.message));
  }, [token, tab, logQuery, authed]);

  // live tail: fetch only records newer than the last one seen
  useEffect(() => {
    if (!token || tab !== "logs" || !live) return;
    const id = setInterval(() => {
      const q = new URLSearchParams(logQuery);
      if (lastTs.current !== undefined) q.set("since_ts", String(lastTs.current));
      authed(`/admin/logs?${q}`)
        .then((r) => r.json())
        .then((d: { records: LogRecord[] }) => {
          if (!d.records.length) return;
          lastTs.current = d.records.at(-1)?.ts;
          setRecords((prev) => [...prev, ...d.records].slice(-3000));
        })
        .catch(() => undefined);
    }, 3000);
    return () => clearInterval(id);
  }, [token, tab, live, logQuery, authed]);

  // users / runs / stats tabs
  useEffect(() => {
    if (!token || tab === "logs") return;
    const load = () =>
      authed(`/admin/${tab}`)
        .then((r) => r.json())
        .then((d: Json) => {
          setData(d);
          setErr(null);
        })
        .catch((e: Error) => setErr(e.message));
    void load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [token, tab, authed]);

  const download = async () => {
    try {
      const r = await authed(`/admin/logs/download?${new URLSearchParams({ ...Object.fromEntries(logQuery), limit: "50000" })}`);
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `nomo-${stream}-${new Date().toISOString().replace(/[:.]/g, "-")}.ndjson`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const signIn = (e: React.FormEvent) => {
    e.preventDefault();
    sessionStorage.setItem(TOKEN_KEY, draft.trim());
    window.dispatchEvent(new Event(TOKEN_EVENT));
  };

  const box = "border border-neutral-800 bg-black px-2 py-1 text-xs text-neutral-100 focus:border-neutral-300 focus:outline-none";

  if (!token) {
    return (
      <main className="mx-auto max-w-md p-10 font-mono">
        <h1 className="mb-1 text-xl text-neutral-50">nomo · admin</h1>
        <p className="mb-6 text-xs text-neutral-500">backend: {apiBase()}</p>
        <form onSubmit={signIn} className="space-y-3">
          <input type="password" className={`${box} w-full py-2`} placeholder="NOMO_ADMIN_TOKEN" value={draft}
            onChange={(e) => setDraft(e.target.value)} autoFocus />
          <button className="w-full bg-neutral-100 py-2 text-sm text-black hover:bg-white">sign in</button>
        </form>
        <p className="mt-4 text-[10px] text-neutral-600">
          The token is the NOMO_ADMIN_TOKEN environment variable on the server. It is kept in this tab only.
        </p>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-5 font-mono text-neutral-200">
      <header className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-xs text-neutral-500 hover:text-neutral-200">← nomo</Link>
          <h1 className="text-lg text-neutral-50">admin</h1>
          <nav className="flex gap-1">
            {(["logs", "users", "runs", "stats"] as Tab[]).map((t) => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-3 py-1 text-xs ${tab === t ? "bg-neutral-100 text-black" : "border border-neutral-800 text-neutral-400 hover:text-neutral-100"}`}>
                {t}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-neutral-500">
          <span>{apiBase()}</span>
          <button className="border border-neutral-800 px-2 py-0.5 hover:text-neutral-200"
            onClick={() => { sessionStorage.removeItem(TOKEN_KEY); setDraft(""); window.dispatchEvent(new Event(TOKEN_EVENT)); }}>sign out</button>
        </div>
      </header>
      {err && <div className="mb-3 border border-neutral-600 p-2 text-xs">{err}</div>}

      {tab === "logs" && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {STREAMS.map((s) => (
              <button key={s} onClick={() => setStream(s)}
                className={`px-2 py-0.5 text-[11px] ${stream === s ? "bg-neutral-200 text-black" : "border border-neutral-800 text-neutral-400"}`}>{s}</button>
            ))}
            <select className={box} value={level} onChange={(e) => setLevel(e.target.value)}>
              <option value="">all levels</option><option>INFO</option><option>WARNING</option><option>ERROR</option>
            </select>
            <input className={`${box} w-56`} placeholder="search (run id, client, event…)" value={contains}
              onChange={(e) => setContains(e.target.value)} />
            <label className="flex items-center gap-1 text-[11px] text-neutral-400">
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} /> live tail
            </label>
            <button onClick={download} className="border border-neutral-700 px-2 py-0.5 text-[11px] hover:border-neutral-300">download .ndjson</button>
            <span className="text-[10px] text-neutral-600">{records.length} records</span>
          </div>
          <div className="overflow-x-auto border border-neutral-900">
            <table className="w-full text-[11px]">
              <thead className="text-left text-[10px] uppercase tracking-wider text-neutral-500">
                <tr><th className="px-2 py-1">time (UTC)</th><th>level</th><th>stream</th><th>event</th><th>fields</th></tr>
              </thead>
              <tbody>
                {records.slice().reverse().map((r, i) => {
                  const fields = Object.entries(r).filter(([k]) => !CORE.has(k));
                  return (
                    <tr key={`${r.ts}-${i}`} className="cursor-pointer border-t border-neutral-900 align-top hover:bg-neutral-950"
                      onClick={() => setExpanded(expanded === i ? null : i)}>
                      <td className="whitespace-nowrap px-2 py-1 text-neutral-500">{fmtTime(r.ts)}</td>
                      <td className="pr-2"><span className={LEVEL_STYLE[r.level] ?? ""}>{r.level}</span></td>
                      <td className="pr-2 text-neutral-500">{r.stream}</td>
                      <td className="whitespace-nowrap pr-3 text-neutral-100">{r.event}</td>
                      <td className="text-neutral-400">
                        {expanded === i ? (
                          <pre className="whitespace-pre-wrap break-all text-[10px]">{JSON.stringify(Object.fromEntries(fields), null, 2)}</pre>
                        ) : (
                          <span className="line-clamp-1 break-all">
                            {fields.slice(0, 8).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`).join("  ")}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {tab === "users" && data && (
        <table className="w-full text-xs">
          <thead className="text-left text-[10px] uppercase tracking-wider text-neutral-500">
            <tr><th className="py-1">client</th><th>first seen</th><th>last seen</th><th>requests</th><th>runs</th><th>sockets</th><th>ip hash</th><th>origin</th><th>user agent</th></tr>
          </thead>
          <tbody>
            {((data.users as Json[]) ?? []).map((u) => (
              <tr key={String(u.client_id)} className="border-t border-neutral-900 align-top">
                <td className="py-1 pr-2 text-neutral-100">{String(u.client_id)}</td>
                <td className="pr-2">{ago(Number(u.first_seen))}</td>
                <td className="pr-2">{ago(Number(u.last_seen))}</td>
                <td>{String(u.requests)}</td><td>{String(u.runs_started)}</td><td>{String(u.ws_sessions)}</td>
                <td className="pr-2 text-neutral-500">{String(u.ip_hash)}</td>
                <td className="pr-2 text-neutral-500">{String(u.origin)}</td>
                <td className="max-w-xs truncate text-neutral-600" title={String(u.user_agent)}>{String(u.user_agent)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {tab === "runs" && data && (
        <table className="w-full text-xs">
          <thead className="text-left text-[10px] uppercase tracking-wider text-neutral-500">
            <tr><th className="py-1">run</th><th>created</th><th>client</th><th>model → hw</th><th>status</th><th>gen</th><th>hv</th><th>candidates</th><th>viewers</th><th>error</th></tr>
          </thead>
          <tbody>
            {((data.runs as Json[]) ?? []).map((r) => {
              const cfg = r.config as Json;
              const lg = (r.last_gen as Json) ?? {};
              return (
                <tr key={String(r.run_id)} className="border-t border-neutral-900">
                  <td className="py-1 pr-2"><Link className="underline" href={`/runs/${String(r.run_id)}`}>{String(r.run_id)}</Link></td>
                  <td className="pr-2">{ago(Number(r.created_at))}</td>
                  <td className="pr-2 text-neutral-500">{String(r.client_id)}</td>
                  <td className="pr-2">{String(cfg.model)} → {String(cfg.hardware)}</td>
                  <td className="pr-2">{String(r.status)}</td>
                  <td>{String(lg.gen ?? 0)} / {String(cfg.generations)}</td>
                  <td>{Number(lg.hv ?? 0).toFixed(4)}</td>
                  <td>{String(r.candidates)}</td><td>{String(r.subscribers)}</td>
                  <td className="text-neutral-400">{r.error ? String(r.error) : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {tab === "stats" && data && (
        <pre className="border border-neutral-900 p-4 text-xs text-neutral-300">{JSON.stringify(data, null, 2)}</pre>
      )}
    </main>
  );
}
