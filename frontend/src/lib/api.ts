"use client";

import { apiBase } from "./telemetry/protocol";

const KEY = "nomo.client_id";

/** Anonymous per-browser id so the backend can attribute requests, runs and sockets (users log). */
export function clientId(): string {
  if (typeof window === "undefined") return "ssr";
  try {
    let id = window.localStorage.getItem(KEY);
    if (!id) {
      id = (crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`)
        .replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 36);
      window.localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    return "no-storage";
  }
}

export async function api(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("X-Nomo-Client", clientId());
  return fetch(`${apiBase()}${path}`, { ...init, headers });
}

/**
 * Free hosts (e.g. Render) put idle services to sleep; the first request can take ~30-60 s.
 * Polls /healthz with backoff and reports progress until the backend answers or `timeoutMs` passes.
 */
export async function waitForBackend(onProgress: (elapsedS: number) => void, timeoutMs = 120_000): Promise<boolean> {
  const t0 = Date.now();
  let delay = 1000;
  while (Date.now() - t0 < timeoutMs) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 15_000);
      const r = await api("/healthz", { signal: ctl.signal, cache: "no-store" });
      clearTimeout(timer);
      if (r.ok) return true;
    } catch {
      /* still waking */
    }
    onProgress(Math.round((Date.now() - t0) / 1000));
    await new Promise((res) => setTimeout(res, delay));
    delay = Math.min(delay * 1.5, 5000);
  }
  return false;
}
