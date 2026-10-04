import { create } from "zustand";

import type { ConnectionState } from "./client";
import type { Envelope, EvalItem, GenCompleted, RunStarted, RunStatus } from "./protocol";

export interface HvPoint { gen: number; hv: number; feasible: number }

export interface RunState {
  runId: string | null;
  status: RunStatus;
  connection: ConnectionState;
  connectionDetail?: string;
  run: RunStarted | null;
  /** all candidates ever streamed, keyed by genome key; mutated in place, `version` signals change */
  items: Map<string, EvalItem>;
  version: number;
  front: Set<string>;
  population: Set<string>;
  lastGen: GenCompleted | null;
  hv: HvPoint[];
  recommendedKey: string | null;
  selectedKey: string | null;
  /** true once the user has clicked a candidate; until then the view follows the recommendation */
  pinned: boolean;
  error: string | null;
  lastSeq: number;
  /** trade-off filters: hide designs above an energy/latency limit or below an accuracy floor (null = off) */
  filter: { eMax: number | null; lMax: number | null; accMin: number | null };
  /** layer index opened in the inspector (clicked in the design graph) */
  inspect: number | null;
  drawer: "copilot" | "export" | null;

  reset: (runId: string) => void;
  ingest: (batch: Envelope[]) => void;
  setConnection: (s: ConnectionState, detail?: string) => void;
  select: (key: string | null, byUser?: boolean) => void;
  setFilter: (f: Partial<RunState["filter"]>) => void;
  setInspect: (i: number | null) => void;
  setDrawer: (d: RunState["drawer"]) => void;
}

const empty = (runId: string | null) => ({
  runId,
  status: "pending" as RunStatus,
  connection: "idle" as ConnectionState,
  connectionDetail: undefined,
  run: null,
  items: new Map<string, EvalItem>(),
  version: 0,
  front: new Set<string>(),
  population: new Set<string>(),
  lastGen: null,
  hv: [] as HvPoint[],
  recommendedKey: null,
  selectedKey: null,
  pinned: false,
  error: null,
  lastSeq: 0,
  filter: { eMax: null, lMax: null, accMin: null },
  inspect: null,
  drawer: null as RunState["drawer"],
});

function applyGen(s: RunState, g: GenCompleted): void {
  s.lastGen = g;
  s.front = new Set(g.front);
  s.population = new Set(g.population);
  if (g.recommended) {
    s.items.set(g.recommended.key, { ...s.items.get(g.recommended.key), ...g.recommended });
    s.recommendedKey = g.recommended.key;
  }
  if (!s.hv.length || s.hv[s.hv.length - 1]!.gen < g.gen) {
    s.hv = [...s.hv, { gen: g.gen, hv: g.hv, feasible: g.feasible_fraction }];
  }
}

/**
 * Single store for one run. `ingest` takes a batch of envelopes (one animation frame's worth)
 * and produces exactly one state update, so bursts of eval.batch coalesce into one React commit.
 */
export const useRunStore = create<RunState>((set, get) => ({
  ...empty(null),

  reset: (runId) => set(empty(runId)),

  setConnection: (connection, connectionDetail) => set({ connection, connectionDetail }),

  select: (selectedKey, byUser = true) => set({ selectedKey, pinned: byUser && selectedKey !== null }),

  setFilter: (f) => set((s) => ({ filter: { ...s.filter, ...f }, version: s.version + 1 })),

  setInspect: (inspect) => set({ inspect }),

  setDrawer: (drawer) => set({ drawer }),

  ingest: (batch) => {
    if (!batch.length) return;
    const s: RunState = { ...get(), items: get().items };
    for (const env of batch) {
      s.lastSeq = env.seq;
      switch (env.type) {
        case "snapshot": {
          const d = env.data;
          s.items = new Map(d.items.map((it) => [it.key, it]));
          s.run = d.run;
          s.status = d.status;
          s.hv = [];
          if (d.last_gen) applyGen(s, d.last_gen);
          break;
        }
        case "run.started":
          s.run = env.data;
          s.status = "running";
          break;
        case "eval.batch":
          for (const it of env.data.items) s.items.set(it.key, it);
          break;
        case "gen.completed":
          applyGen(s, env.data);
          break;
        case "run.completed":
          s.status = "completed";
          for (const it of env.data.front) s.items.set(it.key, it);
          s.front = new Set(env.data.front.map((it) => it.key));
          s.recommendedKey = env.data.recommended?.key ?? s.recommendedKey;
          break;
        case "run.failed":
          s.status = "failed";
          s.error = env.data.error;
          break;
      }
    }
    if (!s.pinned && s.recommendedKey) s.selectedKey = s.recommendedKey;
    s.version += 1;
    set(s);
  },
}));
