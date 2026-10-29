/// <reference lib="webworker" />
/** Web-worker surface for the capacity clinic: keeps the 13-simulation sweep
 * off the main thread. Same simulator, same seed discipline — the worker is
 * an execution detail, not a different model. */
import { buildResult, type SimJob } from "../lab/clinic-shared";

type Incoming = { requestId: string } & { kind: "run"; job: SimJob; rates: number[]; load: number };

self.onmessage = (event: MessageEvent<Incoming>) => {
  const { requestId, kind, job, rates, load } = event.data;
  if (kind !== "run") return;
  const result = buildResult(job, rates, load);
  (self as unknown as Worker).postMessage({ requestId, result });
};
