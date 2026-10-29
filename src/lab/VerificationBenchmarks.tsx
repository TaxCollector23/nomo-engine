import verification from "../generated/verification.json";

/** Shape of src/generated/verification.json — regenerate with `npm run stats`. */
interface VerificationSnapshot {
  generatedAt: string;
  verify: { checks: number; worstRelativeDiff: string };
  pythonTests: { passed: number; seconds: number | null };
  simulationGolden: { checks: number };
  bench: { domain: string; plans: number; ms: number; explainMs: number }[];
}

const stats = verification as VerificationSnapshot;

/** Descriptions of the four fixed `npm run bench` configurations, in scripts/bench.mjs output order. */
const BENCH_WORKLOADS = [
  "llm_training · default",
  "llm_training · 64–4,096 devices",
  "llm_inference · default",
  "arch_codesign · $5M budget",
];

/** Live numbers from the repository's own verification commands (see `npm run stats`). */
export default function VerificationBenchmarks() {
  return (
    <section className="lab-card lab-benchmarks" aria-labelledby="verification-benchmarks-title">
      <div className="lab-card-bar">
        <div>
          <p className="section-kicker">Measured checks</p>
          <h3 id="verification-benchmarks-title">Verification &amp; benchmarks</h3>
        </div>
        <span className="lab-muted">snapshot generated {stats.generatedAt}</span>
      </div>
      <div className="lab-stat-row">
        <div className="lab-stat"><b>{stats.pythonTests.passed.toLocaleString("en-US")}</b><span>Python planner tests passed</span></div>
        <div className="lab-stat"><b>{stats.verify.checks.toLocaleString("en-US")}</b><span>TypeScript parity checks passed</span></div>
        <div className="lab-stat"><b>{stats.simulationGolden.checks.toLocaleString("en-US")}</b><span>simulation golden checks passed</span></div>
      </div>
      <div className="lab-table-scroll">
        <table className="lab-table lab-benchmark-table">
          <caption>Planner search benchmark from <code>npm run bench</code></caption>
          <thead><tr><th>Workload</th><th>Plans</th><th>Search</th><th>Explain</th></tr></thead>
          <tbody>
            {stats.bench.map((row, index) => (
              <tr key={`${row.domain}:${index}`}>
                <td>{BENCH_WORKLOADS[index] ?? row.domain}</td>
                <td>{row.plans.toLocaleString("en-US")}</td>
                <td>{row.ms.toLocaleString("en-US")} ms</td>
                <td>{row.explainMs.toLocaleString("en-US")} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lab-note">
        Search timings are a reproducible local snapshot of this build and machine, not a hardware, customer, or production-latency guarantee.
        The test counts are from the same verification pass; rerun the repository commands for a fresh measurement.
      </p>
    </section>
  );
}
