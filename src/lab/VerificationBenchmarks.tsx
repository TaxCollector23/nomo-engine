const SEARCH_BENCHMARKS = [
  { workload: "llm_training · default", plans: "7,680", search: "63 ms", explain: "1 ms" },
  { workload: "llm_training · 64–4,096 devices", plans: "13,440", search: "150 ms", explain: "1 ms" },
  { workload: "llm_inference · default", plans: "216", search: "2 ms", explain: "0 ms" },
  { workload: "arch_codesign · $5M budget", plans: "1,215", search: "13 ms", explain: "0 ms" },
];

/** Checked-in results from the repository's deterministic benchmark command. */
export default function VerificationBenchmarks() {
  return (
    <section className="lab-card lab-benchmarks" aria-labelledby="verification-benchmarks-title">
      <div className="lab-card-bar">
        <div>
          <p className="section-kicker">Measured checks</p>
          <h3 id="verification-benchmarks-title">Verification &amp; benchmarks</h3>
        </div>
        <span className="lab-muted">local run · 2026-10-06</span>
      </div>
      <div className="lab-stat-row">
        <div className="lab-stat"><b>134</b><span>Python planner tests passed</span></div>
        <div className="lab-stat"><b>14,337</b><span>TypeScript parity checks passed</span></div>
        <div className="lab-stat"><b>123</b><span>simulation golden checks passed</span></div>
      </div>
      <div className="lab-table-scroll">
        <table className="lab-table lab-benchmark-table">
          <caption>Planner search benchmark from <code>npm run bench</code></caption>
          <thead><tr><th>Workload</th><th>Plans</th><th>Search</th><th>Explain</th></tr></thead>
          <tbody>
            {SEARCH_BENCHMARKS.map((row) => (
              <tr key={row.workload}>
                <td>{row.workload}</td><td>{row.plans}</td><td>{row.search}</td><td>{row.explain}</td>
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
