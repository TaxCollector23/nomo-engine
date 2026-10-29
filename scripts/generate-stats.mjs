// Regenerates src/generated/verification.json from the repository's own commands, so the
// Lab UI renders live numbers instead of stale hardcoded ones.
// Run: npm run stats
// Steps (sequential, so bench timings are not skewed by concurrent load):
//   1. node scripts/verify-engine.mjs   with NOMO_STATS_JSON pointed at a temp file
//   2. node scripts/bench.mjs           per-domain search timings
//   3. node scripts/simulation-golden.mjs
//   4. python3 -m pytest nomo-planner/tests -q
// A failing step records { error: "..." } for that section instead of throwing.
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const outPath = join(root, "src", "generated", "verification.json");

function run(command, args, env = undefined, cwd = root) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { cwd, env: env ?? process.env, shell: false });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => resolve({ code: -1, stdout, stderr: `${stderr}${error.message}\n` }));
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

const errorFor = (label, result) => `${label} exited ${result.code}: ${(result.stderr || result.stdout).trim().split("\n").filter(Boolean).slice(-3).join(" | ").slice(0, 500)}`;

const stats = { generatedAt: new Date().toISOString() };

// 1. TypeScript <-> Python parity checks (verify-engine writes its own summary when NOMO_STATS_JSON is set).
{
  const statsFile = join(tmpdir(), `nomo-stats-verify-${process.pid}.json`);
  const result = await run(process.execPath, ["scripts/verify-engine.mjs"], { ...process.env, NOMO_STATS_JSON: statsFile });
  const last = result.stdout.trim().split("\n").filter(Boolean).pop() ?? "";
  if (result.code === 0 && /ALL MATCH: \d+ checks/.test(last)) {
    try {
      stats.verify = JSON.parse(readFileSync(statsFile, "utf8")).verify;
    } catch (error) {
      stats.verify = { error: `could not read verify stats file: ${error.message}` };
    }
  } else {
    stats.verify = { error: errorFor("scripts/verify-engine.mjs", result) };
  }
  rmSync(statsFile, { force: true });
}

// 2. Planner search benchmark. Output lines look like:
//    llm_training      7680 plans    96 ms search, 13 ms explain
{
  const result = await run(process.execPath, ["scripts/bench.mjs"]);
  const bench = [];
  for (const line of result.stdout.split("\n")) {
    const match = line.match(/^(\S+)\s+(\d+)\s+plans\s+(\d+)\s+ms search,\s*(\d+)\s+ms explain/);
    if (match) bench.push({ domain: match[1], plans: Number(match[2]), ms: Number(match[3]), explainMs: Number(match[4]) });
  }
  stats.bench = result.code === 0 && bench.length > 0 ? bench : { error: result.code === 0 ? "no parsable benchmark lines" : errorFor("scripts/bench.mjs", result) };
}

// 3. Browser simulation golden checks. Summary line: "ALL MATCH: 123 checks, worst relative difference ...".
{
  const result = await run(process.execPath, ["scripts/simulation-golden.mjs"]);
  const match = result.stdout.match(/ALL MATCH: (\d+) checks/);
  stats.simulationGolden = result.code === 0 && match ? { checks: Number(match[1]) } : { error: errorFor("scripts/simulation-golden.mjs", result) };
}

// 4. Python reference test suite. Run from nomo-planner/ because the suite resolves its data files
//    relative to the package root ("python3 -m pytest nomo-planner/tests -q" from the repo root
//    fails one CWD-dependent test). Summary line: "138 passed in 22.05s".
{
  const result = await run("python3", ["-m", "pytest", "tests", "-q"], undefined, join(root, "nomo-planner"));
  const match = result.stdout.match(/(\d+) passed/);
  const seconds = result.stdout.match(/in ([\d.]+)s/);
  stats.pythonTests = result.code === 0 && match
    ? { passed: Number(match[1]), seconds: seconds ? Number(seconds[1]) : null }
    : { error: errorFor("pytest nomo-planner/tests", result) };
}

mkdirSync(join(root, "src", "generated"), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(stats, null, 2)}\n`);

const failed = Object.entries(stats).filter(([, value]) => value && typeof value === "object" && "error" in value);
if (failed.length) {
  console.error(`stats written to ${outPath} with failures:`);
  for (const [section, value] of failed) console.error(`  ${section}: ${value.error}`);
} else {
  console.log(`stats written to ${outPath}`);
}
