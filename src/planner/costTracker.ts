export interface CostRow {
  providerOrLab: string; item: string; kind: "training" | "provider"; inputUsdPerMTok: number | null; outputUsdPerMTok: number | null;
  publishedLowUsd: number | null; publishedHighUsd: number | null; unit: string; sourceTitle: string; sourceUrl: string; asOf: string; notes: string;
  openModel: boolean; checkedOn: string | null; expiresOn: string | null;
}

export type CostFreshnessStatus = "fresh" | "stale" | "undated" | "expired" | "invalid";

export interface CostFreshness {
  status: CostFreshnessStatus;
  ageDays: number | null;
  expiresInDays: number | null;
  reason: string;
}

export const PUBLIC_COST_ROWS: CostRow[] = [
  { providerOrLab: "Meta", item: "Llama 3 8B pretraining", kind: "training", inputUsdPerMTok: null, outputUsdPerMTok: null, publishedLowUsd: 1_300_000, publishedHighUsd: 1_300_000, unit: "H100 GPU hours", sourceTitle: "Llama 3 model card", sourceUrl: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", asOf: "published model card", notes: "Cumulative pretraining compute, not a dollar invoice.", openModel: true, checkedOn: null, expiresOn: null },
  { providerOrLab: "Meta", item: "Llama 3 70B pretraining", kind: "training", inputUsdPerMTok: null, outputUsdPerMTok: null, publishedLowUsd: 6_400_000, publishedHighUsd: 6_400_000, unit: "H100 GPU hours", sourceTitle: "Llama 3 model card", sourceUrl: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", asOf: "published model card", notes: "Cumulative pretraining compute, not a dollar invoice.", openModel: true, checkedOn: null, expiresOn: null },
  { providerOrLab: "OpenAI", item: "GPT-4o", kind: "provider", inputUsdPerMTok: 2.5, outputUsdPerMTok: 10, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "OpenAI API pricing", sourceUrl: "https://openai.com/api/pricing/", asOf: "published list price as of 2024-08-02", notes: "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-08-02", expiresOn: null },
  { providerOrLab: "OpenAI", item: "GPT-4o mini", kind: "provider", inputUsdPerMTok: 0.15, outputUsdPerMTok: 0.6, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "OpenAI API pricing", sourceUrl: "https://openai.com/api/pricing/", asOf: "published list price as of 2024-07-18", notes: "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-07-18", expiresOn: null },
  { providerOrLab: "Anthropic", item: "Claude 3.5 Sonnet", kind: "provider", inputUsdPerMTok: 3, outputUsdPerMTok: 15, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Anthropic pricing", sourceUrl: "https://www.anthropic.com/pricing", asOf: "published list price as of 2024-06-20", notes: "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-06-20", expiresOn: null },
  { providerOrLab: "Anthropic", item: "Claude 3 Haiku", kind: "provider", inputUsdPerMTok: 0.25, outputUsdPerMTok: 1.25, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Anthropic pricing", sourceUrl: "https://www.anthropic.com/pricing", asOf: "published list price as of 2024-03-13", notes: "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-03-13", expiresOn: null },
  { providerOrLab: "Google", item: "Gemini 1.5 Flash (<=128k context)", kind: "provider", inputUsdPerMTok: 0.075, outputUsdPerMTok: 0.3, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Google AI pricing", sourceUrl: "https://ai.google.dev/pricing", asOf: "published list price as of 2024-05-15", notes: "Historic published list price for the <=128k-context tier captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-05-15", expiresOn: null },
  { providerOrLab: "Google", item: "Gemini 1.5 Pro (<=128k context)", kind: "provider", inputUsdPerMTok: 1.25, outputUsdPerMTok: 5, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Google AI pricing", sourceUrl: "https://ai.google.dev/pricing", asOf: "published list price as of 2024-05-15", notes: "Historic published list price for the <=128k-context tier captured from the official pricing page. Provider prices have changed since; the freshness engine will mark this row stale. Verify current pricing before procurement.", openModel: false, checkedOn: "2024-05-15", expiresOn: null },
];

const openModelTrainingRows = PUBLIC_COST_ROWS.filter((row) => row.openModel && row.kind === "training").length;
const openModelServingRows = PUBLIC_COST_ROWS.filter((row) => row.openModel && row.kind === "provider").length;
const missingCostCoverage = [
  ...(openModelTrainingRows === 0 ? ["open-model training compute"] : []),
  ...(openModelServingRows === 0 ? ["open-model serving/token prices"] : []),
];
export const PUBLIC_COST_COVERAGE = {
  status: missingCostCoverage.length ? "partial" as const : "complete" as const,
  openModelTrainingRows,
  openModelServingRows,
  providerRows: PUBLIC_COST_ROWS.filter((row) => row.kind === "provider").length,
  missing: missingCostCoverage,
};

function isoDay(value: string): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value ? null : parsed;
}

export function costSourceFreshness(row: CostRow, asOf = new Date(), maxAgeDays = 90): CostFreshness {
  if (Number.isNaN(asOf.getTime())) return { status: "invalid", ageDays: null, expiresInDays: null, reason: "reference date is invalid" };
  const reference = new Date(Date.UTC(asOf.getUTCFullYear(), asOf.getUTCMonth(), asOf.getUTCDate()));
  const checked = row.checkedOn ? isoDay(row.checkedOn) : null;
  const expires = row.expiresOn ? isoDay(row.expiresOn) : null;
  if ((row.checkedOn && !checked) || (row.expiresOn && !expires) || maxAgeDays < 0) {
    return { status: "invalid", ageDays: null, expiresInDays: null, reason: "source date metadata is not a valid ISO date" };
  }
  const expiresInDays = expires ? Math.round((expires.getTime() - reference.getTime()) / 86400000) : null;
  if (expiresInDays !== null && expiresInDays < 0) {
    return { status: "expired", ageDays: checked ? Math.round((reference.getTime() - checked.getTime()) / 86400000) : null, expiresInDays, reason: `source metadata expired on ${row.expiresOn}` };
  }
  if (!checked) {
    return { status: "undated", ageDays: null, expiresInDays, reason: "no checked-on date is recorded; re-check before procurement or calibration" };
  }
  const ageDays = Math.round((reference.getTime() - checked.getTime()) / 86400000);
  if (ageDays < 0) return { status: "invalid", ageDays, expiresInDays, reason: "checked-on date is after the reference date" };
  if (ageDays > maxAgeDays) return { status: "stale", ageDays, expiresInDays, reason: `checked ${ageDays} days ago; freshness window is ${maxAgeDays} days` };
  const expiryNote = expiresInDays !== null && expiresInDays <= 30 ? `; expires in ${expiresInDays} days` : "";
  return { status: "fresh", ageDays, expiresInDays, reason: `checked ${ageDays} days ago within the ${maxAgeDays}-day freshness window${expiryNote}` };
}

export function physicalCostPerMTok(gpuHourlyCostUsd: number, measuredTokensPerSecond: number): number {
  if (!(gpuHourlyCostUsd > 0) || !(measuredTokensPerSecond > 0)) throw new Error("GPU hourly cost and measured throughput must be positive");
  return gpuHourlyCostUsd / 3600 / measuredTokensPerSecond * 1_000_000;
}

export function compareProviderToPhysical(row: CostRow, gpuHourlyCostUsd: number, measuredTokensPerSecond: number) {
  if (row.kind !== "provider" || row.outputUsdPerMTok === null) throw new Error("physical comparison requires a provider output-token row");
  const physical = physicalCostPerMTok(gpuHourlyCostUsd, measuredTokensPerSecond);
  return { item: row.item, providerOutputUsdPerMTok: row.outputUsdPerMTok, physicalOutputUsdPerMTok: physical, ratio: physical / row.outputUsdPerMTok };
}

// ================================================================== user-supplied provider price CSV
// Mirrors nomo_planner/cost_tracker.py parse_price_csv. Rows are parsed strictly:
// the exact column set is required, every field is validated, and errors name the row.

export const PRICE_CSV_COLUMNS = ["provider", "model", "input_usd_per_mtok", "output_usd_per_mtok", "as_of", "source_url"] as const;

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!;
    const next = text[index + 1];
    if (quoted) {
      if (char === '"' && next === '"') { cell += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
      continue;
    }
    if (char === '"' && cell.length === 0) { quoted = true; continue; }
    if (char === ",") { row.push(cell.trim()); cell = ""; continue; }
    if (char === "\n" || char === "\r") {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell.trim()); cell = "";
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      continue;
    }
    cell += char;
  }
  if (quoted) throw new Error("provider price CSV has an unterminated quoted field");
  row.push(cell.trim());
  if (row.some((value) => value !== "")) rows.push(row);
  return rows;
}

export function parsePriceCsv(text: string): CostRow[] {
  const rows = parseCsvRows(text);
  if (!rows.length) throw new Error("provider price CSV is empty");
  const headers = rows[0]!.map((value) => value.trim().toLowerCase());
  const missing = PRICE_CSV_COLUMNS.filter((name) => !headers.includes(name));
  if (missing.length) throw new Error(`provider price CSV is missing columns: ${missing.join(", ")}`);
  if (new Set(headers).size !== headers.length) throw new Error("provider price CSV has duplicate columns");
  const extra = headers.filter((name) => name !== "" && !(PRICE_CSV_COLUMNS as readonly string[]).includes(name));
  if (extra.length) throw new Error(`provider price CSV has unsupported columns: ${extra.join(", ")}; expected only ${PRICE_CSV_COLUMNS.join(", ")}`);
  const seen = new Set<string>();
  const out: CostRow[] = [];
  rows.slice(1).forEach((values, index) => {
    const rowNo = index + 2;
    const value = (name: string) => (values[headers.indexOf(name)] ?? "").trim();
    const provider = value("provider");
    const model = value("model");
    if (!provider || !model) throw new Error(`provider price CSV row ${rowNo}: provider and model must be non-empty`);
    const inputText = value("input_usd_per_mtok");
    const outputText = value("output_usd_per_mtok");
    const input = Number(inputText);
    const output = Number(outputText);
    if (inputText === "" || !Number.isFinite(input) || input < 0) throw new Error(`provider price CSV row ${rowNo}: input_usd_per_mtok must be a non-negative number`);
    if (outputText === "" || !Number.isFinite(output) || output < 0) throw new Error(`provider price CSV row ${rowNo}: output_usd_per_mtok must be a non-negative number`);
    const asOf = value("as_of");
    if (!isoDay(asOf)) throw new Error(`provider price CSV row ${rowNo}: as_of must be an ISO date like 2024-08-02`);
    const sourceUrl = value("source_url");
    if (!sourceUrl.startsWith("https://")) throw new Error(`provider price CSV row ${rowNo}: source_url must start with https://`);
    const key = `${provider.toLowerCase()}/${model.toLowerCase()}`;
    if (seen.has(key)) throw new Error(`provider price CSV row ${rowNo}: duplicate provider/model pair ${provider}/${model}`);
    seen.add(key);
    out.push({
      providerOrLab: provider, item: model, kind: "provider", inputUsdPerMTok: input, outputUsdPerMTok: output,
      publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens",
      sourceTitle: `${provider} pricing (user-supplied)`, sourceUrl,
      asOf: `user-supplied list price as of ${asOf}`,
      notes: "user-supplied prices; not verified by Nomo; re-check the source before procurement",
      openModel: false, checkedOn: asOf, expiresOn: null,
    });
  });
  if (!out.length) throw new Error("provider price CSV has no data rows");
  return out;
}
