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
  { providerOrLab: "OpenAI", item: "gpt-5.3-codex", kind: "provider", inputUsdPerMTok: 3.5, outputUsdPerMTok: 28, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "OpenAI API pricing", sourceUrl: "https://developers.openai.com/api/docs/pricing", asOf: "pricing page accessed 2026-10-04", notes: "Standard fast-mode row; verify before procurement.", openModel: false, checkedOn: "2026-10-04", expiresOn: null },
  { providerOrLab: "Anthropic", item: "Claude Sonnet 4", kind: "provider", inputUsdPerMTok: 3, outputUsdPerMTok: 15, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Anthropic model pricing", sourceUrl: "https://docs.anthropic.com/en/docs/about-claude/pricing", asOf: "pricing page accessed 2026-10-04", notes: "Standard global row; verify before procurement.", openModel: false, checkedOn: "2026-10-04", expiresOn: null },
  { providerOrLab: "Google", item: "Gemini 3.8 Flash", kind: "provider", inputUsdPerMTok: 0.75, outputUsdPerMTok: 3.75, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Gemini latest-model pricing", sourceUrl: "https://ai.google.dev/gemini-api/docs/latest-model", asOf: "introductory pricing through 2026-12-31", notes: "Official introductory price; the page states a later standard price.", openModel: false, checkedOn: null, expiresOn: "2026-12-31" },
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
