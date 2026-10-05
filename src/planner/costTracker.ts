export interface CostRow {
  providerOrLab: string; item: string; kind: "training" | "provider"; inputUsdPerMTok: number | null; outputUsdPerMTok: number | null;
  publishedLowUsd: number | null; publishedHighUsd: number | null; unit: string; sourceTitle: string; sourceUrl: string; asOf: string; notes: string;
}

export const PUBLIC_COST_ROWS: CostRow[] = [
  { providerOrLab: "Meta", item: "Llama 3 8B pretraining", kind: "training", inputUsdPerMTok: null, outputUsdPerMTok: null, publishedLowUsd: 1_300_000, publishedHighUsd: 1_300_000, unit: "H100 GPU hours", sourceTitle: "Llama 3 model card", sourceUrl: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", asOf: "published model card", notes: "Cumulative pretraining compute, not a dollar invoice." },
  { providerOrLab: "Meta", item: "Llama 3 70B pretraining", kind: "training", inputUsdPerMTok: null, outputUsdPerMTok: null, publishedLowUsd: 6_400_000, publishedHighUsd: 6_400_000, unit: "H100 GPU hours", sourceTitle: "Llama 3 model card", sourceUrl: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", asOf: "published model card", notes: "Cumulative pretraining compute, not a dollar invoice." },
  { providerOrLab: "OpenAI", item: "gpt-5.3-codex", kind: "provider", inputUsdPerMTok: 3.5, outputUsdPerMTok: 28, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "OpenAI API pricing", sourceUrl: "https://developers.openai.com/api/docs/pricing", asOf: "pricing page accessed 2026-10-04", notes: "Standard fast-mode row; verify before procurement." },
  { providerOrLab: "Anthropic", item: "Claude Sonnet 4", kind: "provider", inputUsdPerMTok: 3, outputUsdPerMTok: 15, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Anthropic model pricing", sourceUrl: "https://docs.anthropic.com/en/docs/about-claude/pricing", asOf: "pricing page accessed 2026-10-04", notes: "Standard global row; verify before procurement." },
  { providerOrLab: "Google", item: "Gemini 3.8 Flash", kind: "provider", inputUsdPerMTok: 0.75, outputUsdPerMTok: 3.75, publishedLowUsd: null, publishedHighUsd: null, unit: "USD / 1M tokens", sourceTitle: "Gemini latest-model pricing", sourceUrl: "https://ai.google.dev/gemini-api/docs/latest-model", asOf: "introductory pricing through 2026-12-31", notes: "Official introductory price; the page states a later standard price." },
];

export function physicalCostPerMTok(gpuHourlyCostUsd: number, measuredTokensPerSecond: number): number {
  if (!(gpuHourlyCostUsd > 0) || !(measuredTokensPerSecond > 0)) throw new Error("GPU hourly cost and measured throughput must be positive");
  return gpuHourlyCostUsd / 3600 / measuredTokensPerSecond * 1_000_000;
}

export function compareProviderToPhysical(row: CostRow, gpuHourlyCostUsd: number, measuredTokensPerSecond: number) {
  if (row.kind !== "provider" || row.outputUsdPerMTok === null) throw new Error("physical comparison requires a provider output-token row");
  const physical = physicalCostPerMTok(gpuHourlyCostUsd, measuredTokensPerSecond);
  return { item: row.item, providerOutputUsdPerMTok: row.outputUsdPerMTok, physicalOutputUsdPerMTok: physical, ratio: physical / row.outputUsdPerMTok };
}
