# Gate 5 report — public cost tracker

Status: Partial, cited preview shipped.

Built:

- Cited model-card GPU-hour rows for Llama 3 8B and 70B.
- Cited provider-token price rows for OpenAI, Anthropic, and Google.
- A local calculator for physical output cost from user-entered GPU hourly cost and measured tokens per second.
- Source URLs and as-of labels are shown in the UI; rows are hand-entered and are not scraped at runtime.

Sources:

- [Meta Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [Anthropic Claude pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)
- [Google Gemini model/pricing page](https://ai.google.dev/gemini-api/docs/latest-model)

Integrity boundary:

- GPU-hour compute and provider-token prices are different quantities and are not mixed.
- Physical cost requires the user's measured throughput and GPU rate.
- Prices can change; the tracker must be rechecked before procurement.

Open:

- Broader model coverage, historical price ranges, and automated source freshness checks.
