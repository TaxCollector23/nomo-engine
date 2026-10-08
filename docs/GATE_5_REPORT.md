# Gate 5 report — public cost tracker

Status: Partial, cited preview shipped.

Built:

- Cited model-card GPU-hour rows for Llama 3 8B and 70B.
- Cited provider-token price rows for OpenAI, Anthropic, and Google.
- A local calculator for physical output cost from user-entered GPU hourly cost and measured tokens per second.
- Source URLs and as-of labels are shown in the UI; rows are hand-entered and are not scraped at runtime.
- Metadata-only source freshness checks now classify each row as fresh, stale, undated, or expired from checked-in
  ISO dates. The browser never fetches a URL, and undated rows remain undated.
- The tracker now reports its open-model coverage explicitly: two cited Llama 3 training-compute rows and zero
  cited open-model serving/token-price rows. Closed-provider rows are not counted as open-model coverage.

Sources:

- [Meta Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [Anthropic Claude pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)
- [Google Gemini model/pricing page](https://ai.google.dev/gemini-api/docs/latest-model)

Integrity boundary:

- GPU-hour compute and provider-token prices are different quantities and are not mixed.
- Physical cost requires the user's measured throughput and GPU rate.
- Prices can change; the tracker must be rechecked before procurement. A fresh metadata label means only that the
  checked-on date is within the configured window; it does not prove that the linked page was fetched or unchanged.

Open:

- Broader open-model serving/token-price coverage, historical price ranges, and network-backed source verification.
