# Troubleshooting

## LLMStats degraded

Read `data/quality/latest.json` and the latest workflow artifact. A missing
supplemental category top list is a warning when the General table still
publishes that category column. A missing/empty General table fails closed and
uses last-known-good cleaned data.

## Artificial Analysis row collapse

The pipeline fails publication when the expanded table returns zero, falls below
the expected range, loses required identifiers, or produces duplicate IDs.
Inspect `data/pipeline_reports/` and the captured raw artifact.

## Pending match

Review `data/identity/unresolved_matches.json`. Add a narrowly scoped approved
alias to `manual_overrides.json` only after verifying provider, family, and
version. Never approve solely from fuzzy similarity.

## Advisor unavailable

Run `python scripts/check_gemini_keys.py`. Set `GEMINI_API_KEY` or advisor pool
keys in the server environment. Cloudflare Pages has no server-side secret runtime,
so the browser uses deterministic local analysis there.

## Frontend data not found

Serve through `api_server.py` or another HTTP server; do not open `index.html`
directly. Confirm the deployment copied `data/capabilities`, `data/families`,
`data/quality`, `data/cleaned/llmstats`, and history JSON.
