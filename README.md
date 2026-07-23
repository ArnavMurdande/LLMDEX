# LLMDEX

LLMDEX is a multi-source LLM intelligence and capability analytics platform. It
keeps upstream observations separate, matches models at family level, and adds a
transparent percentile consensus without pretending that unlike source scores
share a raw scale.

## What powers each view

- **General / Performance, Value, Efficiency:** Artificial Analysis.
- **Coding, Math, Reasoning, Writing, Research, Long Context, Tool Calling:** LLMStats source-native rankings.
- **LLMDEX General Consensus v1:** 50% Artificial Analysis family percentile +
  50% LLMStats General family percentile in the confidently matched
  intersection.
- **Sentiment:** experimental, model-specific community reactions. It never
  affects a score or rank.

LLMDEX does not independently benchmark models. Every published observation
retains its source name, URL, update metadata, and provenance.

## Safety rules

- Missing values stay null; they never become zero.
- A family absent from one source remains visible and unscored.
- Fuzzy identity matches are review candidates only.
- Only the selected Artificial Analysis representative variant displays the
  numeric family consensus score.
- Raw Artificial Analysis and LLMStats composite scores are never averaged.
- Failed live scrapes preserve the last-known-good source snapshot.
- SOTA and OPEN SOTA require a valid consensus and approved identity match.

## Architecture

```text
Artificial Analysis expanded table ─┐
                                    ├─ identity registry ─ family representatives
LLMStats public server HTML ────────┘                         │
                                                              ▼
                                            tie-aware source percentiles
                                                              │
                         ┌────────────────────────────────────┼──────────────┐
                         ▼                                    ▼              ▼
                 General index                     Capability views    Quality/audit
                 JSON + CSV                        JSON + CSV           JSON + CSV
                         │                                    │              │
                         └────────────────────────────────────┴──────────────┘
                                                              │
                                                     Static dashboard/API
```

The implementation remains Python plus a static HTML/CSS/JavaScript frontend:

- `scraper/`: source contracts and policy-aware extractors.
- `pipeline/identity.py`: family/variant parsing and deterministic matching.
- `pipeline/consensus.py`: representatives, percentiles, agreement, statuses,
  and badges.
- `pipeline/multisource.py`: stable publication, history, quality, and exports.
- `api_server.py`: static server, Advisor, and stable read APIs.
- `website/`: responsive dashboard.
- `data/`: source-native, identity, index, history, methodology, and quality
  contracts.

See [METHODOLOGY.md](METHODOLOGY.md) and
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Run locally

Windows PowerShell:

```powershell
cd "C:\Users\Arnav\Desktop\LLMDEX"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pipeline.multisource
.\.venv\Scripts\python.exe api_server.py
```

Open <http://localhost:8080>.

Run the full Artificial Analysis + LLMStats + optional sentiment pipeline:

```powershell
.\.venv\Scripts\python.exe pipeline\run_pipeline.py --with-sentiment
.\.venv\Scripts\python.exe pipeline\build_family_history.py
.\.venv\Scripts\python.exe pipeline\validate_publication.py
```

The Artificial Analysis stage requires Chrome. LLMStats extraction uses only
the public server-rendered pages recorded in
`data/methodology/source_config.json`; it does not use disallowed API paths or
bypass verification.

## API

When `api_server.py` is running:

- `GET /api/leaderboards/general`
- `GET /api/leaderboards/general?sort=llmdex`
- `GET /api/leaderboards/capabilities/{capability}`
- `GET /api/models/{family_id}`
- `GET /api/models/{family_id}/history`
- `GET /api/data-quality`
- `GET /api/methodology`
- `POST /api/advisor`
- `GET /api/health`

Published responses include generation time, methodology version, source update
time, and source health.

## Data exports

- `data/index/latest.csv`
- `data/index/latest.json`
- `data/capabilities/latest.csv`
- `data/history/family_snapshots.csv`
- `data/history/score_history.csv`
- `data/identity/match_audit.csv`
- `data/quality/latest.json`

Power BI should consume these processed outputs rather than recreating the score
in DAX. See [docs/POWER_BI.md](docs/POWER_BI.md).

## Automation and deployment

`.github/workflows/update.yml` runs tests, source collection, identity matching,
scoring, history, publication validation, diagnostics, data commits, and GitHub
Pages deployment. Generated files are committed only when content changes.

For manual deployment, run the validation commands above, push the branch to
GitHub, merge to `main`, and trigger **Daily LLM Benchmark Update + Deploy**.

## Environment variables

All are optional unless their feature is enabled:

- `GEMINI_API_KEY` or `GEMINI_ADVISOR_KEY_1..5`
- `GEMINI_SENTIMENT_KEY_1..4`
- `X_BEARER_TOKEN`
- `PORT` (default `8080`)

Keys remain server-side and are never written to public datasets.

## Attribution

General intelligence, pricing, and API performance data are from
[Artificial Analysis](https://artificialanalysis.ai/leaderboards/models).
Capability rankings and benchmark data are from
[LLMStats](https://llm-stats.com/leaderboards/llm-leaderboard).

## License

See [LICENSE](LICENSE).
