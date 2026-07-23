# Repository Audit

## Architecture found

The project is a Python data pipeline with a framework-free static dashboard.
`api_server.py` combines a `ThreadingHTTPServer`, the server-side Gemini Advisor,
and static/data serving. GitHub Actions updates data and deploys GitHub Pages.

The original path was moved from `LLM Benchmark Intelligence Dashboard` to
`LLMDEX`; a local junction restores the saved workspace path without duplicating
the checkout.

## Original data flow

```text
Selenium AA expanded table
  → ScrapedRow
  → validator
  → generic merger
  → performance/value/efficiency scoring
  → data/index/latest.*
  → static website
```

Sentiment and family history were post-processing paths. Publication validation
expected exactly one benchmark source.

## Main risks found

- The legacy identity normalizer stripped meaningful parenthetical metadata.
- Substring and difflib matches could auto-merge source rows.
- The merger had a broad numeric aggregation list, unsafe for a second source.
- `coding_score` was a custom multi-benchmark proxy, not AA's official Coding
  Index.
- IDs were largely null and no current identity registry existed.
- Source health lacked schema-change and last-success fields.
- The frontend had no capability IA, consensus statuses, data-quality view, or
  dual-source details.
- The README and several comments contained encoding artifacts.
- CI covered only 13 regression tests and the previous scheduled error referred
  to deprecated Node 20 action releases.

## Incremental migration

The existing General, Value, Efficiency, Advisor, comparison, sentiment, family
explorer, static deployment, and routes are preserved. New source-native,
identity, consensus, quality, capability, and history contracts are layered
around them.
