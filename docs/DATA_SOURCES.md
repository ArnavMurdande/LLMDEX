# Data Sources

## Artificial Analysis

Role: default General ranking, Intelligence, official capability indices,
prices, throughput, latency, and expanded benchmark observations.

Extraction: public model leaderboard, browser-expanded table. All visible
heading/value pairs are retained in `source_details`.

Methodology links and the complete methodology-tile inventory are in
`data/methodology/source_config.json`.

## LLMStats

Role: source-native capability ranks/scores and the second General consensus
signal.

Extraction: public server-rendered General table plus public server-rendered
category top-model payloads. The scraper does not call paths disallowed by
robots, automate verification, or expand client-only populations.

The public Terms allow reuse with attribution. LLMDEX displays attribution and
links back to source pages. A missing supplemental top-model payload is a
warning if the visible category column remains available.

## Failure policy

A zero/invalid live result does not overwrite a healthy cleaned snapshot. The
quality report records degraded status, age, row count, warnings, schema
changes, and last successful update.
