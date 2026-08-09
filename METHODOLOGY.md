# LLMDEX Methodology

## Overview

LLMDEX is an auditable, source-transparent LLM analytics platform. Every ranking decision is documented, every data point is traceable to its upstream source, and no missing value is ever fabricated.

LLMDEX does not independently execute or rerun upstream benchmark evaluations. A successfully processed dataset snapshot indicates that source data was ingested, validated, and normalized without pipeline failure; it does not imply independent verification of third-party benchmark executions.

## Source Ownership

Artificial Analysis powers the default General leaderboard, listed API prices, generation throughput, latency, Value, and Efficiency views. LLMStats powers the capability-specific leaderboards. LLMDEX preserves observations from both sources separately and links them only at the model-family level.

The live URLs and extraction policies are versioned in `data/methodology/source_config.json`. Benchmark names, versions, units, and provenance are versioned in `data/methodology/benchmark_registry.json`.

## Metric Distinctions

LLMDEX clearly distinguishes between related but distinct performance and operational dimensions:

- **Output Throughput vs. Latency:** Output throughput measures generated output tokens per second. Total response latency includes network transit, queuing, and processing delay. Time to First Token (TTFT) measures initial response latency. Output throughput is distinct from both TTFT and total latency.
- **Context Capacity vs. Long-Context Quality:** Listed context-window size reflects the maximum published token capacity accepted by a model or deployment. It does not prove equivalent retrieval accuracy, attention preservation, or reasoning quality across the full context window.

## General Order

General opens on Artificial Analysis Performance. The upstream Intelligence order is preserved unchanged. The LLMDEX Score is a sortable secondary column and never silently replaces the default source order.

Value retains the existing intent:

- 50% performance
- 30% cost efficiency
- 20% speed

A valid performance score and blended token price are required for Value eligibility. If speed is unavailable, its 20% weight is redistributed proportionally between performance and cost, producing 62.5% performance and 37.5% cost. Missing price is never redistributed: the model remains unranked in Value. LLMStats does not influence Value or Efficiency.

The displayed blended token price is `60% input price + 40% output price`. When only one price is available, that published price is used without inventing the missing component.

Models with a listed zero API price may rank highest on API-price efficiency. However, self-hosting, hardware, electricity, engineering, and operational costs are not included in listed API pricing metrics.

Efficiency eligibility is applied before normalization: a model must have valid pricing and adjusted performance of at least 25. Eligible raw performance-per-dollar ratios receive average ranks that are mapped across the observed eligible rank range from `0–100`. Tied leaders receive 100, tied lowest models receive 0, and a population with one distinct efficiency value receives 100. Ineligible models have no Efficiency score or rank and cannot affect eligible percentiles.

## Family Identity

Family and variant are separate entities. Parsing retains reasoning effort, thinking/adaptive labels, fallback state, dates, context labels, parameter counts, and the original source name.

Automatic matching order:

1. Exact official source model ID.
2. Exact source model URL.
3. Manually approved alias.
4. Exact provider + normalized family + version.
5. Exact known family mapping.

Fuzzy similarity can only create an audit candidate. It cannot publish a merge. Statuses are `matched_exact`, `matched_family`, `matched_manual`, `source_missing`, `identity_unresolved`, `ambiguous`, and `rejected`.

## Representative Selection

Consensus is family-level. One Artificial Analysis variant represents a family:

1. Best AA Performance rank.
2. Highest AA Intelligence.
3. Stable alphabetical source name.

Manual representative overrides are supported for exceptional deployments. Only the representative AA row receives the numeric family score. Other variants show `Family score available`.

## LLMDEX General Consensus v1

The universe is the confidently matched intersection of families that have an AA representative and an LLMStats General score.

For each source, descending scores receive average ranks for ties:

```text
percentile = 100 × (N - average_rank) / (N - 1)
```

For `N = 1`, the percentile is 100.

```text
LLMDEX Score =
  0.50 × AA family percentile
  + 0.50 × LLMStats General percentile
```

Raw composites are never averaged.

Agreement is informational:

```text
Agreement = 100 - |AA percentile - LLMStats percentile|
```

- 90–100: High agreement
- 75–89: Moderate agreement
- below 75: Low agreement

Agreement never changes score, rank, or SOTA status.

## Coding Consensus

Coding consensus is available only where both directly comparable fields exist:

- Official Artificial Analysis Coding Index (current methodology: equal-weight Terminal-Bench v2.1 and SciCode).
- LLMStats Coding Index.

The older broad AA mean remains `aa_coding_proxy_legacy`; it is not presented as the official Coding Index.

Other capability views remain LLMStats source-native until a directly comparable second source is available.

## Missing Sources

- AA only: AA values and rank remain unchanged; LLMDEX Score is null.
- LLMStats only: retained in capability datasets; AA match is null.
- Identity review: both may have a candidate, but no score is published.
- Every AA configuration in an approved family displays the same family-level LLMDEX score, agreement, and consensus status. Source-native configuration metrics remain distinct.

Null is never converted to zero.

## Availability and Badges

Availability is conservative: `open_source`, `open_weights`, `research_license`, `proprietary`, or `unknown`. A visible non-proprietary license is not automatically labeled open source.

`SOTA` is the highest valid proprietary consensus family. `Open-Weights SOTA` is the highest valid eligible open-weights consensus family. A single-source AA leader can receive `AA LEADER`, never LLMDEX SOTA.

## History

Family, model, source, and score history are appended using date + entity + methodology version keys. A methodology change starts a new version; it does not rewrite old scores.
