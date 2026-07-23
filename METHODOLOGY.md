# LLMDEX Methodology

## Source ownership

Artificial Analysis powers the default General leaderboard, prices, throughput,
latency, Value, and Efficiency. LLMStats powers the capability leaderboards.
LLMDEX does not claim to have run either source's benchmarks.

The live URLs and extraction policies are versioned in
`data/methodology/source_config.json`. Benchmark names, versions, units, and
provenance are versioned in `data/methodology/benchmark_registry.json`.

## General order

General opens on Artificial Analysis Performance. The upstream Intelligence
order is unchanged. LLMDEX Score is a sortable secondary column and never
silently replaces the default source order.

Value retains the existing intent:

- 50% performance
- 30% cost efficiency
- 20% speed

Available weights are redistributed when a component is missing. LLMStats does
not influence Value or Efficiency.

The displayed blended token price is `60% input price + 40% output price`.
When only one price is available, that published price is used without
inventing the missing component.

## Family identity

Family and variant are separate entities. Parsing retains reasoning effort,
thinking/adaptive labels, fallback state, dates, context labels, parameter
counts, and the original source name.

Automatic matching order:

1. Exact official source model ID.
2. Exact source model URL.
3. Manually approved alias.
4. Exact provider + normalized family + version.
5. Exact known family mapping.

Fuzzy similarity can only create an audit candidate. It cannot publish a merge.
Statuses are `matched_exact`, `matched_family`, `matched_manual`,
`source_missing`, `identity_unresolved`, `ambiguous`, and `rejected`.

## Representative selection

Consensus is family-level. One Artificial Analysis variant represents a family:

1. Best AA Performance rank.
2. Highest AA Intelligence.
3. Stable alphabetical source name.

Manual representative overrides are supported for exceptional deployments.
Only the representative AA row receives the numeric family score. Other
variants show `Family score available`.

## LLMDEX General Consensus v1

The universe is the confidently matched intersection of families that have an
AA representative and an LLMStats General score.

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

## Coding consensus

Coding consensus is available only where both directly comparable fields exist:

- Official Artificial Analysis Coding Index (current methodology: equal-weight
  Terminal-Bench v2.1 and SciCode).
- LLMStats Coding Index.

The older broad AA mean remains `aa_coding_proxy_legacy`; it is not presented as
the official Coding Index.

Other capability views remain LLMStats source-native until a directly
comparable second source is available.

## Missing sources

- AA only: AA values and rank remain unchanged; LLMDEX Score is null.
- LLMStats only: retained in capability datasets; AA match is null.
- Identity review: both may have a candidate, but no score is published.
- Every AA configuration in an approved family displays the same family-level
  LLMDEX score, agreement, and consensus status. Source-native configuration
  metrics remain distinct.

Null is never converted to zero.

## Availability and badges

Availability is conservative: `open_source`, `open_weights`,
`research_license`, `proprietary`, or `unknown`. A visible non-proprietary
license is not automatically labeled open source.

`SOTA` is the highest valid proprietary consensus family. `OPEN SOTA` is the
highest valid open-weight/open-source consensus family. A single-source AA
leader can receive `AA LEADER`, never LLMDEX SOTA.

## History

Family, model, source, and score history are appended using date + entity +
methodology version keys. A methodology change starts a new version; it does not
rewrite old scores.
