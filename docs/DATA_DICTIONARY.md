# Data Dictionary

## Core identity

| Field | Meaning |
|---|---|
| `family_id` | Stable provider/family identifier. |
| `variant_id` | Stable AA variant identifier. |
| `deployment_profile_id` | Deployment-profile identifier. |
| `source_model_id` | Upstream source ID when published. |
| `source_name` | Exact upstream display name. |
| `reasoning_effort` | Parsed effort such as max/xhigh/high. |
| `fallback_enabled` | Whether the source name explicitly identifies fallback. |

## General and consensus

| Field | Meaning |
|---|---|
| `intelligence_score` | Source-native AA Intelligence. |
| `llmstats_general_score` | Source-native LLMStats General score. |
| `aa_percentile` | AA matched-universe percentile. |
| `llmstats_percentile` | LLMStats matched-universe percentile. |
| `llmdex_score` | 50/50 percentile consensus; null without both approved sources. |
| `llmdex_rank` | Tie-aware family consensus rank. |
| `agreement` | 100 minus absolute source-percentile difference. |
| `score_status` | Machine-readable publication status. |
| `score_version` | Methodology version stored with the observation. |

## Coding

| Field | Meaning |
|---|---|
| `aa_coding_proxy_legacy` | Historical broad AA coding proxy. |
| `aa_official_coding_index` | Current official AA Coding Index field. |
| `llmstats_coding_score` | LLMStats Coding Index. |
| `llmdex_coding_score` | Optional matched-universe percentile consensus. |

## Availability

`availability_class` is one of `open_source`, `open_weights`,
`research_license`, `proprietary`, or `unknown`. Evidence fields remain nullable:
`weights_available`, `source_code_available`, `training_data_disclosed`,
`commercial_use_allowed`, `license_name`, and `license_url`.

## Null handling

Empty CSV cells and JSON `null` mean unavailable. Zero means the source
explicitly published or the methodology legitimately calculated zero. Consumers
must not replace null with zero.

## Capability contracts

Each `data/capabilities/{capability}.json` includes metadata,
`benchmark_columns`, and rows with exact LLMStats `source_name`,
`category_rank`, `category_score`, `rank_evidence`, source identity, match
status, and provenance-carrying `benchmark_observations`.
