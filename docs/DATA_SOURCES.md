# Data Sources

LLMDEX is an independent educational analytics project. LLMDEX is not affiliated with, endorsed by, or sponsored by Artificial Analysis, LLMStats, or any AI model provider.

## Primary Source Matrix

| View or metric | Primary source | LLMDEX processing |
| --- | --- | --- |
| General performance | Artificial Analysis | Source-native order preserved |
| Pricing and API speed | Artificial Analysis | Typed normalization and presentation |
| Capability rankings | LLMStats | Source-native publication |
| General consensus | Artificial Analysis + LLMStats | Family matching and percentile consensus |
| Value and efficiency | Artificial Analysis metrics | LLMDEX-derived rankings |
| Model identity | Both sources | LLMDEX family and variant resolution |

## Source Overview & Methodological Notes

### Artificial Analysis

- **Role:** Default General ranking, Intelligence Index, pricing, generation throughput, latency, and expanded benchmark observations.
- **Extraction:** Public model leaderboard and browser-expanded tables. All visible heading/value pairs are retained in `source_details`.
- Methodology links and the complete methodology inventory are configured in `data/methodology/source_config.json`.

### LLMStats

- **Role:** Source-native capability ranks/scores and the secondary General consensus signal.
- **Extraction:** Public server-rendered General table plus public server-rendered category top-model payloads.
- Extraction adheres to public server contracts, displaying attribution and linking directly back to source pages.

### Multi-Source Characteristics & Limitations

- **Differing Update Times:** Upstream sources update their datasets independently. Timestamp differences between Artificial Analysis and LLMStats collections are tracked in metadata.
- **Differing Model Coverage:** A model present in one source may be unlisted in another. Missing coverage is preserved as unavailable and does not produce a penalty.
- **Non-Interchangeable Raw Scores:** Source-native scores reflect different evaluation suites and scoring scales. Raw scores from different sources are never directly averaged or treated as interchangeable raw units.
- **Independence:** LLMDEX processes third-party observations independently and presents source attribution on every view.

## Failure Policy

A zero or invalid live result does not overwrite a healthy cleaned snapshot. The quality report records degraded status, age, row count, warnings, schema changes, and last successful update date.
