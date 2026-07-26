# Power BI Integration Layer (v2 Wide Schema)

LLMDEX provides a dedicated, Power BI-oriented export layer located under `data/powerbi/v1/` formatted as wide, typed, relational tables.

---

## Published Export Contracts (`powerbi-v2-wide`)

The export layer provides four primary wide datasets alongside manifest and data dictionary files:

| File | Type | Table Grain | Natural Key | Primary Purpose |
| --- | --- | --- | --- | --- |
| `artificial_analysis_benchmarks.csv` | CSV | 1 row per `snapshot_date` + `model_key` | `snapshot_date` + `model_key` | Wide typed table containing performance, pricing, speed, latency, TTFT, response time, context window, and benchmark columns (`aa_` prefix for source-native, `llmdex_` for derived). |
| `llmstats_benchmarks.csv` | CSV | 1 row per `snapshot_date` + `capability` + `source_model_id/source_name` | `snapshot_date` + `capability` + `source_model_id/source_name` | Wide typed table containing category ranks, category scores, and dynamic benchmark metric columns (`benchmark_` prefix) across 8 capability categories. |
| `combined_latest.json` | JSON | Source-separated JSON document | `schema_version` + `generated_from_snapshot` | Wide JSON contract preserving source-native namespaces (`artificial_analysis`, `llmstats`) and consensus (`llmdex`). |
| `model_family_history.csv` | CSV | 1 row per `observation_date` + `family_id` + `record_type` | `observation_date` + `family_id` + `record_type` | Complete model family history with explicit record types (`observed_snapshot`, `release_event`, `catalog_event`) spanning 2022 to present. |
| `manifest.json` | JSON | Metadata manifest | N/A | Hashes, row counts, wide column counts, snapshot dates, coverage limitations, and Git provenance. |
| `data_dictionary.csv` | CSV | Field metadata dictionary | `table_name` + `column_name` | Column data types, descriptions, units, nullable status, computed source populations, and recommended aggregations. |

---

## Data Governance & Design Principles

1. **Source Separation**: Artificial Analysis (`aa_` prefix) and LLMStats (`benchmark_` prefix) observations remain in their respective source schemas. Raw source scores are never directly averaged.
2. **Missing Value Preservation**: Missing metric values remain blank in CSV files and `null` in JSON contracts. They are never converted to zero.
3. **Evidence-Based History**: Model family history clearly distinguishes between actual observed LLMDEX snapshots (`observed_snapshot`) and repository catalogue/release events (`catalog_event`).
4. **Stable Contracts**: Existing website-facing data files (`data/index/latest.csv`, `data/index/latest.json`, `data/history/score_history.csv`, `data/identity/match_audit.csv`) remain unchanged for backward compatibility.

---

## Recommended Power BI Star-Schema Model

```
DimFamily (family_id) ────┬───> FactArtificialAnalysis (family_id)
                          ├───> FactLLMStats (family_id)
                          └───> FactFamilyHistory (family_id)

DimModel (model_key) ─────────> FactArtificialAnalysis (model_key)
DimCapability (capability) ────> FactLLMStats (capability)

DimDate (date) ───────────┬───> FactArtificialAnalysis (snapshot_date)
                          ├───> FactLLMStats (snapshot_date)
                          └───> FactFamilyHistory (observation_date)
```

### Relationships:
- **`DimFamily[family_id]`** $\rightarrow$ **`FactArtificialAnalysis[family_id]`** (1:Many)
- **`DimFamily[family_id]`** $\rightarrow$ **`FactLLMStats[family_id]`** (1:Many)
- **`DimFamily[family_id]`** $\rightarrow$ **`FactFamilyHistory[family_id]`** (1:Many)
- **`DimModel[model_key]`** $\rightarrow$ **`FactArtificialAnalysis[model_key]`** (1:Many)
- **`DimCapability[capability]`** $\rightarrow$ **`FactLLMStats[capability]`** (1:Many)
- **`DimDate[date]`** $\rightarrow$ **`FactArtificialAnalysis[snapshot_date]`** (1:Many)
- **`DimDate[date]`** $\rightarrow$ **`FactLLMStats[snapshot_date]`** (1:Many)
- **`DimDate[date]`** $\rightarrow$ **`FactFamilyHistory[observation_date]`** (1:Many)

---

## Recommended Power Query Import

```powerquery
let
    Source = Csv.Document(
        Web.Contents("https://raw.githubusercontent.com/ArnavMurdande/LLMDEX/main/data/powerbi/v1/artificial_analysis_benchmarks.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    Headers
```

> **Note**: CSV exports use **UTF-8 with BOM** for seamless encoding detection in Power BI Desktop and Microsoft Excel.
