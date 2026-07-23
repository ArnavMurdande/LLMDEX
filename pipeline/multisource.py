"""Multi-source publication, history and data-quality layer for LLMDEX."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pipeline.consensus import GENERAL_SCORE_VERSION, build_consensus
from pipeline.identity import enrich_aa_rows, write_identity_audit
from scraper.contracts import ScrapeResult, ScraperHealthReport
from scraper.scrape_llmstats import scrape_llmstats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CAPABILITIES = (
    "coding",
    "math",
    "reasoning",
    "writing",
    "research",
    "long_context",
    "tool_calling",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(safe, handle, indent=2, ensure_ascii=False, allow_nan=False)
        temp_name = handle.name
    os.replace(temp_name, path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
        temp_name = handle.name
    os.replace(temp_name, path)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _timestamp_slug(value: str) -> str:
    return value.replace(":", "-").replace("+", "_")


def _health_from_current_aa(rows: List[dict], generated_at: str) -> dict:
    report = _load_json(DATA_DIR / "pipeline_reports" / "latest_summary.json", {})
    for health in report.get("scraper_health", []):
        if health.get("source") == "Artificial Analysis":
            merged = dict(health)
            merged.setdefault("warnings", [])
            merged.setdefault("schema_changes", [])
            snapshot_dates = sorted(
                {
                    str(row.get("snapshot_date"))
                    for row in rows
                    if row.get("snapshot_date")
                }
            )
            if snapshot_dates:
                merged["last_successful_update"] = (
                    snapshot_dates[-1] + "T00:00:00+00:00"
                )
            else:
                merged.setdefault("last_successful_update", generated_at)
            return merged
    return {
        "source": "Artificial Analysis",
        "status": "healthy" if rows else "failed",
        "rows_scraped": len(rows),
        "expected_range": [150, 500],
        "duration_seconds": 0,
        "warnings": ["Health synthesized from the current validated publication."],
        "last_successful_update": generated_at if rows else None,
        "schema_changes": [],
        "error_message": None if rows else "No AA rows available.",
        "timestamp": generated_at,
    }


def _last_known_good_llmstats() -> List[dict]:
    cached = _load_json(
        DATA_DIR / "cleaned" / "llmstats" / "latest.json",
        [],
    )
    if isinstance(cached, dict):
        return list(cached.get("rows") or [])
    return list(cached or [])


def _use_live_or_fallback(
    result: ScrapeResult,
) -> tuple[List[dict], dict, bool]:
    health = result.health.to_dict()
    if result.rows and result.health.status in {"healthy", "degraded"}:
        return [row.to_dict() for row in result.rows], health, False
    fallback = _last_known_good_llmstats()
    if fallback:
        health["status"] = "degraded"
        health.setdefault("warnings", []).append(
            "Live extraction failed; serving the last-known-good LLMStats snapshot."
        )
        health["rows_scraped"] = len(fallback)
        health["last_successful_update"] = fallback[0].get("scraped_at")
        return fallback, health, True
    return [], health, False


def _capability_contract(
    capability: str,
    llmstats_rows: List[dict],
    generated_at: str,
    source_health: dict,
) -> dict:
    registry = _load_json(DATA_DIR / "methodology" / "benchmark_registry.json", {})
    benchmarks = registry.get("benchmarks", {})
    relevant_ids = {
        benchmark_id
        for benchmark_id, entry in benchmarks.items()
        if entry.get("capability_family") in {
            capability,
            "software_engineering" if capability == "coding" else capability,
            "terminal_agentic_coding" if capability == "coding" else capability,
            "scientific_reasoning" if capability == "reasoning" else capability,
        }
    }
    schema_key = f"LLMDEX benchmark columns::{capability}"
    published_columns: Dict[str, dict] = {}
    for source in llmstats_rows:
        for column in (source.get("source_details") or {}).get(schema_key, []):
            benchmark_id = column.get("benchmark_id")
            if benchmark_id:
                published_columns.setdefault(benchmark_id, column)
    published_ids = set(published_columns)
    rows = []
    for source in llmstats_rows:
        score = (source.get("category_scores") or {}).get(capability)
        rank = (source.get("category_ranks") or {}).get(capability)
        if score is None and rank is None:
            continue
        observations = {
            benchmark_id: observation
            for benchmark_id, observation in (
                source.get("benchmark_observations") or {}
            ).items()
            if (
                benchmark_id in published_ids
                if published_ids
                else benchmark_id in relevant_ids
                or capability in (observation.get("capabilities") or [])
                or (
                    benchmark_id.startswith("llmstats__")
                    and capability
                    in str(observation.get("canonical_name") or "").casefold()
                )
            )
        }
        rows.append(
            {
                "category_rank": rank,
                "category_score": score,
                "source_name": source.get("source_name"),
                "provider": source.get("provider"),
                "source_model_id": source.get("source_model_id"),
                "source_model_url": source.get("source_model_url"),
                "family_id": source.get("family_id"),
                "matched_aa_family_id": source.get("matched_aa_family_id"),
                "match_status": source.get("match_status"),
                "match_confidence": source.get("match_confidence"),
                "score_status": source.get("score_status"),
                "score_status_label": source.get("score_status_label"),
                "availability_class": source.get("availability_class"),
                "is_sota": source.get("is_sota"),
                "is_open_sota": source.get("is_open_sota"),
                "rank_evidence": (source.get("rank_evidence") or {}).get(capability),
                "benchmark_observations": observations,
                "source_updated_at": source.get("source_updated_at"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["category_rank"] if row["category_rank"] is not None else float("inf"),
            -(row["category_score"] or float("-inf")),
            (row["source_name"] or "").casefold(),
        )
    )
    column_observations: Dict[str, dict] = {}
    for benchmark_id, column in published_columns.items():
        column_observations.setdefault(benchmark_id, column)
    for row in rows:
        for benchmark_id, observation in row["benchmark_observations"].items():
            column_observations.setdefault(benchmark_id, observation)
    benchmark_columns = sorted(
        column_observations,
        key=lambda benchmark_id: (
            (
                column_observations[benchmark_id]
                .get("source_column_indices", {})
                .get(
                    capability,
                    column_observations[benchmark_id].get(
                        "source_column_index",
                        float("inf"),
                    ),
                )
            ),
            str(
                column_observations[benchmark_id].get("canonical_name")
                or benchmark_id
            ).casefold(),
        ),
    )
    return {
        "generated_at": generated_at,
        "methodology_version": "LLMStats source-native current",
        "source": "LLMStats",
        "source_updated_at": source_health.get("last_successful_update"),
        "source_health": source_health,
        "capability": capability,
        "benchmark_columns": [
            {
                "benchmark_id": benchmark_id,
                "canonical_name": column_observations[benchmark_id].get(
                    "canonical_name"
                )
                or benchmarks.get(benchmark_id, {}).get(
                    "canonical_name", benchmark_id
                ),
                "version": column_observations[benchmark_id].get("version")
                or benchmarks.get(benchmark_id, {}).get("version"),
                "source_name": column_observations[benchmark_id].get(
                    "source_name"
                ),
                "source_url": column_observations[benchmark_id].get(
                    "source_url"
                ),
                "source_population": column_observations[benchmark_id].get(
                    "source_population"
                ),
            }
            for benchmark_id in benchmark_columns
        ],
        "rows": rows,
        "attribution": "Capability rankings and benchmark data from LLMStats.",
    }


def _append_idempotent_csv(
    path: Path,
    rows: List[dict],
    key_fields: tuple[str, ...],
) -> None:
    existing: List[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    keys = {
        tuple(str(row.get(field, "")) for field in key_fields) for row in existing
    }
    additions = [
        row
        for row in rows
        if tuple(str(row.get(field, "")) for field in key_fields) not in keys
    ]
    if not additions:
        return
    _write_csv(path, [*existing, *additions])


def _quality_report(
    aa_health: dict,
    llmstats_health: dict,
    families: List[dict],
    matches: List[dict],
    generated_at: str,
) -> dict:
    now = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    warnings: List[str] = []
    source_reports = {}
    for key, health in (
        ("artificial_analysis", aa_health),
        ("llmstats", llmstats_health),
    ):
        last = health.get("last_successful_update") or health.get("timestamp")
        age_hours = None
        if last:
            try:
                parsed = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                age_hours = max(0.0, (now - parsed).total_seconds() / 3600)
            except ValueError:
                warnings.append(f"{health.get('source', key)} timestamp is invalid.")
        if health.get("status") != "healthy":
            warnings.append(
                f"{health.get('source', key)} is {health.get('status', 'unknown')}."
            )
        if age_hours is not None and age_hours > 48:
            warnings.append(
                f"{health.get('source', key)} snapshot is {age_hours:.1f} hours old."
            )
        source_reports[key] = {**health, "snapshot_age_hours": age_hours}
    counts = {
        "matched_families": sum(
            row.get("score_status") == "consensus" for row in families
        ),
        "aa_only_families": sum(
            row.get("score_status") == "aa_only" for row in families
        ),
        "llmstats_only_families": sum(
            row.get("score_status") == "llmstats_only" for row in families
        ),
        "identity_review_count": sum(
            row.get("score_status") == "identity_review" for row in families
        ),
        "ambiguous_match_count": sum(
            row.get("match_status") == "ambiguous" for row in matches
        ),
        "consensus_scored_families": sum(
            row.get("llmdex_score") is not None for row in families
        ),
    }
    return {
        "generated_at": generated_at,
        "methodology_version": GENERAL_SCORE_VERSION,
        "status": "degraded" if warnings else "healthy",
        "sources": source_reports,
        "counts": counts,
        "missing_benchmark_counts": {
            "aa_intelligence": sum(
                row.get("aa_intelligence") is None for row in families
            ),
            "llmstats_general": sum(
                row.get("llmstats_general_score") is None for row in families
            ),
            "aa_official_coding_index": sum(
                row.get("aa_official_coding_index") is None for row in families
            ),
            "llmstats_coding": sum(
                row.get("llmstats_coding_score") is None for row in families
            ),
        },
        "warnings": warnings,
    }


def publish_multisource(
    *,
    aa_index_path: Optional[Path] = None,
    llmstats_result: Optional[ScrapeResult] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate all multi-source contracts without corrupting source snapshots."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    aa_path = aa_index_path or DATA_DIR / "index" / "latest.json"
    aa_rows = _load_json(aa_path, [])
    if not isinstance(aa_rows, list) or not aa_rows:
        raise ValueError("A non-empty validated AA index is required.")

    live_result = llmstats_result or scrape_llmstats()
    llmstats_rows, llmstats_health, used_fallback = _use_live_or_fallback(live_result)
    aa_health = _health_from_current_aa(aa_rows, timestamp)
    overrides = _load_json(DATA_DIR / "identity" / "manual_overrides.json", {})
    enriched_aa, registry = enrich_aa_rows(aa_rows)
    consensus = build_consensus(
        enriched_aa,
        llmstats_rows,
        registry,
        overrides=overrides,
        generated_at=timestamp,
    )

    # Raw snapshots are append-only and only written when the live source
    # returned rows. A failed scrape never replaces the last-known-good file.
    slug = _timestamp_slug(timestamp)
    _write_json(
        DATA_DIR / "raw" / "artificial_analysis" / f"{slug}.json",
        {"captured_at": timestamp, "rows": aa_rows, "health": aa_health},
    )
    if live_result.rows and not used_fallback:
        _write_json(
            DATA_DIR / "raw" / "llmstats" / f"{slug}.json",
            live_result.to_dict(),
        )

    _write_json(
        DATA_DIR / "cleaned" / "artificial_analysis" / "latest.json",
        consensus["aa_rows"],
    )
    _write_csv(
        DATA_DIR / "cleaned" / "artificial_analysis" / "latest.csv",
        consensus["aa_rows"],
    )
    if llmstats_rows and not used_fallback:
        _write_json(
            DATA_DIR / "cleaned" / "llmstats" / "latest.json",
            {
                "generated_at": timestamp,
                "source_health": llmstats_health,
                "rows": consensus["llmstats_rows"],
            },
        )
        _write_csv(
            DATA_DIR / "cleaned" / "llmstats" / "latest.csv",
            consensus["llmstats_rows"],
        )

    # Existing consumers keep receiving a list and the default AA order.
    consensus["aa_rows"].sort(
        key=lambda row: (
            row.get("performance_rank")
            if row.get("performance_rank") is not None
            else float("inf"),
            _csv_value(row.get("canonical_name") or row.get("model_name")),
        )
    )
    _write_json(DATA_DIR / "index" / "latest.json", consensus["aa_rows"])
    _write_csv(DATA_DIR / "index" / "latest.csv", consensus["aa_rows"])
    _write_json(
        DATA_DIR / "families" / "latest.json",
        {
            "generated_at": timestamp,
            "methodology_version": GENERAL_SCORE_VERSION,
            "source_updated_at": {
                "artificial_analysis": aa_health.get("last_successful_update"),
                "llmstats": llmstats_health.get("last_successful_update"),
            },
            "source_health": {
                "artificial_analysis": aa_health,
                "llmstats": llmstats_health,
            },
            "rows": consensus["families"],
        },
    )

    capabilities = {
        capability: _capability_contract(
            capability,
            consensus["llmstats_rows"],
            timestamp,
            llmstats_health,
        )
        for capability in CAPABILITIES
    }
    for capability, contract in capabilities.items():
        _write_json(DATA_DIR / "capabilities" / f"{capability}.json", contract)
    _write_json(
        DATA_DIR / "capabilities" / "latest.json",
        {
            "generated_at": timestamp,
            "methodology_version": "LLMStats source-native current",
            "source_updated_at": llmstats_health.get("last_successful_update"),
            "source_health": llmstats_health,
            "categories": {
                name: {
                    "row_count": len(contract["rows"]),
                    "url": f"data/capabilities/{name}.json",
                    "benchmark_columns": contract["benchmark_columns"],
                }
                for name, contract in capabilities.items()
            },
        },
    )
    capability_csv_rows = []
    for capability, contract in capabilities.items():
        capability_csv_rows.extend(
            {"capability": capability, **row} for row in contract["rows"]
        )
    _write_csv(DATA_DIR / "capabilities" / "latest.csv", capability_csv_rows)

    write_identity_audit(
        DATA_DIR / "identity",
        consensus["registry"],
        consensus["matches"],
    )
    match_dicts = [match.to_dict() for match in consensus["matches"]]
    quality = _quality_report(
        aa_health,
        llmstats_health,
        consensus["families"],
        match_dicts,
        timestamp,
    )
    _write_json(DATA_DIR / "quality" / "latest.json", quality)

    snapshot_date = timestamp[:10]
    family_history = [
        {
            "snapshot_date": snapshot_date,
            "family_id": row.get("family_id"),
            "canonical_family_name": row.get("canonical_family_name"),
            "aa_representative_variant_id": row.get("aa_representative_variant_id"),
            "aa_intelligence": row.get("aa_intelligence"),
            "aa_rank": row.get("aa_rank"),
            "llmstats_general_score": row.get("llmstats_general_score"),
            "llmstats_general_rank": row.get("llmstats_general_rank"),
            "llmdex_score": row.get("llmdex_score"),
            "llmdex_rank": row.get("llmdex_rank"),
            "agreement": row.get("agreement"),
            "score_status": row.get("score_status"),
            "score_version": row.get("score_version"),
            "availability_class": row.get("availability_class"),
            "is_sota": row.get("is_sota"),
            "is_open_sota": row.get("is_open_sota"),
            "input_cost": row.get("input_cost"),
            "output_cost": row.get("output_cost"),
            "tokens_per_second": row.get("tokens_per_second"),
            "latency": row.get("latency"),
            "source_coverage": row.get("source_coverage"),
            "mapping_status": row.get("mapping_status"),
        }
        for row in consensus["families"]
    ]
    _append_idempotent_csv(
        DATA_DIR / "history" / "family_snapshots.csv",
        family_history,
        ("snapshot_date", "family_id", "score_version"),
    )
    with (DATA_DIR / "history" / "family_snapshots.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        _write_json(
            DATA_DIR / "history" / "family_snapshots.json",
            list(csv.DictReader(handle)),
        )
    _append_idempotent_csv(
        DATA_DIR / "history" / "score_history.csv",
        family_history,
        ("snapshot_date", "family_id", "score_version"),
    )
    _append_idempotent_csv(
        DATA_DIR / "history" / "model_snapshots.csv",
        [
            {
                "snapshot_date": snapshot_date,
                "variant_id": row.get("variant_id"),
                "family_id": row.get("family_id"),
                "source_name": row.get("source_name"),
                "aa_intelligence": row.get("intelligence_score"),
                "aa_rank": row.get("performance_rank"),
                "llmdex_score": row.get("llmdex_score"),
                "score_status": row.get("score_status"),
                "score_version": row.get("score_version"),
            }
            for row in consensus["aa_rows"]
        ],
        ("snapshot_date", "variant_id", "score_version"),
    )
    _append_idempotent_csv(
        DATA_DIR / "history" / "source_snapshots.csv",
        [
            {
                "snapshot_date": snapshot_date,
                "source": health.get("source"),
                "status": health.get("status"),
                "rows_scraped": health.get("rows_scraped"),
                "last_successful_update": health.get("last_successful_update"),
                "schema_changes": health.get("schema_changes"),
            }
            for health in (aa_health, llmstats_health)
        ],
        ("snapshot_date", "source"),
    )
    _append_idempotent_csv(
        DATA_DIR / "quality" / "source_health_history.csv",
        [
            {
                "snapshot_date": snapshot_date,
                "generated_at": timestamp,
                "source": health.get("source"),
                "status": health.get("status"),
                "rows_scraped": health.get("rows_scraped"),
                "last_successful_update": health.get("last_successful_update"),
                "warnings": health.get("warnings"),
                "schema_changes": health.get("schema_changes"),
                "error_message": health.get("error_message"),
            }
            for health in (aa_health, llmstats_health)
        ],
        ("snapshot_date", "source"),
    )

    manifest = {
        "generated_at": timestamp,
        "methodology_version": GENERAL_SCORE_VERSION,
        "source_updated_at": {
            "artificial_analysis": aa_health.get("last_successful_update"),
            "llmstats": llmstats_health.get("last_successful_update"),
        },
        "source_health": {
            "artificial_analysis": aa_health,
            "llmstats": llmstats_health,
        },
        "attribution": {
            "general": "General intelligence, pricing and API performance data from Artificial Analysis.",
            "capabilities": "Capability rankings and benchmark data from LLMStats.",
        },
        "contracts": {
            "general": "data/index/latest.json",
            "families": "data/families/latest.json",
            "capabilities": "data/capabilities/latest.json",
            "quality": "data/quality/latest.json",
            "identity_audit": "data/identity/match_audit.csv",
        },
    }
    _write_json(
        DATA_DIR / "methodology" / "publication_manifest.json",
        manifest,
    )
    return {
        "generated_at": timestamp,
        "aa_rows": len(consensus["aa_rows"]),
        "llmstats_rows": len(consensus["llmstats_rows"]),
        "families": len(consensus["families"]),
        "quality": quality,
        "source_health": {
            "artificial_analysis": aa_health,
            "llmstats": llmstats_health,
        },
        "used_llmstats_fallback": used_fallback,
    }


def main() -> int:
    result = publish_multisource()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
