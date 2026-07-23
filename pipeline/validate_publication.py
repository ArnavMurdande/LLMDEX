"""Fail CI before publishing structurally stale or incomplete LLMDEX data."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.sentiment_pipeline import get_models_for_sentiment


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_publication() -> None:
    index = _read("data/index/latest.json")
    history = _read("data/history/family_growth.json")
    sentiment = _read("data/sentiment/latest.json")
    families_contract = _read("data/families/latest.json")
    quality = _read("data/quality/latest.json")
    manifest = _read("data/methodology/publication_manifest.json")
    match_audit = _read("data/identity/match_audit.json")
    registry = _read("data/identity/model_registry.json")
    families = families_contract.get("rows", [])

    if len(index) < 100:
        raise ValueError(f"Index unexpectedly small: {len(index)} rows")
    ranked = [row for row in index if row.get("performance_rank") is not None]
    if not ranked or min(row["performance_rank"] for row in ranked) != 1:
        raise ValueError("Performance ranking has no rank 1")
    sources = {
        source
        for row in index
        for source in (row.get("sources") or [])
        if source
    }
    if sources != {"Artificial Analysis"}:
        raise ValueError(f"Unexpected benchmark sources: {sorted(sources)}")

    required_statuses = {
        "consensus",
        "aa_only",
        "llmstats_only",
        "identity_review",
    }
    invalid_statuses = {
        row.get("score_status")
        for row in index
        if row.get("score_status") not in required_statuses
    }
    if invalid_statuses:
        raise ValueError(f"Invalid score statuses: {sorted(invalid_statuses)}")
    duplicate_family_ids = len({row.get("family_id") for row in families}) != len(
        families
    )
    if duplicate_family_ids:
        raise ValueError("Duplicate family IDs in family publication")
    if set(registry) != {
        row.get("family_id")
        for row in families
        if row.get("aa_representative_variant_id")
    }:
        raise ValueError("Identity registry does not match published AA families")
    families_by_id = {row.get("family_id"): row for row in families}
    allowed_badges = {"SOTA", "OPEN SOURCE", "OPEN WEIGHTS", "PROPRIETARY"}
    for row in index:
        family = families_by_id.get(row.get("family_id"), {})
        if row.get("llmdex_score") is not None and row.get(
            "llmdex_score"
        ) != family.get("llmdex_score"):
            raise ValueError(
                "An AA variant does not share its published family score: "
                f"{row.get('canonical_name')}"
            )
        unexpected_badges = set(row.get("badges") or []) - allowed_badges
        if unexpected_badges:
            raise ValueError(
                f"Unsupported model tags: {sorted(unexpected_badges)}"
            )
        if row.get("score_status") in {"aa_only", "identity_review"} and row.get(
            "llmdex_score"
        ) is not None:
            raise ValueError(
                f"Single-source/review model has a fabricated score: {row.get('canonical_name')}"
            )
    duplicate_source_ids = [
        source_id
        for source_id in {
            row.get("source_model_id") for row in match_audit if row.get("source_model_id")
        }
        if sum(row.get("source_model_id") == source_id for row in match_audit) > 1
    ]
    if duplicate_source_ids:
        raise ValueError(f"Duplicate LLMStats source IDs: {duplicate_source_ids}")
    for capability in (
        "coding",
        "math",
        "reasoning",
        "writing",
        "research",
        "long_context",
        "tool_calling",
    ):
        contract = _read(f"data/capabilities/{capability}.json")
        if contract.get("source") != "LLMStats":
            raise ValueError(f"{capability} is not source-native LLMStats data")
        capability_rows = contract.get("rows", [])
        if not capability_rows:
            raise ValueError(f"{capability} publication is empty")
        if len(contract.get("benchmark_columns") or []) < 10:
            raise ValueError(
                f"{capability} publication lost source benchmark columns"
            )
        for row in capability_rows:
            if not row.get("source_name") or not row.get("source_model_url"):
                raise ValueError(f"{capability} row lacks source identity")
            if row.get("category_score") is None and row.get("category_rank") is None:
                raise ValueError(f"{capability} contains an empty source row")
    if not manifest.get("attribution", {}).get("general") or not manifest.get(
        "attribution", {}
    ).get("capabilities"):
        raise ValueError("Publication attribution metadata is missing")
    if quality.get("status") not in {"healthy", "degraded"}:
        raise ValueError(f"Invalid quality status: {quality.get('status')}")
    history_rows = _read_csv("data/history/family_snapshots.csv")
    history_keys = [
        (
            row.get("snapshot_date"),
            row.get("family_id"),
            row.get("score_version"),
        )
        for row in history_rows
    ]
    if len(history_keys) != len(set(history_keys)):
        raise ValueError("Historical family append is not idempotent")

    if any(
        "fable" in (row.get("canonical_name") or "").lower() for row in index
    ) and "Claude Fable" not in history:
        raise ValueError("Claude Fable exists in the index but not family history")
    historical_releases = [
        member
        for members in history.values()
        for member in members
    ]
    dated_releases = [
        member for member in historical_releases if member.get("release_date")
    ]
    if len(dated_releases) < max(10, int(len(historical_releases) * 0.9)):
        raise ValueError(
            "Family history is missing too many release dates: "
            f"{len(dated_releases)}/{len(historical_releases)}"
        )

    expected_models = get_models_for_sentiment(limit=10)
    actual_models = [row.get("model_name") for row in sentiment]
    if len(actual_models) != 10:
        raise ValueError(f"Sentiment must contain exactly 10 models, got {len(actual_models)}")
    if actual_models != expected_models:
        raise ValueError(
            "Sentiment model list does not match the current top ten: "
            f"expected={expected_models}, actual={actual_models}"
        )
    for row in sentiment:
        if row.get("_experimental") is not True:
            raise ValueError(f"Sentiment row is not labeled experimental: {row}")
        if not isinstance(row.get("source_counts"), dict):
            raise ValueError(f"Sentiment source breakdown missing: {row.get('model_name')}")
        unexpected_sources = set(row["source_counts"]) - {
            "Reddit",
            "HackerNews",
            "X",
        }
        if unexpected_sources:
            raise ValueError(
                f"Non-community sentiment sources found for {row.get('model_name')}: "
                f"{sorted(unexpected_sources)}"
            )

    print(
        f"Publication contract passed: {len(index)} models, "
        f"{len(families)} scored/source families, {len(history)} growth families, "
        f"{len(sentiment)} sentiment rows"
    )


if __name__ == "__main__":
    validate_publication()
