"""LLMDEX family-level percentile consensus scoring.

Raw source composites are intentionally never averaged. Each source is ranked
inside the same confidently matched family population, converted to a tie-aware
percentile, then combined 50/50.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pipeline.identity import (
    MATCHED_STATUSES,
    MatchResult,
    generate_match_candidate,
    parse_model_identity,
)


GENERAL_SCORE_VERSION = "LLMDEX General Consensus v1"
CODING_SCORE_VERSION = "LLMDEX Coding Consensus v1"


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def descending_average_ranks(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Return deterministic descending average ranks with ties."""
    indexed = [
        (index, float(value))
        for index, value in enumerate(values)
        if _valid_number(value)
    ]
    indexed.sort(key=lambda item: (-item[1], item[0]))
    output: List[Optional[float]] = [None] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for offset in range(position, end):
            output[indexed[offset][0]] = float(average_rank)
        position = end
    return output


def rank_to_percentile(rank: Optional[float], population_size: int) -> Optional[float]:
    if rank is None or population_size <= 0:
        return None
    if population_size == 1:
        return 100.0
    return 100.0 * (population_size - float(rank)) / (population_size - 1)


def _stable_name(row: Mapping[str, Any]) -> str:
    return str(row.get("source_name") or row.get("canonical_name") or row.get("model_name") or "")


def select_aa_representatives(
    aa_rows: Iterable[Mapping[str, Any]],
    *,
    overrides: Optional[Mapping[str, str]] = None,
    selected_at: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Pick one AA variant per family using source rank and documented ties."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in aa_rows:
        grouped.setdefault(str(row["family_id"]), []).append(dict(row))
    selected: Dict[str, Dict[str, Any]] = {}
    timestamp = selected_at or datetime.now(timezone.utc).isoformat()
    for family_id, variants in grouped.items():
        override_id = (overrides or {}).get(family_id)
        override = next(
            (row for row in variants if row.get("variant_id") == override_id), None
        )
        if override:
            representative = override
            method = "manual_override"
        else:
            representative = min(
                variants,
                key=lambda row: (
                    (
                        float(row.get("performance_rank"))
                        if _valid_number(row.get("performance_rank"))
                        else float("inf")
                    ),
                    (
                        -float(row.get("intelligence_score"))
                        if _valid_number(row.get("intelligence_score"))
                        else float("inf")
                    ),
                    _stable_name(row).casefold(),
                ),
            )
            method = "best_aa_performance_rank_then_intelligence_then_name"
        selected[family_id] = {
            **representative,
            "aa_representative_variant_id": representative.get("variant_id"),
            "aa_representative_name": _stable_name(representative),
            "representative_selection_method": method,
            "representative_selected_at": timestamp,
        }
    return selected


def _agreement_label(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value >= 90:
        return "High agreement"
    if value >= 75:
        return "Moderate agreement"
    return "Low agreement"


def _availability(row: Mapping[str, Any]) -> str:
    current = row.get("availability_class")
    if current in {
        "open_source",
        "open_weights",
        "research_license",
        "proprietary",
        "unknown",
    }:
        return str(current)
    license_text = str(row.get("license_type") or "").casefold()
    if "proprietary" in license_text:
        return "proprietary"
    if "research" in license_text or "non-commercial" in license_text:
        return "research_license"
    if "open source" in license_text:
        return "open_source"
    if row.get("open_source") is True or any(
        token in license_text for token in ("apache", "mit", "community")
    ):
        return "open_weights"
    return "unknown"


def _source_status_label(code: str) -> str:
    return {
        "consensus": "Consensus",
        "aa_only": "AA only",
        "llmstats_only": "LLMStats only",
        "identity_review": "Identity review",
        "family_score_available": "Family score available",
    }.get(code, code.replace("_", " ").title())


def _apply_percentile_consensus(
    family_records: List[Dict[str, Any]],
    *,
    left_field: str,
    right_field: str,
    score_field: str,
    rank_field: str,
    left_percentile_field: str,
    right_percentile_field: str,
    version: str,
    scope: str,
) -> None:
    eligible = [
        row
        for row in family_records
        if _valid_number(row.get(left_field))
        and _valid_number(row.get(right_field))
    ]
    left_ranks = descending_average_ranks([row[left_field] for row in eligible])
    right_ranks = descending_average_ranks([row[right_field] for row in eligible])
    population = len(eligible)
    for row, left_rank, right_rank in zip(eligible, left_ranks, right_ranks):
        left_percentile = rank_to_percentile(left_rank, population)
        right_percentile = rank_to_percentile(right_rank, population)
        row[left_percentile_field] = left_percentile
        row[right_percentile_field] = right_percentile
        row[score_field] = (
            (left_percentile + right_percentile) / 2
            if left_percentile is not None and right_percentile is not None
            else None
        )
        row[f"{score_field}_version"] = version
        row[f"{score_field}_scope"] = scope
        row[f"{score_field}_matched_population_size"] = population

    score_ranks = descending_average_ranks(
        [row.get(score_field) for row in eligible]
    )
    for row, score_rank in zip(eligible, score_ranks):
        row[rank_field] = score_rank


def build_consensus(
    aa_rows: List[Dict[str, Any]],
    llmstats_rows: List[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Return enriched AA variants, LLMStats rows, families and match audit."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    representative_overrides = (overrides or {}).get("representative_overrides", {})
    representatives = select_aa_representatives(
        aa_rows, overrides=representative_overrides, selected_at=timestamp
    )
    match_results: List[MatchResult] = []
    llmstats_enriched: List[Dict[str, Any]] = []
    matched_llmstats_by_family: Dict[str, Dict[str, Any]] = {}
    review_candidate_families = set()

    for source_row in llmstats_rows:
        row = dict(source_row)
        row.setdefault("source", "llmstats")
        match = generate_match_candidate(row, registry, overrides)
        match_results.append(match)
        if match.match_status in MATCHED_STATUSES and match.candidate_family_id:
            family_id = match.candidate_family_id
            row["family_id"] = family_id
            row["canonical_family_name"] = registry[family_id][
                "canonical_family_name"
            ]
            matched_llmstats_by_family[family_id] = row
            aliases = registry[family_id]["source_aliases"].setdefault("llmstats", [])
            if row["source_name"] not in aliases:
                aliases.append(row["source_name"])
        else:
            parsed = parse_model_identity(
                row.get("source_name", ""),
                row.get("provider"),
                source="llmstats",
                source_model_id=row.get("source_model_id"),
                source_url=row.get("source_model_url"),
            )
            row["family_id"] = parsed["family_id"]
            row["canonical_family_name"] = parsed["canonical_family_name"]
            if match.match_status in {"identity_unresolved", "ambiguous"}:
                if match.candidate_family_id:
                    review_candidate_families.add(match.candidate_family_id)
        row.update(
            {
                "match_status": match.match_status,
                "match_confidence": match.match_confidence,
                "match_method": match.match_method,
                "matched_aa_family_id": (
                    match.candidate_family_id
                    if match.match_status in MATCHED_STATUSES
                    else None
                ),
                "score_status": (
                    "consensus"
                    if match.match_status in MATCHED_STATUSES
                    else "identity_review"
                    if match.match_status in {"identity_unresolved", "ambiguous"}
                    else "llmstats_only"
                ),
            }
        )
        row["score_status_label"] = _source_status_label(row["score_status"])
        llmstats_enriched.append(row)

    family_records: List[Dict[str, Any]] = []
    for family_id, representative in representatives.items():
        llmstats = matched_llmstats_by_family.get(family_id)
        score_status = (
            "consensus"
            if llmstats
            and _valid_number(representative.get("intelligence_score"))
            and _valid_number(llmstats.get("general_score"))
            else "identity_review"
            if family_id in review_candidate_families
            else "aa_only"
        )
        family_records.append(
            {
                "family_id": family_id,
                "canonical_family_name": registry[family_id]["canonical_family_name"],
                "provider": registry[family_id].get("provider"),
                "aa_representative_variant_id": representative.get(
                    "aa_representative_variant_id"
                ),
                "aa_representative_name": representative.get(
                    "aa_representative_name"
                ),
                "representative_selection_method": representative.get(
                    "representative_selection_method"
                ),
                "representative_selected_at": timestamp,
                "aa_intelligence": representative.get("intelligence_score"),
                "aa_rank": representative.get("performance_rank")
                or representative.get("source_rank"),
                "aa_official_coding_index": representative.get(
                    "aa_official_coding_index"
                ),
                "llmstats_source_name": llmstats.get("source_name") if llmstats else None,
                "llmstats_source_model_id": (
                    llmstats.get("source_model_id") if llmstats else None
                ),
                "llmstats_source_model_url": (
                    llmstats.get("source_model_url") if llmstats else None
                ),
                "llmstats_general_score": (
                    llmstats.get("general_score") if llmstats else None
                ),
                "llmstats_general_rank": (
                    llmstats.get("general_rank") if llmstats else None
                ),
                "llmstats_coding_score": (
                    (llmstats.get("category_scores") or {}).get("coding")
                    if llmstats
                    else None
                ),
                "score_status": score_status,
                "score_status_label": _source_status_label(score_status),
                "availability_class": _availability(representative),
                "license_name": representative.get("license_name")
                or representative.get("license_type"),
                "weights_available": representative.get("weights_available"),
                "source_code_available": representative.get(
                    "source_code_available"
                ),
                "training_data_disclosed": representative.get(
                    "training_data_disclosed"
                ),
                "commercial_use_allowed": representative.get(
                    "commercial_use_allowed"
                ),
                "input_cost": representative.get("input_cost_per_1m"),
                "output_cost": representative.get("output_cost_per_1m"),
                "tokens_per_second": representative.get("tokens_per_second"),
                "latency": representative.get("latency_seconds"),
                "context_window": representative.get("context_window"),
                "source_coverage": 2 if score_status == "consensus" else 1,
                "mapping_status": (
                    "matched"
                    if llmstats
                    else "review"
                    if score_status == "identity_review"
                    else "source_missing"
                ),
                "generated_at": timestamp,
            }
        )

    matched_family_ids = set(representatives)
    for llmstats in llmstats_enriched:
        if llmstats["family_id"] in matched_family_ids:
            continue
        family_records.append(
            {
                "family_id": llmstats["family_id"],
                "canonical_family_name": llmstats["canonical_family_name"],
                "provider": llmstats.get("provider"),
                "aa_representative_variant_id": None,
                "aa_representative_name": None,
                "aa_intelligence": None,
                "aa_rank": None,
                "aa_official_coding_index": None,
                "llmstats_source_name": llmstats.get("source_name"),
                "llmstats_source_model_id": llmstats.get("source_model_id"),
                "llmstats_source_model_url": llmstats.get("source_model_url"),
                "llmstats_general_score": llmstats.get("general_score"),
                "llmstats_general_rank": llmstats.get("general_rank"),
                "llmstats_coding_score": (
                    llmstats.get("category_scores") or {}
                ).get("coding"),
                "score_status": llmstats["score_status"],
                "score_status_label": llmstats["score_status_label"],
                "availability_class": "unknown",
                "source_coverage": 1,
                "mapping_status": llmstats["match_status"],
                "generated_at": timestamp,
            }
        )

    _apply_percentile_consensus(
        family_records,
        left_field="aa_intelligence",
        right_field="llmstats_general_score",
        score_field="llmdex_score",
        rank_field="llmdex_rank",
        left_percentile_field="aa_percentile",
        right_percentile_field="llmstats_percentile",
        version=GENERAL_SCORE_VERSION,
        scope="confidently_matched_family_intersection",
    )
    for family in family_records:
        if family.get("score_status") == "consensus":
            family["score_version"] = GENERAL_SCORE_VERSION
            family["score_scope"] = "confidently_matched_family_intersection"
            family["matched_population_size"] = family.get(
                "llmdex_score_matched_population_size"
            )
            left = family.get("aa_percentile")
            right = family.get("llmstats_percentile")
            family["agreement"] = (
                100.0 - abs(left - right)
                if left is not None and right is not None
                else None
            )
            family["agreement_label"] = _agreement_label(family["agreement"])
        else:
            family.update(
                {
                    "llmdex_score": None,
                    "llmdex_rank": None,
                    "aa_percentile": None,
                    "llmstats_percentile": None,
                    "agreement": None,
                    "agreement_label": None,
                    "score_version": GENERAL_SCORE_VERSION,
                    "score_scope": "not_scored_missing_or_unapproved_source",
                    "matched_population_size": 0,
                }
            )

    coding_candidates = [
        row
        for row in family_records
        if row.get("mapping_status") == "matched"
        and _valid_number(row.get("aa_official_coding_index"))
        and _valid_number(row.get("llmstats_coding_score"))
    ]
    for row in family_records:
        row["coding_score_status"] = (
            "consensus" if row in coding_candidates else "unavailable"
        )
    _apply_percentile_consensus(
        coding_candidates,
        left_field="aa_official_coding_index",
        right_field="llmstats_coding_score",
        score_field="llmdex_coding_score",
        rank_field="llmdex_coding_rank",
        left_percentile_field="aa_coding_percentile",
        right_percentile_field="llmstats_coding_percentile",
        version=CODING_SCORE_VERSION,
        scope="confidently_matched_family_coding_intersection",
    )

    consensus_families = [
        row
        for row in family_records
        if row.get("score_status") == "consensus"
        and _valid_number(row.get("llmdex_score"))
    ]
    proprietary = [
        row for row in consensus_families if row["availability_class"] == "proprietary"
    ]
    open_models = [
        row
        for row in consensus_families
        if row["availability_class"] in {"open_source", "open_weights"}
    ]
    sota_family_id = (
        max(proprietary, key=lambda row: (row["llmdex_score"], row["family_id"]))[
            "family_id"
        ]
        if proprietary
        else None
    )
    open_sota_family_id = (
        max(open_models, key=lambda row: (row["llmdex_score"], row["family_id"]))[
            "family_id"
        ]
        if open_models
        else None
    )
    for family in family_records:
        family["is_sota"] = family["family_id"] == sota_family_id
        family["is_open_sota"] = family["family_id"] == open_sota_family_id

    family_by_id = {row["family_id"]: row for row in family_records}
    enriched_aa: List[Dict[str, Any]] = []
    for original in aa_rows:
        row = dict(original)
        family = family_by_id[row["family_id"]]
        is_representative = (
            row.get("variant_id") == family.get("aa_representative_variant_id")
        )
        row.update(
            {
                "aa_representative_variant_id": family.get(
                    "aa_representative_variant_id"
                ),
                "aa_representative_name": family.get("aa_representative_name"),
                "representative_selection_method": family.get(
                    "representative_selection_method"
                ),
                "representative_selected_at": family.get(
                    "representative_selected_at"
                ),
                "is_family_representative": is_representative,
                "availability_class": family.get("availability_class"),
                "llmstats_general_score": family.get("llmstats_general_score"),
                "llmstats_general_rank": family.get("llmstats_general_rank"),
                "llmstats_source_name": family.get("llmstats_source_name"),
                "family_score_reference": family.get("family_id"),
                "is_sota": family.get("is_sota") if is_representative else False,
                "is_open_sota": (
                    family.get("is_open_sota") if is_representative else False
                ),
            }
        )
        if is_representative:
            for field in (
                "llmdex_score",
                "llmdex_rank",
                "aa_percentile",
                "llmstats_percentile",
                "agreement",
                "agreement_label",
                "score_version",
                "score_scope",
                "matched_population_size",
                "llmdex_coding_score",
                "llmdex_coding_rank",
            ):
                row[field] = family.get(field)
            row["score_status"] = family["score_status"]
        elif family.get("llmdex_score") is not None:
            row["llmdex_score"] = None
            row["llmdex_rank"] = None
            row["agreement"] = None
            row["score_status"] = "family_score_available"
        else:
            row["llmdex_score"] = None
            row["llmdex_rank"] = None
            row["agreement"] = None
            row["score_status"] = family["score_status"]
        row["score_status_label"] = _source_status_label(row["score_status"])
        badges = []
        if row.get("is_sota"):
            badges.append("SOTA")
        if row.get("is_open_sota"):
            badges.append("OPEN SOTA")
        availability_badge = {
            "open_source": "OPEN SOURCE",
            "open_weights": "OPEN WEIGHTS",
            "research_license": "RESEARCH LICENSE",
            "proprietary": "PROPRIETARY",
            "unknown": "UNKNOWN",
        }.get(row.get("availability_class"))
        if availability_badge:
            badges.append(availability_badge)
        if row["score_status"] == "consensus":
            badges.append("CONSENSUS")
        if row.get("agreement") is not None and row["agreement"] < 75:
            badges.append("LOW AGREEMENT")
        row["badges"] = badges
        enriched_aa.append(row)

    aa_representative_rows = [
        row for row in enriched_aa if row.get("is_family_representative")
    ]
    if aa_representative_rows:
        aa_leader = min(
            aa_representative_rows,
            key=lambda row: (
                row.get("performance_rank")
                if _valid_number(row.get("performance_rank"))
                else float("inf"),
                _stable_name(row),
            ),
        )
        if aa_leader.get("score_status") != "consensus":
            aa_leader["is_aa_leader"] = True
            aa_leader["badges"] = ["AA LEADER", *aa_leader.get("badges", [])]

    return {
        "aa_rows": enriched_aa,
        "llmstats_rows": llmstats_enriched,
        "families": sorted(
            family_records,
            key=lambda row: (
                row.get("llmdex_rank")
                if _valid_number(row.get("llmdex_rank"))
                else float("inf"),
                row["canonical_family_name"].casefold(),
            ),
        ),
        "matches": match_results,
        "registry": registry,
        "generated_at": timestamp,
        "score_version": GENERAL_SCORE_VERSION,
    }


__all__ = [
    "build_consensus",
    "descending_average_ranks",
    "rank_to_percentile",
    "select_aa_representatives",
]
