"""
gemini_advisor.py — Gemini-powered data-grounded conversational advisor.

Gemini receives only structured values from the published LLMDEX datasets.
Missing values remain missing, rankings are never recomputed, and Gemini has
no internet access through this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Response cache and rate limiting
# ──────────────────────────────────────────────────────────────

_response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 600

_rate_limiter: Dict[str, List[float]] = {}
RATE_LIMIT = 5
RATE_WINDOW = 60


def _cache_key(query: str, dataset_version: str = "") -> str:
    value = f"{dataset_version}|{query.strip().casefold()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_cache(
    query: str,
    dataset_version: str = "",
) -> Optional[Dict[str, Any]]:
    key = _cache_key(query, dataset_version)
    entry = _response_cache.get(key)
    if entry is None:
        return None

    if time.time() - entry["timestamp"] >= CACHE_TTL:
        del _response_cache[key]
        return None

    return dict(entry["response"])


def _store_cache(
    query: str,
    response: Dict[str, Any],
    dataset_version: str = "",
) -> None:
    key = _cache_key(query, dataset_version)
    _response_cache[key] = {
        "timestamp": time.time(),
        "response": dict(response),
    }

    if len(_response_cache) > 100:
        oldest_key = min(
            _response_cache,
            key=lambda cache_key: _response_cache[cache_key]["timestamp"],
        )
        del _response_cache[oldest_key]


def _check_rate_limit(user_id: str = "default") -> bool:
    now = time.time()
    timestamps = _rate_limiter.setdefault(user_id, [])
    timestamps[:] = [stamp for stamp in timestamps if now - stamp < RATE_WINDOW]

    if len(timestamps) >= RATE_LIMIT:
        return False

    timestamps.append(now)
    return True


# ──────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the LLMDEX Data-Grounded Advisor.

STRICT RULES:
1. Use ONLY the supplied JSON snapshot. Never invent, estimate, or add outside facts.
2. A null field means unavailable. Never treat null as zero.
3. Before saying data is unavailable, inspect every supplied row for a non-null value.
4. If no supplied row contains the required metric, say exactly: "This data is not available in the current dataset."
5. Be neutral and support every conclusion with specific values.
6. Never alter, override, or silently recompute a published rank.
7. If a named model is absent from the snapshot, say that it is absent.
8. Keep the response concise and practical.
9. Return valid JSON only in the required structure.

METRICS:
- llmdex_score / llmdex_rank: cross-source General consensus score and rank.
- llmdex_coding_score / llmdex_coding_rank: cross-source Coding consensus score and rank.
- aa_intelligence / aa_rank: Artificial Analysis General score and rank.
- llmstats_general_score: LLMStats General score.
- performance_rank: published performance leaderboard rank; lower is better.
- value_rank: published value leaderboard rank; lower is better.
- efficiency_rank: published cost-efficiency rank; lower is better.
- adjusted_performance: published source-native performance score.
- input_cost_per_1m / output_cost_per_1m: USD per one million tokens.
- blended_cost_per_1m: 60% input cost plus 40% output cost; lower is cheaper.
- tokens_per_second: measured output throughput; higher is faster.
- latency_seconds: measured latency; lower is faster.
- speed_index: published normalized speed score; higher is better.
- context_window: maximum context length in tokens; higher is larger.

IMPORTANT:
- Fastest is not the same as most efficient.
- Use tokens_per_second for generation throughput.
- Use latency_seconds for responsiveness.
- For a generic "fastest model" question, report both the highest-throughput model and the lowest-latency model because they may differ.
- Never use efficiency_rank as a substitute for speed.
- Use published value_rank and efficiency_rank instead of creating a new ranking.

Each row may contain:
model_name, provider, family_id, llmdex_score, llmdex_rank,
llmdex_coding_score, llmdex_coding_rank, score_status, agreement_label,
aa_intelligence, aa_rank, llmstats_general_score, llmstats_coding_score,
performance_rank, value_rank, efficiency_rank, adjusted_performance,
intelligence_score, input_cost_per_1m, output_cost_per_1m,
blended_cost_per_1m, tokens_per_second, latency_seconds, speed_index,
context_window, availability_class, coding_score, reasoning_score,
confidence_factor, snapshot_date.

Return JSON with this exact structure:
{
  "answer": "Your analytical response as a string",
  "referenced_models": ["model1", "model2"],
  "data_points_used": ["tokens_per_second", "latency_seconds"]
}
"""


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _round_safe(value: Any, decimals: int = 2) -> Optional[float]:
    if not _is_number(value):
        return None
    return round(float(value), decimals)


def _first_value(row: dict, *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None:
            return value
    return None


def _first_number(row: dict, *fields: str) -> Optional[float]:
    for field in fields:
        value = row.get(field)
        if _is_number(value):
            return float(value)
    return None


def _normalize(value: Any) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in str(value or "")
        ).split()
    )


def _model_name(row: dict) -> str:
    return str(
        row.get("canonical_family_name")
        or row.get("canonical_name")
        or row.get("model_name")
        or row.get("source_name")
        or row.get("aa_representative_name")
        or "Unknown"
    )


def _compute_blended_cost(row: dict) -> Optional[float]:
    published = _first_number(row, "blended_cost_per_1m")
    if published is not None:
        return published

    input_cost = _first_number(row, "input_cost_per_1m", "input_cost")
    output_cost = _first_number(row, "output_cost_per_1m", "output_cost")

    values: List[tuple[float, float]] = []
    if input_cost is not None:
        values.append((input_cost, 0.6))
    if output_cost is not None:
        values.append((output_cost, 0.4))

    if not values:
        return None

    total_weight = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total_weight


def _read_rows(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        logger.warning("Could not load %s: %s", path, error)
        return []

    if isinstance(payload, dict):
        payload = payload.get("rows", [])

    if not isinstance(payload, list):
        return []

    return [row for row in payload if isinstance(row, dict)]


_SCORED_FIELDS = (
    "performance_rank",
    "value_rank",
    "efficiency_rank",
    "performance_index",
    "adjusted_performance",
    "intelligence_score",
    "composite_index",
    "efficiency_score",
    "blended_cost_per_1m",
    "input_cost_per_1m",
    "output_cost_per_1m",
    "tokens_per_second",
    "latency_seconds",
    "latency_first_token",
    "speed_index",
    "coding_score",
    "reasoning_score",
    "confidence_factor",
    "context_window",
    "snapshot_date",
    "last_updated",
)


def _row_identifiers(row: dict) -> Iterable[str]:
    for field in (
        "variant_id",
        "model_id",
        "canonical_model_id",
        "source_model_id",
        "model_slug",
    ):
        value = _normalize(row.get(field))
        if value:
            yield value


def _scored_sort_key(row: dict) -> tuple[float, float, str]:
    rank = _first_number(row, "performance_rank", "source_rank")
    performance = _first_number(
        row,
        "adjusted_performance",
        "performance_index",
        "intelligence_score",
    )
    return (
        rank if rank is not None else 999999.0,
        -(performance if performance is not None else -999999.0),
        _normalize(_model_name(row)),
    )


def _merge_family_and_scored_rows(
    family_rows: List[dict],
    scored_rows: List[dict],
) -> List[dict]:
    """Keep family consensus fields and fill missing source-native fields."""
    by_family: Dict[str, List[dict]] = {}
    by_identifier: Dict[str, List[dict]] = {}
    by_name: Dict[str, List[dict]] = {}

    for row in scored_rows:
        family_id = _normalize(row.get("family_id"))
        if family_id:
            by_family.setdefault(family_id, []).append(row)

        for identifier in _row_identifiers(row):
            by_identifier.setdefault(identifier, []).append(row)

        for field in ("canonical_name", "model_name", "source_name"):
            name = _normalize(row.get(field))
            if name:
                by_name.setdefault(name, []).append(row)

    merged_rows: List[dict] = []

    for family in family_rows:
        merged = dict(family)
        candidates: List[dict] = []

        representative_id = _normalize(family.get("aa_representative_variant_id"))
        if representative_id:
            candidates = by_identifier.get(representative_id, [])

        if not candidates:
            family_id = _normalize(family.get("family_id"))
            candidates = by_family.get(family_id, [])

        if not candidates:
            representative_name = _normalize(family.get("aa_representative_name"))
            candidates = by_name.get(representative_name, [])

        if not candidates:
            canonical_name = _normalize(family.get("canonical_family_name"))
            candidates = by_name.get(canonical_name, [])

        if candidates:
            representative = sorted(candidates, key=_scored_sort_key)[0]
            for field in _SCORED_FIELDS:
                if merged.get(field) is None and representative.get(field) is not None:
                    merged[field] = representative[field]

        merged_rows.append(merged)

    return merged_rows


def _load_dataset(index_path: Optional[str] = None) -> List[dict]:
    """
    Load one explicit test file, or merge the normal family and scored files.
    """
    if index_path is not None:
        return _read_rows(index_path)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    family_rows = _read_rows(
        os.path.join(root, "data", "families", "latest.json")
    )
    scored_rows = _read_rows(
        os.path.join(root, "data", "index", "latest.json")
    )

    if family_rows and scored_rows:
        return _merge_family_and_scored_rows(family_rows, scored_rows)
    return family_rows or scored_rows


def _dataset_version(dataset: List[dict]) -> str:
    dates = sorted(
        {
            str(
                row.get("generated_at")
                or row.get("snapshot_date")
                or row.get("last_updated")
                or ""
            )
            for row in dataset
            if row.get("generated_at")
            or row.get("snapshot_date")
            or row.get("last_updated")
        }
    )
    value = f"{len(dataset)}|{'|'.join(dates[-3:])}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────
# Query-aware snapshot
# ──────────────────────────────────────────────────────────────


def _contains(query: str, *terms: str) -> bool:
    return any(term in query for term in terms)


def _primary_rank(row: dict) -> float:
    rank = _first_number(row, "llmdex_rank", "aa_rank", "performance_rank")
    return rank if rank is not None else 999999.0


def _has_data(row: dict) -> bool:
    """Return True when a row has any publishable ranking or metric.

    Ranking-only rows must remain eligible for query-aware matching even when
    detailed metric values are unavailable. Missing values are still
    serialized as null and are never converted to zero.
    """
    fields = (
        "llmdex_rank",
        "llmdex_score",
        "llmdex_coding_rank",
        "llmdex_coding_score",
        "aa_rank",
        "performance_rank",
        "value_rank",
        "efficiency_rank",
        "aa_intelligence",
        "llmstats_general_score",
        "adjusted_performance",
        "intelligence_score",
        "input_cost",
        "output_cost",
        "input_cost_per_1m",
        "output_cost_per_1m",
        "blended_cost_per_1m",
        "tokens_per_second",
        "latency",
        "latency_seconds",
        "speed_index",
        "context_window",
        "aa_official_coding_index",
        "llmstats_coding_score",
        "coding_score",
        "reasoning_score",
        "confidence_factor",
    )

    return any(row.get(field) is not None for field in fields)


def _extract_compact_snapshot(
    dataset: List[dict],
    user_query: str = "",
    max_models: int = 30,
) -> List[dict]:
    query = user_query.casefold()
    rows = [row for row in dataset if _has_data(row)]
    selected: Dict[str, dict] = {}

    def add(candidate_rows: Iterable[dict], limit: int) -> None:
        count = 0
        for row in candidate_rows:
            if count >= limit or len(selected) >= max_models:
                break
            key = _normalize(row.get("family_id")) or _normalize(_model_name(row))
            if not key or key in selected:
                continue
            selected[key] = row
            count += 1

    query_tokens = {
        token for token in _normalize(user_query).split() if len(token) >= 3
    }
    if query_tokens:
        matches: List[tuple[int, float, dict]] = []
        for row in rows:
            aliases = row.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            searchable = _normalize(
                " ".join(
                    str(value or "")
                    for value in (
                        _model_name(row),
                        row.get("provider"),
                        row.get("aa_representative_name"),
                        row.get("llmstats_source_name"),
                        " ".join(str(alias) for alias in aliases),
                    )
                )
            )
            overlap = sum(token in searchable for token in query_tokens)
            if overlap:
                matches.append((overlap, _primary_rank(row), row))
        matches.sort(key=lambda item: (-item[0], item[1], _model_name(item[2])))
        add((row for _, _, row in matches), 12)

    speed_query = _contains(
        query,
        "fast",
        "speed",
        "throughput",
        "tokens per second",
        "token per second",
        "tps",
    )
    latency_query = _contains(
        query,
        "latency",
        "responsive",
        "responsiveness",
        "quickest response",
        "lowest delay",
    )

    if speed_query:
        throughput_rows = [
            row for row in rows if _first_number(row, "tokens_per_second") is not None
        ]
        throughput_rows.sort(
            key=lambda row: -float(_first_number(row, "tokens_per_second") or 0)
        )
        add(throughput_rows, 12)

    if speed_query or latency_query:
        latency_rows = [
            row
            for row in rows
            if _first_number(row, "latency_seconds", "latency") is not None
        ]
        latency_rows.sort(
            key=lambda row: float(
                _first_number(row, "latency_seconds", "latency")
                if _first_number(row, "latency_seconds", "latency") is not None
                else 999999
            )
        )
        add(latency_rows, 12)

    if _contains(query, "cheap", "cheapest", "cost", "price", "budget"):
        cost_rows = [row for row in rows if _compute_blended_cost(row) is not None]
        cost_rows.sort(
            key=lambda row: float(
                _compute_blended_cost(row)
                if _compute_blended_cost(row) is not None
                else 999999
            )
        )
        add(cost_rows, 12)

    if _contains(query, "value", "performance-to-cost", "bang for"):
        value_rows = [row for row in rows if _first_number(row, "value_rank")]
        value_rows.sort(key=lambda row: float(row["value_rank"]))
        add(value_rows, 12)

    if _contains(query, "efficient", "efficiency", "performance per dollar"):
        efficiency_rows = [
            row for row in rows if _first_number(row, "efficiency_rank")
        ]
        efficiency_rows.sort(key=lambda row: float(row["efficiency_rank"]))
        add(efficiency_rows, 12)

    if _contains(
        query,
        "context",
        "long document",
        "long context",
        "maximum input",
        "max input",
    ):
        context_rows = [
            row for row in rows if _first_number(row, "context_window") is not None
        ]
        context_rows.sort(
            key=lambda row: -float(_first_number(row, "context_window") or 0)
        )
        add(context_rows, 10)

    if _contains(query, "coding", "code", "programming", "developer"):
        coding_rows = [
            row
            for row in rows
            if _first_number(
                row,
                "llmdex_coding_score",
                "aa_official_coding_index",
                "llmstats_coding_score",
                "coding_score",
            )
            is not None
        ]
        coding_rows.sort(
            key=lambda row: (
                _first_number(row, "llmdex_coding_rank") or 999999.0,
                -float(
                    _first_number(
                        row,
                        "llmdex_coding_score",
                        "aa_official_coding_index",
                        "llmstats_coding_score",
                        "coding_score",
                    )
                    or 0
                ),
            )
        )
        add(coding_rows, 12)

    if _contains(query, "reason", "math", "logic", "analytical"):
        reasoning_rows = [
            row for row in rows if _first_number(row, "reasoning_score") is not None
        ]
        reasoning_rows.sort(
            key=lambda row: -float(_first_number(row, "reasoning_score") or 0)
        )
        add(reasoning_rows, 12)

    add(sorted(rows, key=_primary_rank), 12)

    value_rows = [row for row in rows if _first_number(row, "value_rank")]
    value_rows.sort(key=lambda row: float(row["value_rank"]))
    add(value_rows, 8)

    efficiency_rows = [
        row for row in rows if _first_number(row, "efficiency_rank")
    ]
    efficiency_rows.sort(key=lambda row: float(row["efficiency_rank"]))
    add(efficiency_rows, 8)

    snapshot: List[dict] = []

    for row in list(selected.values())[:max_models]:
        input_cost = _first_number(row, "input_cost_per_1m", "input_cost")
        output_cost = _first_number(row, "output_cost_per_1m", "output_cost")
        latency = _first_number(row, "latency_seconds", "latency")
        adjusted_performance = _first_number(
            row,
            "adjusted_performance",
            "aa_intelligence",
            "intelligence_score",
            "performance_index",
        )

        snapshot.append(
            {
                "family_id": row.get("family_id"),
                "model_name": _model_name(row),
                "provider": row.get("provider") or "Unknown",
                "llmdex_rank": _round_safe(row.get("llmdex_rank")),
                "llmdex_score": _round_safe(row.get("llmdex_score")),
                "llmdex_coding_rank": _round_safe(row.get("llmdex_coding_rank")),
                "llmdex_coding_score": _round_safe(row.get("llmdex_coding_score")),
                "score_status": row.get("score_status"),
                "agreement_label": row.get("agreement_label"),
                "aa_rank": _round_safe(row.get("aa_rank")),
                "aa_intelligence": _round_safe(
                    _first_value(row, "aa_intelligence", "intelligence_score")
                ),
                "llmstats_general_score": _round_safe(
                    row.get("llmstats_general_score")
                ),
                "aa_official_coding_index": _round_safe(
                    row.get("aa_official_coding_index")
                ),
                "llmstats_coding_score": _round_safe(
                    row.get("llmstats_coding_score")
                ),
                "performance_rank": _round_safe(row.get("performance_rank")),
                "value_rank": _round_safe(row.get("value_rank")),
                "efficiency_rank": _round_safe(row.get("efficiency_rank")),
                "adjusted_performance": _round_safe(adjusted_performance),
                "intelligence_score": _round_safe(row.get("intelligence_score")),
                "input_cost_per_1m": _round_safe(input_cost),
                "output_cost_per_1m": _round_safe(output_cost),
                "blended_cost_per_1m": _round_safe(_compute_blended_cost(row)),
                "tokens_per_second": _round_safe(row.get("tokens_per_second")),
                "latency_seconds": _round_safe(latency),
                "speed_index": _round_safe(row.get("speed_index")),
                "context_window": (
                    int(float(row["context_window"]))
                    if _is_number(row.get("context_window"))
                    else None
                ),
                "availability_class": row.get("availability_class"),
                "coding_score": _round_safe(
                    _first_value(
                        row,
                        "coding_score",
                        "aa_official_coding_index",
                        "llmstats_coding_score",
                    )
                ),
                "reasoning_score": _round_safe(row.get("reasoning_score")),
                "confidence_factor": _round_safe(row.get("confidence_factor")),
                "snapshot_date": _first_value(
                    row,
                    "snapshot_date",
                    "last_updated",
                    "generated_at",
                ),
            }
        )

    return snapshot


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────


def generate_advisor_response(
    user_query: str,
    user_id: str = "default",
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    query = str(user_query or "").strip()

    if not query:
        return {
            "answer": "Please enter a question about the published LLMDEX model data.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "validation_error",
        }

    if len(query) > 500:
        return {
            "answer": "Please keep the question under 500 characters.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "validation_error",
        }

    if not _check_rate_limit(user_id):
        return {
            "answer": "Rate limit exceeded. Please wait a moment before asking another question. Maximum 5 queries per minute.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "rate_limit",
        }

    dataset = _load_dataset(index_path)
    if not dataset:
        return {
            "answer": "Unable to load model data. The dataset may not be available yet.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "error",
        }

    version = _dataset_version(dataset)
    cached = _check_cache(query, version)
    if cached is not None:
        cached["source"] = "cache"
        return cached

    snapshot = _extract_compact_snapshot(dataset, user_query=query, max_models=30)
    if not snapshot:
        return {
            "answer": "This data is not available in the current dataset.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "error",
        }

    prompt = f"""DATASET SNAPSHOT ({len(snapshot)} query-relevant and leaderboard-leading models from the current LLMDEX publication):
{json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)}

USER QUESTION: {query}

Use only the supplied values. For maximum metrics such as tokens_per_second and context_window, select the highest non-null value. For minimum metrics such as latency_seconds and cost, select the lowest non-null value. Return only the required JSON object."""

    try:
        from utils.gemini_client import call_gemini
    except ImportError:
        logger.exception("utils.gemini_client could not be imported")
        return _fallback_response()

    try:
        result = call_gemini(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            pool_type="advisor",
            temperature=0.2,
            max_output_tokens=900,
            thinking_level="minimal",
        )
    except Exception:
        logger.exception("Gemini advisor request failed")
        return _fallback_response()

    if not isinstance(result, dict):
        return _fallback_response()

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        logger.warning("Gemini advisor response did not contain a valid answer")
        return _fallback_response()

    referenced_models = result.get("referenced_models", [])
    if not isinstance(referenced_models, list):
        referenced_models = []

    data_points_used = result.get("data_points_used", [])
    if not isinstance(data_points_used, list):
        data_points_used = []

    response = {
        "answer": answer.strip(),
        "referenced_models": [
            str(model) for model in referenced_models[:10] if model is not None
        ],
        "data_points_used": [
            str(point) for point in data_points_used[:12] if point is not None
        ],
        "source": "gemini",
    }

    _store_cache(query, response, version)
    return response


def _fallback_response() -> Dict[str, Any]:
    return {
        "answer": "AI advisor temporarily unavailable. Please use the ranking filters and priority selector below to find the best models for your needs.",
        "referenced_models": [],
        "data_points_used": [],
        "source": "fallback",
        "diagnostic_code": "gemini_unavailable",
    }
