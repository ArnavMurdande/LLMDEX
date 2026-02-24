"""
scoring.py — Scoring engine for the LLM Intelligence Index.

SAFETY DESIGN:
    Replaces the old max()-based aggregation with a WEIGHTED MEAN
    using source confidence weights.

    Computes four sub-indices:
        - performance_index: weighted mean of benchmark scores
        - cost_index: inverted cost normalization (cheaper = higher)
        - speed_index: from latency and tokens/second
        - composite_index: weighted combination of above three

UPGRADES (v3):
    - THREE INDEPENDENT LEADERBOARDS:
        performance_rank  — pure intelligence, no cost/speed influence
        value_rank        — composite_index (existing weighted blend)
        efficiency_rank   — performance / blended cost (threshold: 60+)
    - BIAS CORRECTION:
        adjusted_performance uses confidence_factor to prevent
        sparse-data models from unfairly dominating.
    - EFFICIENCY PERCENTILE NORMALIZATION:
        Replaces hard cap (min(100, x*5)) with percentile ranking
        to preserve relative spacing without distortion.
    - Deterministic tie-breaking in ranking (by model_name).
    - Coverage score computed from source diversity.

    RULES:
        - Missing values are EXCLUDED from aggregation, not treated as 0.
        - Efficiency score is UNDEFINED if cost is missing.
        - Latency is IGNORED if missing, not replaced with median.
        - Models missing ALL core benchmarks are excluded from ranking.
        - Composite index computation is documented and reproducible.
        - Normalization uses anchored ranges, NOT daily min/max.
        - Sentiment NEVER influences any ranking.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import math

import numpy as np
import pandas as pd

from utils.normalizer import normalize_anchored, normalize_series_anchored

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Source confidence weights for multi-source aggregation.
# ──────────────────────────────────────────────────────────────
SOURCE_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "LiveBench": 0.9,
    "LMSYS Chatbot Arena": 0.95,
    "Artificial Analysis": 0.8,
    "LLM-Stats": 0.7,
    "Vellum": 0.75,
}

DEFAULT_SOURCE_WEIGHT = 0.5

# ──────────────────────────────────────────────────────────────
# Composite index weights (used for value_rank).
# ──────────────────────────────────────────────────────────────
COMPOSITE_WEIGHTS = {
    "performance": 0.50,
    "cost": 0.30,
    "speed": 0.20,
}

# ──────────────────────────────────────────────────────────────
# Benchmark weights for performance index.
# ──────────────────────────────────────────────────────────────
BENCHMARK_WEIGHTS = {
    "intelligence_score": 0.45,   # AA Intelligence Index (primary)
    "coding_score": 0.25,         # AA Coding Index
    "arena_elo": 0.15,            # LMSYS Arena ELO (supplementary, fixed)
    "gpqa": 0.15,                 # GPQA Diamond (granular benchmark)
}

# ──────────────────────────────────────────────────────────────
# Efficiency ranking threshold: only models with adjusted
# performance >= this value qualify for efficiency_rank.
# ──────────────────────────────────────────────────────────────
EFFICIENCY_PERF_THRESHOLD = 25

# Max possible performance source count (for confidence factor)
# 3 dimensions: AA benchmarks, LMSYS arena, cost data
MAX_POSSIBLE_PERF_SOURCES = 3


def _is_valid(value) -> bool:
    """
    Check if a value is both non-None and non-NaN.
    """
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, (np.floating, np.integer)) and np.isnan(value):
            return False
    except (TypeError, ValueError):
        pass
    return True


def compute_source_weight(source: str, row_confidence: float = 1.0) -> float:
    """
    Get the effective weight for a data point from a given source.
    Combines the source-level trust weight with the row-level parse confidence.
    """
    base = SOURCE_CONFIDENCE_WEIGHTS.get(source, DEFAULT_SOURCE_WEIGHT)
    return base * row_confidence


def weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    """
    Compute weighted mean. Returns None if inputs are empty.
    """
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def compute_performance_index(row: dict) -> Optional[float]:
    """
    Compute the performance index from available benchmark scores.

    CRITICAL DESIGN: AA-first scoring.
    
    AA benchmark fields (intelligence_score, coding_score, gpqa) are the
    PRIMARY performance drivers. arena_elo is a SUPPLEMENTARY signal at
    a fixed 15% weight — it NEVER inflates to fill missing benchmarks.
    
    This means:
    - AA model with intelligence=57 + coding=46 + gpqa=94: perf ≈ 56
    - AA+LMSYS model with same + elo=1500: perf ≈ 70  
    - LMSYS-only model with just elo=1500: perf ≈ 13 (not 85!)
    
    Weight redistribution only happens among AA benchmark fields.
    """
    # AA benchmark fields (these redistribute among themselves)
    aa_fields = {
        "intelligence_score": ("intelligence_score", False),
        "coding_score": ("coding_score", False),
        "gpqa": ("gpqa", False),
    }
    
    # Compute AA benchmark scores
    aa_scores: List[float] = []
    aa_weights: List[float] = []
    
    for field, (metric, invert) in aa_fields.items():
        val = row.get(field)
        if _is_valid(val):
            norm = normalize_anchored(float(val), metric, invert)
            if norm is not None:
                aa_scores.append(norm)
                aa_weights.append(BENCHMARK_WEIGHTS.get(field, 0.1))

    # Compute arena_elo contribution (FIXED weight, never redistributed)
    arena_elo_contribution = 0.0
    arena_elo_weight = BENCHMARK_WEIGHTS.get("arena_elo", 0.15)
    has_arena = False
    
    elo_val = row.get("arena_elo")
    if _is_valid(elo_val):
        elo_norm = normalize_anchored(float(elo_val), "arena_elo", False)
        if elo_norm is not None:
            arena_elo_contribution = elo_norm * arena_elo_weight
            has_arena = True

    if not aa_scores and not has_arena:
        return None

    if aa_scores:
        # Redistribute weights among available AA benchmarks
        # They share the non-arena portion (85%) of the total
        aa_total_weight = sum(aa_weights)
        aa_target_weight = 1.0 - arena_elo_weight  # 0.85
        
        aa_contribution = sum(
            s * (w / aa_total_weight) * aa_target_weight
            for s, w in zip(aa_scores, aa_weights)
        )
        
        result = aa_contribution + arena_elo_contribution
    else:
        # LMSYS-only: arena_elo is the ONLY source
        # Cap at its natural fixed weight — don't inflate
        result = arena_elo_contribution

    return round(result, 2)


def compute_cost_index(row: dict) -> Optional[float]:
    """
    Compute cost index (cheaper = higher score).
    Returns None if cost is missing.
    """
    blended_cost = compute_blended_cost(row)
    if blended_cost is None:
        return None

    return normalize_anchored(blended_cost, "input_cost_per_1m", invert=True)


def compute_speed_index(row: dict) -> Optional[float]:
    """
    Compute speed index from latency and tokens/second.
    Returns None if both are missing.
    """
    scores: List[float] = []

    latency = row.get("latency_seconds")
    if _is_valid(latency):
        norm = normalize_anchored(float(latency), "latency_seconds", invert=True)
        if norm is not None:
            scores.append(norm)

    tps = row.get("tokens_per_second")
    if _is_valid(tps):
        norm = normalize_anchored(float(tps), "tokens_per_second", invert=False)
        if norm is not None:
            scores.append(norm)

    if not scores:
        return None

    return round(sum(scores) / len(scores), 2)


def compute_composite_index(
    performance: Optional[float],
    cost: Optional[float],
    speed: Optional[float],
) -> Optional[float]:
    """
    Compute the composite index from sub-indices.
    Used for value_rank leaderboard.

    SAFETY:
    - If performance is None, the model is EXCLUDED from ranking.
    - If cost or speed is None, we redistribute their weight to
      performance rather than filling in a fake value.
    """
    if not _is_valid(performance):
        return None

    available = {"performance": float(performance)}
    if _is_valid(cost):
        available["cost"] = float(cost)
    if _is_valid(speed):
        available["speed"] = float(speed)

    # Renormalize weights over available indices
    raw_weights = {k: COMPOSITE_WEIGHTS[k] for k in available}
    total_weight = sum(raw_weights.values())

    if total_weight == 0:
        return None

    composite = sum(
        available[k] * (raw_weights[k] / total_weight)
        for k in available
    )

    return round(composite, 2)


def compute_blended_cost(row: dict) -> Optional[float]:
    """
    Compute blended cost (60% input + 40% output).
    Returns None if no cost data is available.
    
    Also checks for pre-computed blended_cost_per_1m from AA.
    """
    input_cost = row.get("input_cost_per_1m")
    output_cost = row.get("output_cost_per_1m")

    if _is_valid(input_cost) or _is_valid(output_cost):
        costs = []
        weights = []
        if _is_valid(input_cost):
            costs.append(float(input_cost))
            weights.append(0.6)
        if _is_valid(output_cost):
            costs.append(float(output_cost))
            weights.append(0.4)

        if costs:
            return sum(c * w for c, w in zip(costs, weights)) / sum(weights)

    # Fallback: use pre-computed blended cost from AA
    blended = row.get("blended_cost_per_1m")
    if _is_valid(blended):
        return float(blended)

    return None


def compute_raw_efficiency(row: dict) -> Optional[float]:
    """
    Compute RAW efficiency = performance / blended_cost.

    SAFETY: Returns None if EITHER performance or cost is missing.
    Returns a large value (9999) for free models with performance data.

    This is the raw ratio BEFORE percentile normalization.
    """
    perf = row.get("performance_index")

    if not _is_valid(perf):
        return None

    blended = compute_blended_cost(row)
    if blended is None:
        return None

    if blended <= 0:
        # Free models get very high raw efficiency
        return 9999.0

    return float(perf) / blended


def compute_efficiency_percentile(df: pd.DataFrame) -> pd.Series:
    """
    Compute efficiency as a PERCENTILE rank (0-100).

    PART 3 FIX: Replaces hard cap scaling (min(100, x*5)) with
    percentile normalization that preserves relative spacing
    and never artificially clamps the distribution.

    Formula: efficiency_percentile = rank(raw_efficiency) / N * 100
    """
    raw = df["raw_efficiency"]
    valid_mask = raw.notna()
    result = pd.Series(np.nan, index=df.index)

    if valid_mask.sum() == 0:
        return result

    # Percentile rank: 0 = worst, 100 = best
    valid_values = raw[valid_mask]
    ranked = valid_values.rank(method="average", ascending=True)
    n = len(ranked)
    percentiles = (ranked / n) * 100.0
    result[valid_mask] = percentiles.round(2)

    return result


def compute_coverage_score(row: dict) -> float:
    """
    Compute a coverage score based on how many data dimensions are present.
    Returns a value 0–100.
    """
    fields = [
        "intelligence_score", "coding_score", "reasoning_score",
        "multimodal_score", "arena_elo",
        "input_cost_per_1m", "output_cost_per_1m",
        "context_window", "latency_seconds", "tokens_per_second",
    ]
    present = sum(1 for f in fields if _is_valid(row.get(f)))
    return round(100 * present / len(fields), 1)


def count_perf_sources(row: dict) -> int:
    """Count how many key data dimensions this model has.
    
    Uses 3 realistic dimensions that our scrapers actually provide:
    1. intelligence_score (from AA) or coding_score
    2. arena_elo (from LMSYS)
    3. cost data (blended_cost or input/output cost from AA)
    """
    count = 0
    # Dimension 1: AA benchmark data
    if _is_valid(row.get("intelligence_score")) or _is_valid(row.get("coding_score")):
        count += 1
    # Dimension 2: LMSYS arena data
    if _is_valid(row.get("arena_elo")):
        count += 1
    # Dimension 3: Cost/economics data
    if (_is_valid(row.get("blended_cost_per_1m")) or 
        _is_valid(row.get("input_cost_per_1m")) or 
        _is_valid(row.get("output_cost_per_1m"))):
        count += 1
    return count


def compute_confidence_factor(perf_source_count: int) -> float:
    """
    PART 2: Confidence weighting to prevent sparse-data bias.

    confidence_factor = perf_source_count / max_possible_sources

    Based on 3 data dimensions (AA benchmarks, LMSYS arena, cost data):
    - 3/3 = 1.0  → Cross-referenced model with full data
    - 2/3 = 0.67 → Model with two data dimensions
    - 1/3 = 0.33 → Single-source model
    """
    return perf_source_count / MAX_POSSIBLE_PERF_SOURCES


def compute_adjusted_performance(
    performance_index: Optional[float],
    perf_source_count: int,
) -> Optional[float]:
    """
    PART 2: Apply bias correction to performance index.

    adjusted_performance = performance_index * (0.85 + 0.15 * confidence_factor)

    This means:
    - Full coverage (5/5): multiplier = 0.85 + 0.15*1.0 = 1.00 (no penalty)
    - 3/5 coverage: multiplier = 0.85 + 0.15*0.6 = 0.94
    - 1/5 coverage: multiplier = 0.85 + 0.15*0.2 = 0.88

    This prevents sparse data from dominating while not breaking
    elite models that genuinely have all benchmarks.
    """
    if not _is_valid(performance_index):
        return None

    cf = compute_confidence_factor(perf_source_count)
    multiplier = 0.85 + 0.15 * cf
    return round(float(performance_index) * multiplier, 2)


def score_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indices for every model in the dataset.

    This is the main entry point for the scoring engine.

    UPGRADES (v3):
    - Computes adjusted_performance with bias correction.
    - Creates THREE independent leaderboards.
    - Uses percentile normalization for efficiency.
    """
    logger.info(f"Scoring {len(df)} models...")

    # ── Sub-indices ──
    df["performance_index"] = df.apply(
        lambda row: compute_performance_index(row.to_dict()), axis=1
    )
    df["cost_index"] = df.apply(
        lambda row: compute_cost_index(row.to_dict()), axis=1
    )
    df["speed_index"] = df.apply(
        lambda row: compute_speed_index(row.to_dict()), axis=1
    )

    # ── Coverage & source count ──
    df["coverage_score"] = df.apply(
        lambda row: compute_coverage_score(row.to_dict()), axis=1
    )
    df["perf_source_count"] = df.apply(
        lambda row: count_perf_sources(row.to_dict()), axis=1
    )

    # ── PART 2: Confidence-weighted bias correction ──
    df["confidence_factor"] = df["perf_source_count"].apply(compute_confidence_factor)
    df["adjusted_performance"] = df.apply(
        lambda row: compute_adjusted_performance(
            row.get("performance_index"),
            int(row.get("perf_source_count", 0)),
        ),
        axis=1,
    )

    # ── Composite index (uses adjusted performance for fairness) ──
    df["composite_index"] = df.apply(
        lambda row: compute_composite_index(
            row.get("adjusted_performance"),
            row.get("cost_index"),
            row.get("speed_index"),
        ),
        axis=1,
    )

    # ── PART 3: Raw efficiency + percentile normalization ──
    df["raw_efficiency"] = df.apply(
        lambda row: compute_raw_efficiency(row.to_dict()), axis=1
    )
    df["efficiency_score"] = compute_efficiency_percentile(df)

    # ══════════════════════════════════════════════════════════
    # PART 1: THREE INDEPENDENT LEADERBOARDS
    # ══════════════════════════════════════════════════════════

    # ── 1. performance_rank: AA-first, then by adjusted performance ──
    perf_eligible = df[df["adjusted_performance"].notna()].copy()
    perf_eligible = perf_eligible.sort_values(
        by=["data_tier", "adjusted_performance", "performance_index", "model_name"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    perf_eligible["performance_rank"] = range(1, len(perf_eligible) + 1)
    df = df.merge(
        perf_eligible[["performance_rank"]],
        left_index=True, right_index=True, how="left",
    )

    # ── 2. value_rank: AA-first, then composite_index ──
    value_eligible = df[df["composite_index"].notna()].copy()
    value_eligible = value_eligible.sort_values(
        by=["data_tier", "composite_index", "adjusted_performance", "model_name"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    value_eligible["value_rank"] = range(1, len(value_eligible) + 1)
    df = df.merge(
        value_eligible[["value_rank"]],
        left_index=True, right_index=True, how="left",
    )

    # ── 3. efficiency_rank: perf/cost, threshold >= 60 ──
    eff_eligible = df[
        (df["adjusted_performance"].notna()) &
        (df["adjusted_performance"] >= EFFICIENCY_PERF_THRESHOLD) &
        (df["raw_efficiency"].notna())
    ].copy()
    eff_eligible = eff_eligible.sort_values(
        by=["efficiency_score", "adjusted_performance", "model_name"],
        ascending=[False, False, True],
        na_position="last",
    )
    eff_eligible["efficiency_rank"] = range(1, len(eff_eligible) + 1)
    df = df.merge(
        eff_eligible[["efficiency_rank"]],
        left_index=True, right_index=True, how="left",
    )

    # ── Legacy: global_rank = value_rank for backward compatibility ──
    df["global_rank"] = df["value_rank"]

    # ── Logging ──
    scored = int(df["composite_index"].notna().sum())
    unscored = int(df["composite_index"].isna().sum())
    perf_ranked = int(df["performance_rank"].notna().sum())
    eff_ranked = int(df["efficiency_rank"].notna().sum())
    logger.info(
        f"Scoring complete: {scored} value-ranked, {perf_ranked} perf-ranked, "
        f"{eff_ranked} efficiency-ranked, {unscored} unranked"
    )

    return df
