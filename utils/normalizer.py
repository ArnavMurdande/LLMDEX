"""
normalizer.py — Anchored normalization for stable scoring.

SAFETY DESIGN:
    The old code used min-max normalization which is UNSTABLE:
    every time a new model is added, ALL historical scores change.
    This makes time-series comparisons meaningless.

    NEW POLICY: Anchored normalization.
    - Each metric has a fixed anchor range (based on known real-world bounds).
    - Scores are clamped to 0–100 within these anchors.
    - Adding new data does NOT change existing normalized scores.

    Alternative: percentile-based normalization with fixed reference
    populations can also be used, but anchored is simpler, more
    transparent, and sufficient for this use case.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Anchor ranges: these define the 0→100 mapping for each metric.
# They are based on observed real-world ranges in LLM benchmarks
# and should be reviewed quarterly but NOT changed daily.
# ──────────────────────────────────────────────────────────────
ANCHOR_RANGES = {
    # Intelligence: benchmarks like MMLU report 25 (random) to ~95 (SOTA)
    "intelligence_score": (25.0, 95.0),

    # Arena Elo: typically ranges from 900 to 1550
    "arena_elo": (900.0, 1550.0),

    # Coding score: 0–100 from coding benchmarks
    "coding_score": (0.0, 100.0),

    # Reasoning score
    "reasoning_score": (0.0, 100.0),

    # Multimodal score
    "multimodal_score": (0.0, 100.0),

    # Cost: input per 1M tokens. $0 = free, $100 = expensive.
    # Higher cost → LOWER normalized score (inverted)
    "input_cost_per_1m": (0.0, 100.0),

    # Latency: 0s = instant, 10s = slow
    # Lower latency → HIGHER normalized score (inverted)
    "latency_seconds": (0.0, 10.0),

    # Speed: tokens per second
    "tokens_per_second": (0.0, 500.0),

    # GPQA Diamond: graduate-level science questions, 25 (random) to ~95 (SOTA)
    "gpqa": (25.0, 100.0),
}


def normalize_anchored(
    value: Optional[float],
    metric: str,
    invert: bool = False,
) -> Optional[float]:
    """
    Normalize a single value to 0–100 using fixed anchor ranges.

    Args:
        value: Raw metric value. None → None (not 0).
        metric: Key into ANCHOR_RANGES.
        invert: If True, higher raw values map to LOWER normalized scores.
                Used for cost and latency.

    Returns:
        Normalized score 0–100, or None if input is None.
    """
    if value is None:
        return None

    anchor = ANCHOR_RANGES.get(metric)
    if anchor is None:
        logger.warning(f"No anchor range defined for metric '{metric}'. Returning raw value.")
        return value

    lo, hi = anchor
    if hi == lo:
        return 50.0  # Avoid division by zero

    # Clamp and scale to 0–100
    normalized = 100.0 * (float(value) - lo) / (hi - lo)
    normalized = max(0.0, min(100.0, normalized))

    if invert:
        normalized = 100.0 - normalized

    return round(normalized, 2)


def normalize_series_anchored(
    series: pd.Series,
    metric: str,
    invert: bool = False,
) -> pd.Series:
    """
    Vectorised anchored normalization for a pandas Series.
    None/NaN inputs produce NaN outputs (NOT 0).
    """
    return series.apply(lambda v: normalize_anchored(v, metric, invert))


def safe_convert(val, type_func, default=None):
    """Safely convert a value with a fallback. None in → default out."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return type_func(val)
    except (ValueError, TypeError):
        return default
