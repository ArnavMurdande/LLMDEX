"""
validator.py — Validation firewall between scraping and publishing.

SAFETY DESIGN:
    This module is the SINGLE POINT of truth for data quality.
    If validation fails, the pipeline ABORTS. Nothing gets published.

    Rules enforced:
    1. Model names must be non-empty, non-numeric strings.
    2. Numeric-only model slugs are rejected (likely parse garbage).
    3. Each row must have at least one benchmark score.
    4. Numeric values must be within sane ranges (no 100000% scores).
    5. Cost cannot be negative or impossibly high.
    6. Outlier detection via IQR method.
    7. Minimum dataset coverage threshold.

    Rejected rows are logged with reasons for audit trail.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Validation rule configuration
# ──────────────────────────────────────────────────────────────

# Sanity ranges: (min, max) for numeric fields. None means "no bound".
SANITY_RANGES: Dict[str, tuple] = {
    "intelligence_score": (0, 100),
    "coding_score": (0, 100),
    "reasoning_score": (0, 100),
    "multimodal_score": (0, 100),
    "arena_elo": (500, 2500),
    "input_cost_per_1m": (0, 10_000),     # $10k/1M tokens is extreme but possible
    "output_cost_per_1m": (0, 50_000),
    "context_window": (256, 50_000_000),   # 256 to 50M tokens
    "latency_seconds": (0, 300),           # Up to 5 minutes
    "tokens_per_second": (0, 100_000),
}

# Minimum rows required for a valid dataset
MIN_DATASET_ROWS = 5

# Minimum percentage of rows that must pass validation
MIN_COVERAGE_PCT = 0.10  # At least 10% of scraped rows must survive

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 3.0


@dataclass
class ValidationResult:
    """Result of validation for a single row."""
    valid: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    """
    Summary of validation across the entire dataset.
    If .passed is False, the pipeline MUST abort.
    """
    total_rows: int
    valid_rows: int
    rejected_rows: int
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    outlier_flags: Dict[str, int] = field(default_factory=dict)
    passed: bool = True
    failure_reason: Optional[str] = None

    @property
    def coverage_pct(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.valid_rows / self.total_rows

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "rejected_rows": self.rejected_rows,
            "coverage_pct": round(self.coverage_pct, 4),
            "rejection_reasons": self.rejection_reasons,
            "outlier_flags": self.outlier_flags,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


def validate_row(row: dict) -> ValidationResult:
    """
    Validate a single data row against all rules.
    Returns a ValidationResult with pass/fail and reasons.
    """
    reasons: List[str] = []

    # ── Rule 1: Model name must be non-empty ──
    name = row.get("model_name", "")
    if not name or not str(name).strip():
        reasons.append("Empty model name")
        return ValidationResult(valid=False, reasons=reasons)

    name = str(name).strip()

    # ── Rule 2: Reject numeric-only model slugs ──
    # These are almost always parse artifacts.
    cleaned = re.sub(r"[\s\-_.]", "", name)
    if cleaned.isdigit():
        reasons.append(f"Numeric-only model slug: '{name}'")
        return ValidationResult(valid=False, reasons=reasons)

    # Very short names are suspicious
    if len(name) < 2:
        reasons.append(f"Model name too short: '{name}'")
        return ValidationResult(valid=False, reasons=reasons)

    # ── Rule 3: Must have at least one score ──
    score_fields = [
        "intelligence_score", "coding_score", "reasoning_score",
        "multimodal_score", "arena_elo",
        # Extended benchmark scores from Artificial Analysis
        "gpqa", "aime25", "hle", "livecodebench", "mmmu_pro",
        "scicode", "ifbench", "critpt", "gdpval", "omniscience",
        "terminalbench_hard", "tau2", "lcr",
    ]
    has_score = any(
        row.get(f) is not None for f in score_fields
    )
    # Also check cost/speed/latency fields — rows with these are still useful
    cost_fields = [
        "input_cost_per_1m", "output_cost_per_1m", "blended_cost_per_1m",
        "tokens_per_second", "latency_seconds",
    ]
    has_cost = any(row.get(f) is not None for f in cost_fields)

    if not has_score and not has_cost:
        reasons.append("No benchmark scores or cost data")
        return ValidationResult(valid=False, reasons=reasons)

    # ── Rule 4: Numeric sanity ranges ──
    for field_name, (lo, hi) in SANITY_RANGES.items():
        val = row.get(field_name)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            reasons.append(f"{field_name}: non-numeric value '{val}'")
            continue

        if lo is not None and val < lo:
            reasons.append(f"{field_name}={val} below minimum {lo}")
        if hi is not None and val > hi:
            reasons.append(f"{field_name}={val} above maximum {hi}")

    # ── Rule 5: Impossible costs ──
    input_cost = row.get("input_cost_per_1m")
    output_cost = row.get("output_cost_per_1m")
    if input_cost is not None and output_cost is not None:
        try:
            if float(input_cost) < 0 or float(output_cost) < 0:
                reasons.append("Negative cost detected")
        except (ValueError, TypeError):
            pass

    if reasons:
        return ValidationResult(valid=False, reasons=reasons)

    return ValidationResult(valid=True)


def validate_dataset(
    rows: List[dict],
    source_health_reports: Optional[List[dict]] = None,
) -> tuple[List[dict], DatasetValidationReport]:
    """
    Validate an entire dataset. Returns (valid_rows, report).

    If the report.passed is False, the pipeline MUST abort.
    """
    if not rows:
        return [], DatasetValidationReport(
            total_rows=0,
            valid_rows=0,
            rejected_rows=0,
            passed=False,
            failure_reason="Empty dataset — nothing to validate",
        )

    valid_rows: List[dict] = []
    rejection_counts: Dict[str, int] = {}
    rejected_count = 0

    for row in rows:
        result = validate_row(row)
        if result.valid:
            valid_rows.append(row)
        else:
            rejected_count += 1
            for reason in result.reasons:
                # Group by reason category
                key = reason.split(":")[0].strip() if ":" in reason else reason
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
            logger.debug(
                f"Row rejected [{row.get('model_name', '?')}]: {result.reasons}"
            )

    # ── Outlier detection (IQR method) ──
    outlier_flags: Dict[str, int] = {}
    numeric_fields = [
        "intelligence_score", "arena_elo",
        "input_cost_per_1m", "output_cost_per_1m",
    ]
    for field_name in numeric_fields:
        values = [
            float(r[field_name])
            for r in valid_rows
            if r.get(field_name) is not None
        ]
        if len(values) < 5:
            continue

        arr = np.array(values)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr

        count = int(np.sum((arr < lower) | (arr > upper)))
        if count > 0:
            outlier_flags[field_name] = count
            logger.warning(
                f"Outlier detection: {count} outliers in {field_name} "
                f"(IQR: {q1:.2f}–{q3:.2f}, bounds: {lower:.2f}–{upper:.2f})"
            )

    # ── Build report ──
    report = DatasetValidationReport(
        total_rows=len(rows),
        valid_rows=len(valid_rows),
        rejected_rows=rejected_count,
        rejection_reasons=rejection_counts,
        outlier_flags=outlier_flags,
    )

    # ── Minimum dataset coverage check ──
    if len(valid_rows) < MIN_DATASET_ROWS:
        report.passed = False
        report.failure_reason = (
            f"Only {len(valid_rows)} valid rows — "
            f"below minimum threshold of {MIN_DATASET_ROWS}"
        )
    elif report.coverage_pct < MIN_COVERAGE_PCT:
        report.passed = False
        report.failure_reason = (
            f"Coverage {report.coverage_pct:.1%} — "
            f"below minimum threshold of {MIN_COVERAGE_PCT:.0%}"
        )

    if report.passed:
        logger.info(
            f"Validation PASSED: {report.valid_rows}/{report.total_rows} rows valid "
            f"({report.coverage_pct:.1%} coverage)"
        )
    else:
        logger.error(f"Validation FAILED: {report.failure_reason}")

    return valid_rows, report
