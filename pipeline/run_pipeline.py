"""
run_pipeline.py — Main pipeline orchestrator.

SAFETY DESIGN:
    This is the top-level entry point that enforces the full
    data safety pipeline:

    SCRAPE → RAW SNAPSHOT → VALIDATE → NORMALIZE → SCORE → INDEX → PUBLISH

    Critical safety rules:
    1. If ANY scraper health report is "failed", we log it but continue
       with other sources. However, if ALL scrapers fail, we abort.
    2. If validation fails (coverage too low, integrity violations),
       the pipeline ABORTS. Nothing is published.
    3. Each stage writes its own artifact. We never overwrite upstream.
    4. Exit code is non-zero on failure (for CI integration).
    5. A pipeline summary report is written for CI artifact capture.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

# ── Project root on path ──
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logging, Logger
from scraper.contracts import ScrapeResult
from scraper import scrape_artificialanalysis
from scraper import scrape_lmsys
from pipeline.validator import validate_dataset
from pipeline.merge_data import process_and_save


# Minimum number of healthy scrapers required to proceed
MIN_HEALTHY_SOURCES = 1


def run_pipeline() -> bool:
    """
    Run the full pipeline. Returns True on success, False on failure.
    """
    setup_logging()
    log = Logger(name="Pipeline")
    log.info("═══════════════════════════════════════════════════")
    log.info("  LLM Intelligence Index — Pipeline Start")
    log.info("═══════════════════════════════════════════════════")

    pipeline_start = time.monotonic()
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Stage 1: SCRAPE ──
    log.info("Stage 1: SCRAPE — collecting data from sources")
    scrape_tasks = [
        ("Artificial Analysis", scrape_artificialanalysis.scrape_artificialanalysis),
        ("LMSYS", scrape_lmsys.scrape_lmsys),
    ]

    all_rows: List[dict] = []
    health_reports: List[dict] = []
    healthy_sources = 0
    failed_sources = 0

    for name, scrape_fn in scrape_tasks:
        log.info(f"  → Scraping {name}...")
        try:
            result: ScrapeResult = scrape_fn()
            health_reports.append(result.health.to_dict())

            if result.health.is_healthy() or result.health.status == "degraded":
                rows_dicts = [r.to_dict() for r in result.rows]
                all_rows.extend(rows_dicts)
                healthy_sources += 1
                log.info(
                    f"  ✓ {name}: {result.health.rows_scraped} rows "
                    f"(status={result.health.status})"
                )
            else:
                failed_sources += 1
                log.warning(
                    f"  ✗ {name}: FAILED — {result.health.error_message}"
                )
        except Exception as e:
            failed_sources += 1
            log.error(f"  ✗ {name}: EXCEPTION — {e}")
            health_reports.append({
                "source": name,
                "status": "failed",
                "error_message": str(e),
                "rows_scraped": 0,
            })

    log.info(
        f"Scrape complete: {healthy_sources} healthy, "
        f"{failed_sources} failed, {len(all_rows)} total rows"
    )

    # ── Check minimum source threshold ──
    if healthy_sources < MIN_HEALTHY_SOURCES:
        log.error(
            f"PIPELINE ABORT: Only {healthy_sources} healthy sources "
            f"(minimum: {MIN_HEALTHY_SOURCES})"
        )
        _write_summary(health_reports, [], False, "Insufficient healthy sources", snapshot_date)
        return False

    if not all_rows:
        log.error("PIPELINE ABORT: No data collected from any source.")
        _write_summary(health_reports, [], False, "No data collected", snapshot_date)
        return False

    # ── Stage 2: VALIDATE ──
    log.info(f"Stage 2: VALIDATE — checking {len(all_rows)} rows")
    valid_rows, validation_report = validate_dataset(all_rows)

    if not validation_report.passed:
        log.error(
            f"PIPELINE ABORT: Validation failed — "
            f"{validation_report.failure_reason}"
        )
        _write_summary(
            health_reports,
            [validation_report.to_dict()],
            False,
            validation_report.failure_reason,
            snapshot_date,
        )
        return False

    log.info(
        f"Validation passed: {validation_report.valid_rows}/{validation_report.total_rows} "
        f"rows valid ({validation_report.coverage_pct:.1%})"
    )
    if validation_report.rejected_rows > 0:
        log.info(f"  Rejected {validation_report.rejected_rows} rows:")
        for reason, count in validation_report.rejection_reasons.items():
            log.info(f"    • {reason}: {count}")

    # ── Stage 3-5: MERGE → SCORE → INDEX ──
    log.info("Stage 3-5: MERGE → SCORE → INDEX")
    result_df = process_and_save(valid_rows, snapshot_date)

    if result_df is None:
        log.error("PIPELINE ABORT: Processing failed.")
        _write_summary(
            health_reports,
            [validation_report.to_dict()],
            False,
            "Processing failed",
            snapshot_date,
        )
        return False

    # ── Summary ──
    duration = time.monotonic() - pipeline_start
    log.info("═══════════════════════════════════════════════════")
    log.info(f"  Pipeline SUCCEEDED in {duration:.1f}s")
    log.info(f"  Models: {len(result_df)}")
    log.info(f"  Ranked: {int(result_df['composite_index'].notna().sum())}")
    log.info(f"  Snapshot: {snapshot_date}")
    log.info("═══════════════════════════════════════════════════")

    _write_summary(
        health_reports,
        [validation_report.to_dict()],
        True,
        None,
        snapshot_date,
    )
    return True


def run_sentiment_postprocessing() -> bool:
    """
    Run the sentiment pipeline as a POST-PROCESSING step.

    SAFETY:
      - This is fully decoupled from scoring. Sentiment NEVER influences rankings.
      - Failures here are logged but do NOT affect the main pipeline's success status.
      - Uses Gemini for semantic classification with VADER as fallback.
    """
    log = Logger(name="Pipeline")
    log.info("═══════════════════════════════════════════════════")
    log.info("  Post-Processing: Community Sentiment Analysis")
    log.info("═══════════════════════════════════════════════════")

    try:
        from pipeline.sentiment_pipeline import run_sentiment_pipeline
        result = run_sentiment_pipeline()
        if result:
            log.info("  ✓ Sentiment pipeline completed successfully")
            return True
        else:
            log.warning("  ⚠ Sentiment pipeline returned no results")
            return False
    except ImportError as e:
        log.warning(f"  ⚠ Sentiment pipeline not available: {e}")
        return False
    except Exception as e:
        log.error(f"  ✗ Sentiment pipeline failed: {e}")
        log.info("  → Main pipeline results are unaffected.")
        return False


def _write_summary(
    health_reports: List[dict],
    validation_reports: List[dict],
    success: bool,
    failure_reason: str | None,
    snapshot_date: str,
) -> None:
    """
    Write a pipeline summary JSON for CI artifact capture.
    This replaces the old pipeline.log that was committed to git.
    """
    summary = {
        "snapshot_date": snapshot_date,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "failure_reason": failure_reason,
        "scraper_health": health_reports,
        "validation": validation_reports,
    }

    summary_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "pipeline_reports"
    )
    os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, f"{snapshot_date}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also write latest for easy CI access
    latest_path = os.path.join(summary_dir, "latest_summary.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLMDEX Pipeline")
    parser.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Run community sentiment analysis after the main pipeline (optional, does not affect rankings)",
    )
    args = parser.parse_args()

    success = run_pipeline()

    if success and args.with_sentiment:
        run_sentiment_postprocessing()

    sys.exit(0 if success else 1)
