"""
contracts.py — Data contracts for the scraper layer.

SAFETY DESIGN:
    - All scraper outputs MUST use these dataclasses.
    - Missing values are ALWAYS None, NEVER 0 or fabricated.
    - Each row carries its own provenance and confidence metadata.
    - Health reports enforce row-count sanity checks per source.

These contracts are the ONLY interface between scrapers and the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional


@dataclass(frozen=True)
class ScrapedRow:
    """
    A single model observation from one source.

    RULES:
        - model_name must be a non-empty string (validated downstream).
        - All numeric fields are Optional[float]. None means "not available".
        - NEVER substitute 0 for a missing value. 0 means the source
          explicitly reported zero.
        - source must identify the upstream benchmark site.
        - scraped_at is UTC ISO-8601 timestamp of when the row was captured.
        - confidence is a 0.0–1.0 float indicating parse quality.
          1.0 = all fields parsed cleanly.
          Lower values flag partial parses.
        - parse_warnings collects per-field issues for audit.
    """

    model_name: str
    source: str
    scraped_at: str  # ISO-8601 UTC

    # --- Benchmark scores (all optional) ---
    intelligence_score: Optional[float] = None
    coding_score: Optional[float] = None
    reasoning_score: Optional[float] = None
    multimodal_score: Optional[float] = None
    arena_elo: Optional[float] = None
    
    # --- Extended / Agentic / Reasoning Metrics ---
    gdpval: Optional[float] = None
    terminalbench_hard: Optional[float] = None
    tau2: Optional[float] = None
    lcr: Optional[float] = None
    omniscience: Optional[float] = None
    omniscience_hallucination: Optional[float] = None  # Non-hallucination rate
    hle: Optional[float] = None
    gpqa: Optional[float] = None
    scicode: Optional[float] = None
    ifbench: Optional[float] = None
    aime25: Optional[float] = None
    critpt: Optional[float] = None
    mmmu_pro: Optional[float] = None
    livecodebench: Optional[float] = None  # LiveCodeBench coding score

    # --- Economics (all optional) ---
    input_cost_per_1m: Optional[float] = None
    output_cost_per_1m: Optional[float] = None
    blended_cost_per_1m: Optional[float] = None  # Blended USD/1M tokens

    # --- Speed metrics (all optional) ---
    tokens_per_second: Optional[float] = None      # Median tokens/s
    speed_p5: Optional[float] = None               # P5 tokens/s
    speed_p25: Optional[float] = None              # P25 tokens/s
    speed_p75: Optional[float] = None              # P75 tokens/s
    speed_p95: Optional[float] = None              # P95 tokens/s

    # --- Latency metrics (all optional) ---
    latency_seconds: Optional[float] = None         # First answer chunk latency
    latency_first_token: Optional[float] = None     # First answer token (s)
    latency_p5: Optional[float] = None              # P5 first chunk (s)
    latency_p25: Optional[float] = None             # P25 first chunk (s)
    latency_p75: Optional[float] = None             # P75 first chunk (s)
    latency_p95: Optional[float] = None             # P95 first chunk (s)
    total_response_time: Optional[float] = None     # Total response (s)
    reasoning_time: Optional[float] = None          # Reasoning time (s)

    # --- Infrastructure (all optional) ---
    context_window: Optional[int] = None

    # --- Metadata ---
    provider: Optional[str] = None
    creator: Optional[str] = None
    open_source: Optional[bool] = None
    license_type: Optional[str] = None

    # --- Arena-specific ---
    arena_votes: Optional[int] = None

    # --- Quality tracking ---
    confidence: float = 1.0
    parse_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary. None values are preserved."""
        d = asdict(self)
        # parse_warnings is already a list; ensure it serialises cleanly.
        return d


@dataclass
class ScraperHealthReport:
    """
    Every scraper run MUST produce one of these.

    The pipeline uses it to decide whether to trust the output.
    If status != "healthy" the pipeline MUST NOT use the rows.
    """

    source: str
    rows_scraped: int
    expected_range: tuple  # (min_expected, max_expected)
    status: str  # "healthy" | "degraded" | "failed"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parse_warning_count: int = 0

    def is_healthy(self) -> bool:
        return self.status == "healthy"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expected_range"] = list(d["expected_range"])
        return d


@dataclass
class ScrapeResult:
    """
    The top-level return type for every scraper function.

    A scraper returns EITHER:
        - rows + a healthy report, OR
        - an empty rows list + a failed/degraded report.

    The pipeline consumes ScrapeResult objects, NEVER raw lists.
    """

    rows: List[ScrapedRow]
    health: ScraperHealthReport

    def to_dict(self) -> dict:
        return {
            "health": self.health.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
