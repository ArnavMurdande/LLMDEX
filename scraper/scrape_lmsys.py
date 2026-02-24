"""
scrape_lmsys.py — LMSYS Chatbot Arena scraper.

SAFETY DESIGN:
    - Returns ScrapeResult, never a raw list.
    - Elo scores are preserved as raw values, NOT rescaled.
    - Missing values → None.
    - Multi-strategy extraction: Selenium → JSON API → local fallback.
    - HTTP-based fallback with timeout + retry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from scraper.contracts import ScrapedRow, ScraperHealthReport, ScrapeResult
from scraper.utils import safe_float

logger = logging.getLogger(__name__)

SOURCE_NAME = "LMSYS Chatbot Arena"
EXPECTED_ROW_RANGE = (30, 500)
REQUEST_TIMEOUT = 30


# ──────────────────────────────────────────────────────────────
# LMSYS name normalization: convert version dashes to dots
# ──────────────────────────────────────────────────────────────

# Known model family prefixes where a trailing digit-dash-digit pattern
# indicates a version number (e.g., claude-opus-4-1 → claude-opus-4.1)
# IMPORTANT: The minor version (\d) must be a SINGLE digit to avoid
# matching date codes (20250514, 0709) or parameter counts (235b).
_VERSION_DASH_PATTERNS = [
    # (regex pattern, replacement)
    # Claude: claude-opus-4-1 → claude-opus-4.1, claude-sonnet-4-5 → claude-sonnet-4.5
    # But NOT claude-opus-4-20250514 (date code)
    (re.compile(r'^(claude-(?:opus|sonnet|haiku)-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Claude alternate: claude-3-5-sonnet → claude-3.5-sonnet, claude-3-7-sonnet → claude-3.7-sonnet
    (re.compile(r'^(claude-)(\d+)-(\d)(-(?:sonnet|haiku|opus).*)$'), r'\g<1>\2.\3\4'),
    # GPT: gpt-4-1 → gpt-4.1, gpt-5-2 → gpt-5.2
    # But NOT gpt-4-turbo (non-digit after dash) or gpt-4-1-2025 (date follows)
    (re.compile(r'^(gpt-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Gemini: gemini-2-5-pro → gemini-2.5-pro, gemini-1-5-flash → gemini-1.5-flash
    (re.compile(r'^(gemini-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # DeepSeek: deepseek-v3-1 → deepseek-v3.1, deepseek-v3-2 → deepseek-v3.2
    (re.compile(r'^(deepseek-v)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Grok: grok-4-1 → grok-4.1, but NOT grok-4-0709 (date code)
    (re.compile(r'^(grok-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # GLM: glm-4-5 → glm-4.5, glm-4-6 → glm-4.6, glm-4-7 → glm-4.7
    (re.compile(r'^(glm-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Qwen: qwen3-5 → qwen3.5, but NOT qwen3-235b (parameter count)
    (re.compile(r'^(qwen)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Ernie: ernie-5-0 → ernie-5.0
    (re.compile(r'^(ernie-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Kimi: kimi-k2-5 → kimi-k2.5, but NOT kimi-k2-0905 (date code)
    (re.compile(r'^(kimi-k)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # MiniMax: minimax-m2-5 → minimax-m2.5
    (re.compile(r'^(minimax-m)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Jamba: jamba-1-5 → jamba-1.5
    (re.compile(r'^(jamba-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
    # Dola-seed: dola-seed-2-0 → dola-seed-2.0
    (re.compile(r'^(dola-seed-)(\d+)-(\d)(?!\d)(.*)$'), r'\g<1>\2.\3\4'),
]


def _normalize_lmsys_name(name: str) -> str:
    """
    Normalize an LMSYS model name by converting version dashes to dots.
    
    LMSYS uses dashes everywhere (e.g., claude-opus-4-1-20250805-thinking-16k)
    while AA uses dots for versions (e.g., Claude Opus 4.1). This function
    converts known version-dash patterns to dots for better cross-source matching.
    
    Examples:
        claude-opus-4-1-20250805-thinking-16k → claude-opus-4.1-20250805-thinking-16k
        gpt-4-1-2025-04-14 → gpt-4.1-2025-04-14
        gemini-2-5-pro → gemini-2.5-pro
        deepseek-v3-2-exp → deepseek-v3.2-exp
    """
    normalized = name.strip()
    lower = normalized.lower()
    
    for pattern, replacement in _VERSION_DASH_PATTERNS:
        match = pattern.match(lower)
        if match:
            normalized = pattern.sub(replacement, lower)
            logger.debug(f"LMSYS name normalized: '{name}' → '{normalized}'")
            break
    
    return normalized


def scrape_lmsys() -> ScrapeResult:
    start_time = time.monotonic()
    rows: List[ScrapedRow] = []
    warning_count = 0

    try:
        # ── Strategy 1: Selenium NextJS JSON extraction ──
        try:
            from scraper.utils import managed_driver, navigate_with_retry
            with managed_driver() as driver:
                navigate_with_retry(driver, "https://lmarena.ai/leaderboard/text", wait_after_load_sec=10)
                
                # Extract the __next_f data chunks
                script = '''
                    let res = "";
                    for (let item of window.__next_f || []) {
                        if (Array.isArray(item) && typeof item[1] === 'string' && 
                            (item[1].includes('"modelDisplayName":') || item[1].includes('"entries":['))) {
                            res += item[1];
                        }
                    }
                    return res;
                '''
                res = driver.execute_script(script)
                if res:
                    rows, warning_count = _parse_nextjs_data(res)
                    if rows:
                        logger.info(f"Extracted {len(rows)} models from lmarena.ai via Selenium")
        except Exception as e:
            logger.warning(f"Selenium extraction failed: {e}")

        # ── Strategy 2: Try known API endpoints ──
        if not rows:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json"
            }
            json_candidates = [
                "https://lmarena.ai/api/leaderboard",
                "https://lmarena.ai/leaderboard_data.json",
            ]
            for url in json_candidates:
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, list) and len(data) > 10:
                            rows, warning_count = _parse_api_json(data)
                            if rows:
                                logger.info(f"Extracted {len(rows)} models from API: {url}")
                                break
                except Exception:
                    continue

        # ── Strategy 3: Local fallback from text_leaderboard.json ──
        if not rows:
            local_path = os.path.join(
                os.path.dirname(__file__), "..", "text_leaderboard.json"
            )
            local_path = os.path.abspath(local_path)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        raw = f.read()
                    rows, warning_count = _parse_nextjs_data(raw)
                    if rows:
                        logger.info(f"Loaded {len(rows)} models from local text_leaderboard.json fallback")
                except Exception as e:
                    logger.warning(f"Local fallback failed: {e}")

        duration = time.monotonic() - start_time
        status = _evaluate_health(len(rows))
        health = ScraperHealthReport(
            source=SOURCE_NAME,
            rows_scraped=len(rows),
            expected_range=EXPECTED_ROW_RANGE,
            status=status,
            duration_seconds=round(duration, 2),
            parse_warning_count=warning_count,
        )

        if rows:
            logger.info(
                f"Scraped {len(rows)} models from {SOURCE_NAME} "
                f"({warning_count} warnings, status={status})"
            )
        else:
            logger.warning(f"No data extracted from {SOURCE_NAME}")

        return ScrapeResult(rows=rows, health=health)

    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"Error scraping {SOURCE_NAME}: {e}")
        return ScrapeResult(
            rows=[],
            health=ScraperHealthReport(
                source=SOURCE_NAME,
                rows_scraped=0,
                expected_range=EXPECTED_ROW_RANGE,
                status="failed",
                error_message=str(e),
                duration_seconds=round(duration, 2),
            ),
        )


# ──────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────

def _parse_nextjs_data(raw: str) -> tuple[List[ScrapedRow], int]:
    """
    Parse Next.js flight data containing LMSYS arena leaderboard entries.
    Each entry has: rank, modelDisplayName, rating, votes, modelOrganization, license.
    """
    rows: List[ScrapedRow] = []
    warning_count = 0
    
    # Unescape if needed
    if '\\"' in raw:
        raw = raw.replace('\\"', '"').replace('\\\\', '\\')

    # Extract entries using regex patterns
    # Pattern for full entry objects
    entry_pattern = re.compile(
        r'\{[^{}]*"modelDisplayName"\s*:\s*"([^"]+)"'
        r'[^{}]*"rating"\s*:\s*([\d.]+)'
        r'[^{}]*?"votes"\s*:\s*(\d+)'
        r'[^{}]*?"modelOrganization"\s*:\s*"([^"]*)"'
        r'[^{}]*?"license"\s*:\s*"([^"]*)"'
        r'[^{}]*\}',
        re.DOTALL
    )
    
    matches = entry_pattern.findall(raw)
    
    # If the full pattern fails, try a simpler pattern
    if not matches:
        # Try extracting individual entries from entries array
        simple_pattern = re.compile(
            r'"modelDisplayName"\s*:\s*"([^"]+)"'
            r'.*?"rating"\s*:\s*([\d.]+)',
            re.DOTALL
        )
        # Also try to find votes, org, license nearby
        for m in re.finditer(
            r'\{[^{]*?"modelDisplayName"\s*:\s*"([^"]+)".*?"rating"\s*:\s*([\d.]+).*?\}',
            raw, re.DOTALL
        ):
            chunk = m.group(0)
            name = m.group(1)
            rating = m.group(2)
            
            votes_m = re.search(r'"votes"\s*:\s*(\d+)', chunk)
            org_m = re.search(r'"modelOrganization"\s*:\s*"([^"]*)"', chunk)
            lic_m = re.search(r'"license"\s*:\s*"([^"]*)"', chunk)
            rank_m = re.search(r'"rank"\s*:\s*(\d+)', chunk)
            
            matches.append((
                name,
                rating,
                votes_m.group(1) if votes_m else "0",
                org_m.group(1) if org_m else "",
                lic_m.group(1) if lic_m else "",
            ))
    
    # Deduplicate by model name
    seen = set()
    for match in matches:
        name, rating, votes, org, lic = match
        if name in seen:
            continue
        seen.add(name)
        
        elo = safe_float(rating)
        vote_count = safe_float(votes)
        
        rows.append(
            ScrapedRow(
                model_name=_normalize_lmsys_name(name),
                source=SOURCE_NAME,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                arena_elo=elo,
                arena_votes=int(vote_count) if vote_count else None,
                provider=org if org else None,
                license_type=lic if lic else None,
                confidence=1.0,
                parse_warnings=[],
            )
        )
    
    return rows, warning_count


def _parse_api_json(data: list) -> tuple[List[ScrapedRow], int]:
    """Parse JSON API response from LMSYS."""
    rows: List[ScrapedRow] = []
    warning_count = 0
    
    for entry in data:
        if isinstance(entry, dict):
            name = entry.get("modelDisplayName") or entry.get("model") or entry.get("name")
            if not name:
                continue
            
            elo = safe_float(str(entry.get("rating") or entry.get("score") or entry.get("elo", "")))
            votes = entry.get("votes")
            org = entry.get("modelOrganization") or entry.get("organization")
            lic = entry.get("license")
            
            rows.append(
                ScrapedRow(
                    model_name=_normalize_lmsys_name(str(name)),
                    source=SOURCE_NAME,
                    scraped_at=datetime.now(timezone.utc).isoformat(),
                    arena_elo=elo,
                    arena_votes=int(votes) if votes else None,
                    provider=str(org) if org else None,
                    license_type=str(lic) if lic else None,
                    confidence=1.0,
                    parse_warnings=[],
                )
            )
    
    return rows, warning_count


def _evaluate_health(row_count: int) -> str:
    lo, hi = EXPECTED_ROW_RANGE
    if row_count == 0:
        return "failed"
    if lo <= row_count <= hi:
        return "healthy"
    return "degraded"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_lmsys()
    print(f"Scraped {result.health.rows_scraped} models (status={result.health.status})")
    if result.rows:
        sample = result.rows[0]
        print(f"Sample: {sample.model_name} ELO={sample.arena_elo} votes={sample.arena_votes} org={sample.provider}")
    print(result.to_json()[:2000])
