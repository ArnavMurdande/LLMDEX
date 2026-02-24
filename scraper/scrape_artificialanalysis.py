"""
scrape_artificialanalysis.py — Artificial Analysis leaderboard scraper.

SAFETY:
    - Returns ScrapeResult with health report, NEVER a raw list.
    - Missing values → None. We never fabricate a 0.
    - Health report validates row count is within expected range.
    
EXTRACTION STRATEGY:
    1. Primary: Extract from the large RSC payload script tag that contains
       the full model objects with all benchmark/speed/price data.
       Key: self.__next_f.push([1,"28:...{\"models\":[{...}]}"])
    2. Fallback: DOM table scraping from the visible HTML table.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from scraper.contracts import ScrapedRow, ScraperHealthReport, ScrapeResult
from scraper.utils import (
    managed_driver,
    navigate_with_retry,
    scroll_to_bottom,
    safe_float,
    safe_int,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "Artificial Analysis"
URL = "https://artificialanalysis.ai/leaderboards/models?deprecation=all"
EXPECTED_ROW_RANGE = (20, 500)


def scrape_artificialanalysis() -> ScrapeResult:
    """
    Scrape the Artificial Analysis leaderboard.
    """
    start_time = time.monotonic()
    rows: List[ScrapedRow] = []
    warning_count = 0

    try:
        with managed_driver() as driver:
            navigate_with_retry(driver, URL, wait_after_load_sec=8)
            time.sleep(4)  # Let RSC data fully load (the big payload is ~4.7MB)

            # ── Strategy 1: Extract full model data from RSC payload ──
            models = _extract_rsc_models(driver)
            
            # RSC can be flaky — retry once after a short wait
            if not models:
                logger.info("RSC first attempt failed, retrying after 3s...")
                time.sleep(3)
                models = _extract_rsc_models(driver)
            
            if not models:
                # ── Strategy 2: DOM table scraping ──
                logger.info("RSC extraction failed, falling back to DOM scraping")
                scroll_to_bottom(driver, max_scrolls=10, pause=1.0)
                time.sleep(2)
                models = _extract_dom_table(driver)

            if not models:
                raise RuntimeError("Could not extract any model data from Artificial Analysis")

            logger.info(f"Extracted {len(models)} raw model entries from AA")

            # If we got RSC models, enrich them with table price/speed/latency
            if models and any("intelligence_index" in m for m in models[:5]):
                scroll_to_bottom(driver, max_scrolls=10, pause=0.5)
                time.sleep(1)
                models = _enrich_with_table_data(driver, models)

            # Convert extracted models to ScrapedRow objects
            seen_names = set()
            for m in models:
                try:
                    row = _model_to_scraped_row(m)
                    if row and row.model_name not in seen_names:
                        seen_names.add(row.model_name)
                        rows.append(row)
                except Exception as e:
                    warning_count += 1
                    logger.debug(f"Skipped model entry: {e}")

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
        logger.info(
            f"Scraped {len(rows)} models from {SOURCE_NAME} "
            f"({warning_count} warnings, status={status})"
        )
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
# Strategy 1: RSC payload extraction
# ──────────────────────────────────────────────────────────────

def _extract_rsc_models(driver) -> Optional[List[dict]]:
    """
    Extract model data from the React Server Component payload.
    
    AA stores the FULL model data in a large (~4.7MB) script tag that 
    pushes data via self.__next_f.push([1, "28:[...{models:[{...}]}]"]).
    
    The key: the big chunk contains objects with ACTUAL data keys like:
    - intelligence_index, estimated_intelligence_index
    - context_window_tokens, livecodebench, gpqa, hle, aime25, etc.
    - But pricing/speed/latency are NOT in this array — they're in the table.
    """
    try:
        # Find the script tag with the large data payload containing model objects
        script = """
            // Concatenate ALL __next_f data 
            let allData = "";
            for (let item of window.__next_f || []) {
                if (Array.isArray(item) && typeof item[1] === 'string') {
                    allData += item[1];
                }
            }
            
            // Find ALL occurrences of "models":[{ and check which has benchmark data
            let searchStr = '"models":[{';
            let candidates = [];
            let searchFrom = 0;
            
            while (searchFrom < allData.length) {
                let idx = allData.indexOf(searchStr, searchFrom);
                if (idx === -1) break;
                
                // Check if this occurrence has benchmark keys nearby
                let preview = allData.substring(idx, Math.min(allData.length, idx + 2000));
                let hasBenchmarks = preview.includes('intelligence_index') || 
                                   preview.includes('aime25') || 
                                   preview.includes('gpqa') ||
                                   preview.includes('coding_index') ||
                                   preview.includes('context_window_tokens');
                
                candidates.push({idx: idx, hasBenchmarks: hasBenchmarks});
                searchFrom = idx + searchStr.length;
            }
            
            // Pick the candidate with benchmark data, preferring the one with benchmarks
            let bestStart = -1;
            for (let c of candidates) {
                if (c.hasBenchmarks) {
                    bestStart = c.idx;
                    break;
                }
            }
            // If none had benchmarks, try the last one (often the bigger array)
            if (bestStart === -1 && candidates.length > 0) {
                bestStart = candidates[candidates.length - 1].idx;
            }
            
            if (bestStart === -1) return null;
            
            // Find the start of the array
            let arrStart = allData.indexOf('[', bestStart + '"models":'.length - 1);
            if (arrStart === -1) return null;
            
            // Extract the JSON array using bracket matching
            let depth = 0;
            let inStr = false;
            let escape = false;
            
            for (let i = arrStart; i < Math.min(allData.length, arrStart + 6000000); i++) {
                let c = allData[i];
                if (escape) { escape = false; continue; }
                if (c === '\\\\') { escape = true; continue; }
                if (c === '"') { inStr = !inStr; continue; }
                if (!inStr) {
                    if (c === '[') depth++;
                    if (c === ']') {
                        depth--;
                        if (depth === 0) {
                            return allData.substring(arrStart, i + 1);
                        }
                    }
                }
            }
            return null;
        """
        raw = driver.execute_script(script)
        
        if not raw:
            logger.warning("RSC extraction: no models array found")
            return None
        
        logger.info(f"RSC extraction: found models array ({len(raw)} chars)")
        
        # Unescape the JSON if needed
        if '\\"' in raw:
            raw = raw.replace('\\"', '"')
        if '\\\\' in raw:
            raw = raw.replace('\\\\', '\\')
        
        try:
            models = json.loads(raw)
        except json.JSONDecodeError:
            # Try cleaning common RSC artifacts
            raw = re.sub(r'\$[A-Za-z0-9]+', '""', raw)  # Replace React refs
            try:
                models = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed for RSC models: {e}")
                return None
        
        if isinstance(models, list) and len(models) > 10:
            # Verify these have actual data (not just metadata)
            sample = models[0]
            has_data = any(k in sample for k in [
                'intelligence_index', 'estimated_intelligence_index',
                'aime25', 'gpqa', 'hle', 'livecodebench', 'context_window_tokens'
            ])
            if has_data:
                logger.info(f"RSC extraction: found {len(models)} models with benchmark data")
                return models
            else:
                logger.warning(f"RSC extraction: found {len(models)} models but they lack benchmark keys")
                logger.debug(f"Sample keys: {list(sample.keys())[:20]}")
                return None
        
        return None
        
    except Exception as e:
        logger.warning(f"RSC extraction failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Strategy 2: DOM table scraping
# ──────────────────────────────────────────────────────────────

def _extract_dom_table(driver) -> Optional[List[dict]]:
    """
    Scrape the visible HTML table on the AA leaderboard page.
    Table columns: Model, ContextWindow, Creator, Intelligence Index,
    Blended USD/1M Tokens, Median Tokens/s, Latency First Answer Chunk
    """
    models = []
    
    try:
        from selenium.webdriver.common.by import By
        
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        
        if not rows:
            logger.warning("No table rows found in DOM")
            return None
        
        logger.info(f"DOM scraping: found {len(rows)} table rows")
        
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, "td")
            if len(cells) < 5:
                continue
            
            # Table columns based on our debug output:
            # Cell 0: Model name
            # Cell 1: Context window (e.g., "1m", "200k")
            # Cell 2: Creator
            # Cell 3: Intelligence Index
            # Cell 4: Blended USD/1M Tokens  (e.g., "$4.50")
            # Cell 5: Median Tokens/s
            # Cell 6: Latency First Answer Chunk (s)
            
            cell_texts = [c.text.strip() for c in cells]
            
            model_name = cell_texts[0] if len(cell_texts) > 0 else ""
            if not model_name or model_name.lower() in ("model", "name", ""):
                continue
            
            entry = {"name": model_name}
            
            if len(cell_texts) > 1:
                entry["context_window_tokens"] = _parse_context_window(cell_texts[1])
            if len(cell_texts) > 2:
                entry["creator_name"] = cell_texts[2]
            if len(cell_texts) > 3:
                entry["intelligence_index"] = safe_float(cell_texts[3])
            if len(cell_texts) > 4:
                price_str = cell_texts[4].replace("$", "").replace(",", "").strip()
                entry["blended_price_1m"] = safe_float(price_str)
            if len(cell_texts) > 5:
                entry["median_tokens_per_second"] = safe_float(cell_texts[5])
            if len(cell_texts) > 6:
                entry["latency_first_answer_chunk"] = safe_float(cell_texts[6])
            
            models.append(entry)
        
        return models if models else None
        
    except Exception as e:
        logger.warning(f"DOM table extraction failed: {e}")
        return None


def _parse_context_window(text: str) -> Optional[int]:
    """Parse context window strings like '1m', '200k', '128k'."""
    if not text:
        return None
    text = text.lower().strip().replace(",", "")
    try:
        if "m" in text:
            return int(float(text.replace("m", "")) * 1_000_000)
        elif "k" in text:
            return int(float(text.replace("k", "")) * 1_000)
        else:
            val = safe_float(text)
            return int(val) if val else None
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────
# Convert model dict to ScrapedRow
# ──────────────────────────────────────────────────────────────

def _model_to_scraped_row(m: dict) -> Optional[ScrapedRow]:
    """
    Convert a raw model dict (from RSC or DOM) to a ScrapedRow.
    
    RSC keys (from AA's actual data):
        intelligence_index, estimated_intelligence_index, coding_index,
        context_window_tokens, aime25, gpqa, hle, livecodebench,
        mmlu_pro, mmmu_pro, scicode, ifbench, critpt, gdpval,
        lcr, license_name, model_creator_id, math_index, agentic_index,
        tau2, terminalbench_hard, commercial_allowed, frontier_model,
        omniscience_accuracy, non_hallucination_rate
    
    DOM keys (from table):
        name, context_window_tokens, creator_name, intelligence_index,
        blended_price_1m, median_tokens_per_second, latency_first_answer_chunk
    """
    # Get model name
    name = m.get("name") or m.get("shortName")
    if not name:
        return None
    
    # Note: deprecated/deleted models are now included (deprecation=all)
    
    warnings = []
    
    # Intelligence score - AA uses both "intelligence_index" and "estimated_intelligence_index"
    intelligence = safe_float(m.get("intelligence_index"))
    estimated_intelligence = safe_float(m.get("estimated_intelligence_index"))
    
    # Benchmark scores (all 0-1 scale from AA, convert to percentage)
    gpqa = safe_float(m.get("gpqa"))
    if gpqa and gpqa <= 1.0:
        gpqa = round(gpqa * 100, 1)
    
    aime25 = safe_float(m.get("aime25"))
    if aime25 and aime25 <= 1.0:
        aime25 = round(aime25 * 100, 1)
    
    hle = safe_float(m.get("hle"))
    if hle and hle <= 1.0:
        hle = round(hle * 100, 1)
    
    livecodebench = safe_float(m.get("livecodebench"))
    if livecodebench and livecodebench <= 1.0:
        livecodebench = round(livecodebench * 100, 1)
    
    mmlu_pro = safe_float(m.get("mmlu_pro"))
    if mmlu_pro and mmlu_pro <= 1.0:
        mmlu_pro = round(mmlu_pro * 100, 1)
    
    mmmu_pro = safe_float(m.get("mmmu_pro"))
    if mmmu_pro and mmmu_pro <= 1.0:
        mmmu_pro = round(mmmu_pro * 100, 1)
    
    scicode = safe_float(m.get("scicode"))
    if scicode and scicode <= 1.0:
        scicode = round(scicode * 100, 1)
    
    ifbench = safe_float(m.get("ifbench"))
    if ifbench and ifbench <= 1.0:
        ifbench = round(ifbench * 100, 1)
    
    critpt = safe_float(m.get("critpt"))
    if critpt and critpt <= 1.0:
        critpt = round(critpt * 100, 1)
    
    lcr = safe_float(m.get("lcr"))
    if lcr and lcr <= 1.0:
        lcr = round(lcr * 100, 1)
    
    # Other indices
    coding_score = safe_float(m.get("coding_index"))
    gdpval = safe_float(m.get("gdpval"))
    tau2 = safe_float(m.get("tau2"))
    terminalbench = safe_float(m.get("terminalbench_hard"))
    omniscience = safe_float(m.get("omniscience_accuracy"))
    omniscience_hallucination = safe_float(m.get("non_hallucination_rate"))
    
    # Context window
    ctx = safe_int(m.get("context_window_tokens"))
    
    # Economics (from DOM table scraping or from pricing data)
    blended_cost = safe_float(m.get("blended_price_1m"))
    input_cost = safe_float(m.get("input_cost_per_1m"))
    output_cost = safe_float(m.get("output_cost_per_1m"))
    
    # Speed
    speed = safe_float(m.get("median_tokens_per_second"))
    
    # Latency
    latency = safe_float(m.get("latency_first_answer_chunk"))
    
    # Creator
    creator_obj = m.get("creator")
    if isinstance(creator_obj, dict):
        creator = creator_obj.get("name")
    else:
        creator = m.get("creator_name") or (creator_obj if isinstance(creator_obj, str) else None)
    
    # License
    license_name = m.get("license_name")
    
    # Open source status
    open_source = m.get("commercial_allowed")
    
    # Count non-None fields for confidence
    data_fields = [intelligence, gpqa, aime25, hle, livecodebench, coding_score,
                   blended_cost, speed, latency, ctx]
    filled = sum(1 for f in data_fields if f is not None)
    confidence = max(0.3, min(1.0, 0.3 + 0.07 * filled))
    
    return ScrapedRow(
        model_name=str(name),
        source=SOURCE_NAME,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        intelligence_score=intelligence or estimated_intelligence,
        coding_score=coding_score,
        livecodebench=livecodebench,
        gpqa=gpqa,
        hle=hle,
        aime25=aime25,
        mmmu_pro=mmmu_pro,
        scicode=scicode,
        ifbench=ifbench,
        critpt=critpt,
        gdpval=gdpval,
        tau2=tau2,
        terminalbench_hard=terminalbench,
        lcr=lcr,
        omniscience=omniscience,
        omniscience_hallucination=omniscience_hallucination,
        blended_cost_per_1m=blended_cost,
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
        tokens_per_second=speed,
        latency_seconds=latency,
        context_window=ctx,
        creator=str(creator) if creator else None,
        license_type=str(license_name) if license_name else None,
        open_source=open_source,
        confidence=confidence,
        parse_warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────
# Health & enrichment
# ──────────────────────────────────────────────────────────────

def _evaluate_health(row_count: int) -> str:
    lo, hi = EXPECTED_ROW_RANGE
    if row_count == 0:
        return "failed"
    if lo <= row_count <= hi:
        return "healthy"
    return "degraded"


def _enrich_with_table_data(driver, models: List[dict]) -> List[dict]:
    """
    Enrich RSC-extracted models with pricing/speed/latency from the visible table.
    The RSC payload has benchmark scores but NOT pricing/speed/latency.
    The DOM table has pricing/speed/latency but limited benchmarks.
    """
    from selenium.webdriver.common.by import By
    
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        table_data = {}
        
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, "td")
            if len(cells) < 5:
                continue
            
            cell_texts = [c.text.strip() for c in cells]
            name = cell_texts[0]
            if not name:
                continue
            
            entry = {}
            if len(cell_texts) > 4:
                price_str = cell_texts[4].replace("$", "").replace(",", "").strip()
                entry["blended_price_1m"] = safe_float(price_str)
            if len(cell_texts) > 5:
                entry["median_tokens_per_second"] = safe_float(cell_texts[5])
            if len(cell_texts) > 6:
                entry["latency_first_answer_chunk"] = safe_float(cell_texts[6])
            
            table_data[name.lower()] = entry
        
        # Match and enrich
        enriched = 0
        for model in models:
            name = (model.get("name") or model.get("shortName") or "").lower()
            short = (model.get("shortName") or "").lower()
            
            match = table_data.get(name) or table_data.get(short)
            if match:
                for k, v in match.items():
                    if v is not None and model.get(k) is None:
                        model[k] = v
                        enriched += 1
        
        logger.info(f"Enriched {enriched} fields from table data")
        
    except Exception as e:
        logger.warning(f"Table enrichment failed: {e}")
    
    return models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_artificialanalysis()
    print(f"Scraped {result.health.rows_scraped} models (status={result.health.status})")
    if result.rows:
        # Show a sample with actual data
        for r in result.rows[:5]:
            fields = {k: v for k, v in r.to_dict().items() 
                     if v is not None and v != [] and v != ""}
            print(f"\n{r.model_name}:")
            for k, v in fields.items():
                if k not in ("model_name", "source", "scraped_at", "parse_warnings"):
                    print(f"  {k}: {v}")
