"""Artificial Analysis model leaderboard scraper.

Artificial Analysis is the only benchmark source used by LLMDEX. The scraper
opens the exact public leaderboard requested by the project, switches the table
to its expanded view, and captures every visible column for every current model.

Missing values remain ``None``. Raw public cell values are also preserved in
``source_details`` so a newly added Artificial Analysis column is not silently
discarded before the typed schema is updated.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from scraper.contracts import ScrapedRow, ScraperHealthReport, ScrapeResult
from scraper.utils import managed_driver, navigate_with_retry, safe_float

logger = logging.getLogger(__name__)

SOURCE_NAME = "Artificial Analysis"
URL = "https://artificialanalysis.ai/leaderboards/models"
EXPECTED_ROW_RANGE = (100, 500)
MIN_EXPANDED_COLUMNS = 35


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _number(value: Any) -> Optional[float]:
    text = _clean_text(value).replace("−", "-").replace("–", "-")
    if not text or text == "--":
        return None
    return safe_float(
        text.replace("$", "").replace("%", "").replace(",", "").strip()
    )


def _percentage(value: Any) -> Optional[float]:
    """Return percentage text such as ``62%`` as ``62.0``."""
    return _number(value)


def _parse_context_window(text: str) -> Optional[int]:
    text = _clean_text(text).lower().replace(",", "")
    if not text or text == "--":
        return None
    try:
        if text.endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("k"):
            return int(float(text[:-1]) * 1_000)
        value = _number(text)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _field(details: Dict[str, str], prefix: str) -> Optional[str]:
    """Find a column by stable heading prefix.

    Artificial Analysis occasionally changes the descriptive subtitle beneath
    a metric name. Prefix matching keeps extraction stable while the complete
    current heading/value pair remains preserved in ``source_details``.
    """
    wanted = prefix.casefold()
    for heading, value in details.items():
        if heading.casefold().startswith(wanted):
            return value
    return None


def _mean_available(*values: Optional[float]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return round(sum(available) / len(available), 2) if available else None


def _classify_availability(license_type: Optional[str]) -> Dict[str, Any]:
    """Conservatively classify availability from the published license."""
    text = (license_type or "").strip().casefold()
    if not text or text in {"--", "n/a", "unknown"}:
        return {
            "availability_class": "unknown",
            "weights_available": None,
            "source_code_available": None,
            "training_data_disclosed": None,
            "commercial_use_allowed": None,
            "open_source": None,
        }
    if "proprietary" in text or "closed" in text:
        return {
            "availability_class": "proprietary",
            "weights_available": False,
            "source_code_available": False,
            "training_data_disclosed": None,
            "commercial_use_allowed": None,
            "open_source": False,
        }
    if "research" in text or "non-commercial" in text or "noncommercial" in text:
        return {
            "availability_class": "research_license",
            "weights_available": True,
            "source_code_available": None,
            "training_data_disclosed": None,
            "commercial_use_allowed": False,
            "open_source": False,
        }
    if "open source" in text:
        return {
            "availability_class": "open_source",
            "weights_available": True,
            "source_code_available": True,
            "training_data_disclosed": None,
            "commercial_use_allowed": None,
            "open_source": True,
        }
    if text == "open" or any(
        token in text for token in ("apache", "mit", "open weight", "community")
    ):
        return {
            "availability_class": "open_weights",
            "weights_available": True,
            "source_code_available": None,
            "training_data_disclosed": None,
            "commercial_use_allowed": (
                True if "apache" in text or "mit" in text else None
            ),
            "open_source": False,
        }
    return {
        "availability_class": "unknown",
        "weights_available": None,
        "source_code_available": None,
        "training_data_disclosed": None,
        "commercial_use_allowed": None,
        "open_source": None,
    }


def _expand_columns(driver) -> None:
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(normalize-space(.), 'Expand columns')]",
                )
            )
        )
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", button
        )
        button.click()
    except TimeoutException:
        # It may already be expanded after a retry/browser-state restoration.
        if not driver.find_elements(
            By.XPATH,
            "//button[contains(normalize-space(.), 'Collapse columns')]",
        ):
            raise RuntimeError("Artificial Analysis 'Expand columns' control not found")

    WebDriverWait(driver, 20).until(
        lambda active_driver: len(
            active_driver.find_elements(
                By.CSS_SELECTOR, "table thead tr:last-child th"
            )
        )
        >= MIN_EXPANDED_COLUMNS
    )


def _extract_expanded_table(driver) -> List[Dict[str, Any]]:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(driver, 20).until(
        lambda active_driver: len(
            active_driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        )
        >= EXPECTED_ROW_RANGE[0]
    )

    header_elements = driver.find_elements(
        By.CSS_SELECTOR, "table thead tr:last-child th"
    )
    headers = [_clean_text(element.text) for element in header_elements]
    if len(headers) < MIN_EXPANDED_COLUMNS:
        raise RuntimeError(
            f"Expanded table exposed only {len(headers)} columns; "
            f"expected at least {MIN_EXPANDED_COLUMNS}"
        )

    models: List[Dict[str, Any]] = []
    for source_rank, row in enumerate(
        driver.find_elements(By.CSS_SELECTOR, "table tbody tr"), start=1
    ):
        cells = row.find_elements(By.CSS_SELECTOR, "td")
        if len(cells) != len(headers):
            logger.debug(
                "Skipping row %s because it has %s cells for %s headers",
                source_rank,
                len(cells),
                len(headers),
            )
            continue

        values = [_clean_text(cell.text) for cell in cells]
        details = dict(zip(headers, values))
        model_name = _field(details, "Model")
        if not model_name:
            continue

        links = row.find_elements(By.CSS_SELECTOR, "a[href]")
        hrefs = [link.get_attribute("href") for link in links]
        model_url = next(
            (
                urljoin(URL, href)
                for href in hrefs
                if href and "/models/" in href and "/providers" not in href
            ),
            None,
        )
        providers_url = next(
            (
                urljoin(URL, href)
                for href in hrefs
                if href and "/models/" in href and "/providers" in href
            ),
            None,
        )
        normalized_name = model_name.casefold()
        if (
            model_url
            and model_url.rstrip("/").endswith("-non-reasoning")
            and "non-reasoning" not in normalized_name
        ):
            model_name = f"{model_name} (non-reasoning)"
        elif (
            model_url
            and model_url.rstrip("/").endswith("-reasoning")
            and "reasoning" not in normalized_name
        ):
            model_name = f"{model_name} (reasoning)"

        models.append(
            {
                "name": model_name,
                "source_rank": source_rank,
                "model_url": model_url,
                "providers_url": providers_url,
                "source_details": details,
            }
        )

    return models


def _model_to_scraped_row(model: Dict[str, Any]) -> Optional[ScrapedRow]:
    details: Dict[str, str] = model.get("source_details") or {}
    name = _clean_text(model.get("name"))
    if not name:
        return None

    terminal_hard = _percentage(_field(details, "Terminal-Bench Hard"))
    terminal_v21 = _percentage(_field(details, "Terminal-Bench v2.1"))
    scicode = _percentage(_field(details, "SciCode"))
    itbench = _percentage(_field(details, "ITBench-AA"))

    typed_values = {
        "intelligence_score": _number(
            _field(details, "Artificial Analysis Intelligence Index")
        ),
        "omniscience_index": _number(
            _field(details, "Artificial Analysis Omniscience Index")
        ),
        "gdpval": _percentage(_field(details, "GDPval-AA v2")),
        "terminalbench_hard": terminal_hard,
        "terminalbench_v21": terminal_v21,
        "tau2": _percentage(_field(details, "τ²-Bench Telecom")),
        "tau3_banking": _percentage(_field(details, "𝜏³-Banking")),
        "lcr": _percentage(_field(details, "AA-LCR")),
        "omniscience": _percentage(
            _field(details, "AA-Omniscience Accuracy")
        ),
        "omniscience_hallucination": _percentage(
            _field(details, "AA-Omniscience Non-Hallucination Rate")
        ),
        "hle": _percentage(_field(details, "Humanity's Last Exam")),
        "gpqa": _percentage(_field(details, "GPQA Diamond")),
        "scicode": scicode,
        "ifbench": _percentage(_field(details, "IFBench")),
        "critpt": _percentage(_field(details, "CritPt")),
        "apex_agents": _percentage(_field(details, "APEX-Agents-AA")),
        "itbench": itbench,
        "mmmu_pro": _percentage(_field(details, "MMMU Pro")),
        "blended_cost_per_1m": _number(
            _field(details, "Blended USD/1M Tokens")
        ),
        "input_cost_per_1m": _number(
            _field(details, "Input Price USD/1M Tokens")
        ),
        "output_cost_per_1m": _number(
            _field(details, "Output Price USD/1M Tokens")
        ),
        "cache_read_cost_per_1m": _number(
            _field(details, "Cache Read USD/1M Tokens")
        ),
        "cache_write_cost_per_1m": _number(
            _field(details, "Cache Write USD/1M Tokens")
        ),
        "tokens_per_second": _number(_field(details, "Median Tokens/s")),
        "speed_p5": _number(_field(details, "P5 Tokens/s")),
        "speed_p25": _number(_field(details, "P25 Tokens/s")),
        "speed_p75": _number(_field(details, "P75 Tokens/s")),
        "speed_p95": _number(_field(details, "P95 Tokens/s")),
        "latency_seconds": _number(
            _field(details, "Latency First Chunk (s)")
        ),
        "latency_first_token": _number(_field(details, "First Answer (s)")),
        "latency_p5": _number(_field(details, "P5 First Chunk (s)")),
        "latency_p25": _number(_field(details, "P25 First Chunk (s)")),
        "latency_p75": _number(_field(details, "P75 First Chunk (s)")),
        "latency_p95": _number(_field(details, "P95 First Chunk (s)")),
        "total_response_time": _number(
            _field(details, "Total Response (s)")
        ),
        "reasoning_time": _number(_field(details, "Reasoning Time (s)")),
    }
    coding_score = _mean_available(
        terminal_hard, terminal_v21, scicode, itbench
    )
    official_coding_index = _mean_available(terminal_v21, scicode)
    filled = sum(value is not None for value in typed_values.values())
    confidence = round(min(1.0, 0.65 + filled * 0.0125), 3)

    license_type = _field(details, "License")
    creator = _field(details, "Creator")
    availability = _classify_availability(license_type)

    return ScrapedRow(
        model_name=name,
        source=SOURCE_NAME,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        source_rank=int(model["source_rank"]),
        model_url=model.get("model_url"),
        providers_url=model.get("providers_url"),
        source_details=details,
        coding_score=coding_score,
        aa_official_coding_index=official_coding_index,
        context_window=_parse_context_window(
            _field(details, "Context Window") or ""
        ),
        creator=creator,
        provider=creator,
        license_type=license_type,
        **availability,
        confidence=confidence,
        **typed_values,
    )


def _evaluate_health(row_count: int) -> str:
    low, high = EXPECTED_ROW_RANGE
    if row_count == 0:
        return "failed"
    if low <= row_count <= high:
        return "healthy"
    return "degraded"


def scrape_artificialanalysis() -> ScrapeResult:
    start_time = time.monotonic()
    rows: List[ScrapedRow] = []
    warning_count = 0

    try:
        with managed_driver() as driver:
            navigate_with_retry(driver, URL, wait_after_load_sec=8)
            _expand_columns(driver)
            raw_models = _extract_expanded_table(driver)

            for model in raw_models:
                try:
                    row = _model_to_scraped_row(model)
                    if row:
                        rows.append(row)
                except Exception as exc:
                    warning_count += 1
                    logger.warning(
                        "Skipped Artificial Analysis row %s: %s",
                        model.get("source_rank"),
                        exc,
                    )

        status = _evaluate_health(len(rows))
        return ScrapeResult(
            rows=rows,
            health=ScraperHealthReport(
                source=SOURCE_NAME,
                rows_scraped=len(rows),
                expected_range=EXPECTED_ROW_RANGE,
                status=status,
                duration_seconds=round(time.monotonic() - start_time, 2),
                parse_warning_count=warning_count,
            ),
        )
    except Exception as exc:
        logger.exception("Artificial Analysis scrape failed")
        return ScrapeResult(
            rows=[],
            health=ScraperHealthReport(
                source=SOURCE_NAME,
                rows_scraped=0,
                expected_range=EXPECTED_ROW_RANGE,
                status="failed",
                error_message=str(exc),
                duration_seconds=round(time.monotonic() - start_time, 2),
            ),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scrape_artificialanalysis()
    print(result.to_json(indent=2))
