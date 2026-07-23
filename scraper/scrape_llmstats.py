"""Policy-aware LLMStats server-rendered leaderboard scraper.

Only public HTML pages listed in ``data/methodology/source_config.json`` are
requested. The scraper never calls the site's disallowed API paths, expands
client-only rows, or attempts to solve/bypass verification controls.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests

from scraper.contracts import (
    LLMStatsObservation,
    ScrapeResult,
    ScraperHealthReport,
)


SOURCE_NAME = "LLMStats"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_PATH = PROJECT_ROOT / "data" / "methodology" / "source_config.json"
BENCHMARK_REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "methodology" / "benchmark_registry.json"
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "LLMDEX/3.0 public-research scraper "
        "(https://github.com/ArnavMurdande/LLMDEX)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUIRED_HEADERS = {"Model", "LLM Stats", "Organization"}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def _number(value: Any) -> Optional[float]:
    text = _clean(value)
    if not text or text.casefold() in {"-", "--", "—", "n/a", "na", "null"}:
        return None
    text = text.replace(",", "").replace("$", "").replace("%", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


class _PublicTableParser(HTMLParser):
    """Extract table text and first link per cell with no external parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[Dict[str, Optional[str]]]]] = []
        self._table_depth = 0
        self._rows: Optional[List[List[Dict[str, Optional[str]]]]] = None
        self._row: Optional[List[Dict[str, Optional[str]]]] = None
        self._cell_parts: Optional[List[str]] = None
        self._cell_link: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []
            self._cell_link = None
        elif (
            self._table_depth == 1
            and tag == "a"
            and self._cell_parts is not None
            and not self._cell_link
        ):
            self._cell_link = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"th", "td"} and self._cell_parts is not None:
            self._row.append(
                {"text": _clean(" ".join(self._cell_parts)), "href": self._cell_link}
            )
            self._cell_parts = None
            self._cell_link = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._rows is not None:
                self.tables.append(self._rows)
                self._rows = None
            self._table_depth -= 1


def _load_config() -> dict:
    return json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))


def _load_registry() -> dict:
    return json.loads(BENCHMARK_REGISTRY_PATH.read_text(encoding="utf-8"))


def _source_name_to_benchmark() -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for benchmark_id, entry in _load_registry().get("benchmarks", {}).items():
        for name in entry.get("source_names", {}).get("llmstats", []):
            mapping[_clean(name).casefold()] = {
                "benchmark_id": benchmark_id,
                **entry,
            }
    return mapping


def _extract_updated_at(html: str) -> Optional[str]:
    match = re.search(
        r"(?:Updated|Last updated)\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),\s+(20\d{2})",
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = datetime.strptime(" ".join(match.groups()), "%B %d %Y")
    return value.replace(tzinfo=timezone.utc).isoformat()


def _extract_top_models(html: str) -> List[dict]:
    """Read the public server-rendered top-model payload used by category cards."""
    patterns = (
        r'topModels\\":(\[\{.*?\}\])',
        r'"topModels":(\[\{.*?\}\])',
    )
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.DOTALL)
        if not match:
            continue
        payload = match.group(1).replace('\\"', '"').replace("\\\\/", "/")
        try:
            rows = json.loads(payload)
            return rows if isinstance(rows, list) else []
        except json.JSONDecodeError:
            continue
    return []


def _find_leaderboard_table(
    html: str,
) -> tuple[List[str], List[List[Dict[str, Optional[str]]]]]:
    parser = _PublicTableParser()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        for header_index, row in enumerate(table[:4]):
            headers = [_clean(cell["text"]) for cell in row]
            if "Model" in headers and "LLM Stats" in headers:
                return headers, table[header_index + 1 :]
    return [], []


def _rank_descending(values: List[Optional[float]]) -> List[Optional[float]]:
    """Average ranks for descending values; None remains unranked."""
    indexed = [(index, float(value)) for index, value in enumerate(values) if value is not None]
    indexed.sort(key=lambda item: (-item[1], item[0]))
    ranks: List[Optional[float]] = [None] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for offset in range(position, end):
            ranks[indexed[offset][0]] = float(average_rank)
        position = end
    return ranks


def parse_llmstats_table(
    html: str,
    page_url: str,
    *,
    category_top_models: Optional[Dict[str, List[dict]]] = None,
) -> tuple[List[LLMStatsObservation], Dict[str, Any]]:
    """Parse the visible server-rendered General table into source-native rows."""
    config = _load_config()["sources"]["llmstats"]
    headers, table_rows = _find_leaderboard_table(html)
    schema_changes: List[str] = []
    missing = sorted(REQUIRED_HEADERS - set(headers))
    if missing:
        schema_changes.append(f"Missing required columns: {', '.join(missing)}")
    if not headers or missing:
        return [], {
            "headers": headers,
            "schema_changes": schema_changes,
            "duplicate_source_ids": [],
        }

    category_columns = config["category_columns"]
    benchmark_map = _source_name_to_benchmark()
    source_updated_at = _extract_updated_at(html)
    parsed: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}
    duplicate_ids: List[str] = []

    for row in table_rows:
        if len(row) < len(headers):
            continue
        cells = {headers[index]: row[index] for index in range(len(headers))}
        model_cell = cells.get("Model") or {}
        name = _clean(model_cell.get("text"))
        name = re.sub(r"^\s*#?\d+\s+", "", name)
        release_status = None
        status_match = re.search(
            r"\s+(UNRELEASED|NEW)$", name, flags=re.IGNORECASE
        )
        if status_match:
            release_status = status_match.group(1).casefold()
            name = name[: status_match.start()].strip()
        if not name or name.casefold() == "model":
            continue
        source_model_url = urljoin(page_url, model_cell.get("href") or "")
        if source_model_url == page_url:
            source_model_url = page_url
        source_model_id = None
        if model_cell.get("href"):
            source_model_id = Path(urlparse(source_model_url).path).name or None
        if source_model_id:
            seen_ids[source_model_id] = seen_ids.get(source_model_id, 0) + 1
            if seen_ids[source_model_id] == 2:
                duplicate_ids.append(source_model_id)

        details = {heading: _clean(cell.get("text")) for heading, cell in cells.items()}
        if release_status:
            details["LLMDEX parsed release status"] = release_status
        category_scores = {
            category: _number(details.get(column))
            for category, column in category_columns.items()
        }
        observations: Dict[str, Dict[str, Any]] = {}
        excluded = {
            "Model",
            "LLM Stats",
            "Organization",
            "Country",
            "License",
            "Context",
            "Input $/M",
            "Output $/M",
            "Speed",
            "Latency",
            "Knowledge Cutoff",
            "Multimodal",
            "Released",
            "Parameters (B)",
            *category_columns.values(),
        }
        for heading, raw_value in details.items():
            if heading in excluded:
                continue
            numeric = _number(raw_value)
            if numeric is None:
                continue
            registered = benchmark_map.get(heading.casefold())
            benchmark_id = (
                registered["benchmark_id"]
                if registered
                else f"llmstats__{_slug(heading)}"
            )
            observations[benchmark_id] = {
                "benchmark_id": benchmark_id,
                "canonical_name": (
                    registered.get("canonical_name") if registered else heading
                ),
                "source_name": heading,
                "value": numeric,
                "raw_value": raw_value,
                "version": registered.get("version") if registered else "unknown",
                "provenance": (
                    registered.get("provenance")
                    if registered
                    else "Unknown provenance"
                ),
                "source_url": source_model_url,
            }
        parsed.append(
            {
                "source_name": name,
                "source_model_url": source_model_url,
                "source_model_id": source_model_id,
                "provider": details.get("Organization") or None,
                "general_score": category_scores.get("general"),
                "category_scores": category_scores,
                "benchmark_observations": observations,
                "source_details": details,
            }
        )

    # Category pages publish their leading models as server-rendered structured
    # data. Preserve models that are not in the currently visible General table
    # so capability views do not silently lose a source-published leader.
    for category, published in (category_top_models or {}).items():
        for item in published:
            item_id = item.get("model_id")
            item_name = _clean(item.get("name"))
            published_score = _number(item.get("elo_score"))
            if not item_name or published_score is None:
                continue
            existing = next(
                (
                    row
                    for row in parsed
                    if (item_id and row.get("source_model_id") == item_id)
                    or row["source_name"].casefold() == item_name.casefold()
                ),
                None,
            )
            if existing is None:
                existing = {
                    "source_name": item_name,
                    "source_model_url": config["leaderboard_urls"].get(
                        category, page_url
                    ),
                    "source_model_id": item_id,
                    "provider": item.get("organization"),
                    "general_score": None,
                    "category_scores": {
                        key: None for key in category_columns
                    },
                    "benchmark_observations": {},
                    "source_details": {
                        "Model": item_name,
                        "Organization": item.get("organization"),
                        "LLMDEX extraction population": (
                            "source_published_category_top_models"
                        ),
                    },
                }
                parsed.append(existing)
            existing["category_scores"][category] = published_score

    categories = list(category_columns)
    ranks_by_category = {
        category: _rank_descending(
            [row["category_scores"].get(category) for row in parsed]
        )
        for category in categories
    }
    top_models = category_top_models or {}
    timestamp = datetime.now(timezone.utc).isoformat()
    observations: List[LLMStatsObservation] = []
    for index, row in enumerate(parsed):
        ranks = {
            category: ranks_by_category[category][index] for category in categories
        }
        evidence = {
            category: "derived_score_order_visible_server_population"
            for category in categories
        }
        for category, published in top_models.items():
            for source_rank, item in enumerate(published, start=1):
                same_id = (
                    row["source_model_id"]
                    and item.get("model_id") == row["source_model_id"]
                )
                same_name = _clean(item.get("name")).casefold() == row["source_name"].casefold()
                if same_id or same_name:
                    ranks[category] = float(source_rank)
                    evidence[category] = "source_published_top_models_order"
                    break
        observations.append(
            LLMStatsObservation(
                source_name=row["source_name"],
                source_model_url=row["source_model_url"],
                source_model_id=row["source_model_id"],
                provider=row["provider"],
                scraped_at=timestamp,
                general_score=row["general_score"],
                general_rank=ranks.get("general"),
                category_scores=row["category_scores"],
                category_ranks=ranks,
                rank_evidence=evidence,
                benchmark_observations=row["benchmark_observations"],
                source_details=row["source_details"],
                source_updated_at=source_updated_at,
            )
        )

    return observations, {
        "headers": headers,
        "schema_changes": schema_changes,
        "duplicate_source_ids": duplicate_ids,
        "source_updated_at": source_updated_at,
    }


def _fetch_html(session: requests.Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def scrape_llmstats(
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 45,
    delay_seconds: Optional[float] = None,
) -> ScrapeResult:
    """Fetch the configured public pages and return a structured health report."""
    started = time.monotonic()
    config = _load_config()["sources"]["llmstats"]
    urls = config["leaderboard_urls"]
    expected = tuple(config["extraction"]["expected_rows"])
    delay = (
        config["extraction"]["rate_limit_seconds"]
        if delay_seconds is None
        else delay_seconds
    )
    client = session or requests.Session()
    client.headers.update(DEFAULT_HEADERS)
    warnings: List[str] = []
    try:
        general_html = _fetch_html(client, urls["general"], timeout)
        category_top_models: Dict[str, List[dict]] = {}
        missing_categories: List[str] = []
        for category, url in urls.items():
            if category == "general":
                continue
            if delay:
                time.sleep(delay)
            try:
                category_html = _fetch_html(client, url, timeout)
                top = _extract_top_models(category_html)
                if top:
                    category_top_models[category] = top
                else:
                    missing_categories.append(category)
            except requests.RequestException as error:
                missing_categories.append(category)
                warnings.append(f"{category} page unavailable: {type(error).__name__}")

        rows, diagnostics = parse_llmstats_table(
            general_html,
            urls["general"],
            category_top_models=category_top_models,
        )
        schema_changes = list(diagnostics["schema_changes"])
        if missing_categories:
            warnings.append(
                "Supplemental published top-model payload was unavailable for: "
                + ", ".join(sorted(missing_categories))
                + ". Visible-table category scores remain available."
            )
        if diagnostics["duplicate_source_ids"]:
            schema_changes.append(
                "Duplicate source IDs: "
                + ", ".join(diagnostics["duplicate_source_ids"])
            )
        low, high = expected
        if not rows:
            status = "failed"
            error_message = "No valid public leaderboard rows were extracted."
        elif diagnostics["duplicate_source_ids"] or len(rows) < low:
            status = "failed"
            error_message = "LLMStats extraction failed publication safety checks."
        elif len(rows) > high or schema_changes:
            status = "degraded"
            error_message = None
        else:
            status = "healthy"
            error_message = None
        timestamp = datetime.now(timezone.utc).isoformat()
        return ScrapeResult(
            rows=rows,  # type: ignore[arg-type]
            health=ScraperHealthReport(
                source=SOURCE_NAME,
                rows_scraped=len(rows),
                expected_range=expected,
                status=status,
                error_message=error_message,
                duration_seconds=round(time.monotonic() - started, 3),
                timestamp=timestamp,
                parse_warning_count=sum(len(row.parse_warnings) for row in rows),
                warnings=warnings,
                last_successful_update=(
                    diagnostics.get("source_updated_at") or timestamp
                    if status in {"healthy", "degraded"}
                    else None
                ),
                schema_changes=schema_changes,
            ),
        )
    except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
        return ScrapeResult(
            rows=[],
            health=ScraperHealthReport(
                source=SOURCE_NAME,
                rows_scraped=0,
                expected_range=expected,
                status="failed",
                error_message=f"{type(error).__name__}: {error}",
                duration_seconds=round(time.monotonic() - started, 3),
                warnings=warnings,
                schema_changes=[],
            ),
        )


__all__ = [
    "parse_llmstats_table",
    "scrape_llmstats",
    "_extract_top_models",
    "_rank_descending",
]
