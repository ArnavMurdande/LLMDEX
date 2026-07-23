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
from selenium.common.exceptions import TimeoutException

from scraper.contracts import (
    LLMStatsObservation,
    ScrapeResult,
    ScraperHealthReport,
)
from scraper.utils import ScraperBrowserError, managed_driver, navigate_with_retry


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


def _rendered_header_name(value: Any) -> str:
    """Return the visible primary label without coverage/help subtitles."""
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return _clean(lines[0]) if lines else ""


def parse_rendered_capability_table(
    table: Dict[str, Any],
    category: str,
    page_url: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse one public, browser-rendered LLMStats capability table.

    The rendered DOM is used because the source page does not server-render
    its interactive leaderboard. This parser consumes only the visible public
    table: it does not call private APIs, expand hidden rows, or solve
    verification challenges.
    """
    headers = list(table.get("headers") or [])
    rows = list(table.get("rows") or [])
    names = [_rendered_header_name(header.get("text")) for header in headers]
    schema_changes: List[str] = []
    if len(names) < 7 or names[:2] != ["MODEL", "Rating"]:
        schema_changes.append(
            f"{category} rendered table header changed: {', '.join(names[:7])}"
        )
        return [], {
            "headers": names,
            "schema_changes": schema_changes,
            "benchmark_columns": [],
        }

    benchmark_map = _source_name_to_benchmark()
    benchmark_columns: List[Dict[str, Any]] = []
    for index, header in enumerate(headers[6:]):
        source_name = _rendered_header_name(header.get("text"))
        if not source_name:
            continue
        registered = benchmark_map.get(source_name.casefold())
        benchmark_id = (
            registered["benchmark_id"]
            if registered
            else f"llmstats__{_slug(source_name)}"
        )
        coverage_match = re.search(
            r"(\d[\d,]*)\s+models?",
            str(header.get("text") or ""),
            flags=re.IGNORECASE,
        )
        benchmark_columns.append(
            {
                "benchmark_id": benchmark_id,
                "canonical_name": (
                    registered.get("canonical_name")
                    if registered
                    else source_name
                ),
                "source_name": source_name,
                "version": registered.get("version") if registered else None,
                "provenance": (
                    registered.get("provenance")
                    if registered
                    else "LLMStats public capability leaderboard"
                ),
                "source_url": header.get("source_url") or page_url,
                "source_column_index": index,
                "source_population": (
                    int(coverage_match.group(1).replace(",", ""))
                    if coverage_match
                    else None
                ),
            }
        )

    parsed: List[Dict[str, Any]] = []
    for source_position, row in enumerate(rows, start=1):
        cells = list(row.get("cells") or [])
        if len(cells) < len(headers):
            continue
        source_name = _clean(row.get("model_name"))
        source_model_url = row.get("model_url") or page_url
        if not source_name:
            model_lines = [
                line.strip()
                for line in str(cells[0].get("text") or "").splitlines()
                if line.strip()
            ]
            if model_lines and re.fullmatch(r"#?\d+", model_lines[0]):
                model_lines = model_lines[1:]
            source_name = _clean(model_lines[0] if model_lines else "")
        if not source_name:
            continue
        source_model_id = (
            Path(urlparse(source_model_url).path).name
            if source_model_url and source_model_url != page_url
            else None
        )
        category_score = _number(cells[1].get("text"))
        if category_score is None:
            continue

        observations: Dict[str, Dict[str, Any]] = {}
        for offset, column in enumerate(benchmark_columns, start=6):
            raw_value = _clean(cells[offset].get("text"))
            value = _number(raw_value)
            if value is None:
                continue
            benchmark_id = column["benchmark_id"]
            observations[benchmark_id] = {
                "benchmark_id": benchmark_id,
                "canonical_name": column["canonical_name"],
                "source_name": column["source_name"],
                "value": value,
                "raw_value": raw_value,
                "version": column["version"],
                "provenance": column["provenance"],
                "source_url": column["source_url"],
                "capabilities": [category],
                "source_column_indices": {
                    category: column["source_column_index"]
                },
                "source_population": column["source_population"],
            }

        parsed.append(
            {
                "source_name": source_name,
                "source_model_url": source_model_url,
                "source_model_id": source_model_id,
                "provider": None,
                "category_score": category_score,
                "category_rank": float(source_position),
                "rank_evidence": "source_rendered_table_order",
                "benchmark_observations": observations,
                "source_details": {
                    "LLMDEX extraction population": (
                        "public_browser_rendered_capability_table"
                    ),
                    "LLMDEX capability": category,
                    f"LLMDEX benchmark columns::{category}": benchmark_columns,
                    "LLMDEX category rating": _clean(cells[1].get("text")),
                    "LLMStats blended price": _clean(cells[2].get("text")),
                    "LLMStats context": _clean(cells[3].get("text")),
                    "LLMStats speed": _clean(cells[4].get("text")),
                    "LLMStats TTFT": _clean(cells[5].get("text")),
                },
            }
        )
    return parsed, {
        "headers": names,
        "schema_changes": schema_changes,
        "benchmark_columns": benchmark_columns,
    }


def _extract_rendered_capability_tables(
    urls: Dict[str, str],
    *,
    timeout: int,
    delay_seconds: float,
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Read the visible public capability tables with one bounded browser."""
    tables: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    category_urls = {
        category: url for category, url in urls.items() if category != "general"
    }
    try:
        with managed_driver(headless=True) as driver:
            for position, (category, url) in enumerate(category_urls.items()):
                if position and delay_seconds:
                    time.sleep(delay_seconds)
                try:
                    navigate_with_retry(
                        driver,
                        url,
                        retries=2,
                        wait_after_load_sec=0.5,
                    )
                    table = driver.execute_async_script(
                        """
                        const done = arguments[arguments.length - 1];
                        const deadline = Date.now() + arguments[0];
                        const inspect = () => {
                          const candidates = Array.from(
                            document.querySelectorAll("table"),
                          );
                          const table = candidates.find((candidate) => {
                            const first = candidate.querySelector("thead th");
                            return (
                              first &&
                              first.textContent.trim().toLowerCase() === "model" &&
                              candidate.querySelectorAll("tbody tr").length >= 10
                            );
                          });
                          if (!table) {
                            if (Date.now() < deadline) {
                              window.setTimeout(inspect, 250);
                              return;
                            }
                            done(null);
                            return;
                          }
                          const headers = Array.from(
                            table.querySelectorAll("thead th"),
                          ).map((header) => ({
                            text: header.innerText.trim(),
                            title: header.getAttribute("title"),
                            source_url:
                              header.querySelector("a[href]")?.href || null,
                          }));
                          const rows = Array.from(
                            table.querySelectorAll("tbody tr"),
                          ).map((row) => {
                            const cells = Array.from(
                              row.querySelectorAll(":scope > td"),
                            ).map((cell) => ({
                              text: cell.innerText.trim(),
                              title: cell.getAttribute("title"),
                            }));
                            const modelLink = row.querySelector(
                              'td:first-child a[href*="/models/"]',
                            );
                            return {
                              model_name: modelLink?.textContent.trim() || null,
                              model_url: modelLink?.href || null,
                              cells,
                            };
                          });
                          done({ headers, rows });
                        };
                        inspect();
                        """,
                        max(1000, timeout * 1000),
                    )
                    if not table:
                        body_text = driver.find_element("tag name", "body").text
                        if "confirm you're human" in body_text.casefold():
                            warnings.append(
                                f"{category} verification challenge shown; "
                                "no bypass attempted."
                            )
                        else:
                            warnings.append(
                                f"{category} rendered public table unavailable."
                            )
                        continue
                    parsed, diagnostics = parse_rendered_capability_table(
                        table,
                        category,
                        url,
                    )
                    if not parsed or diagnostics["schema_changes"]:
                        warnings.extend(diagnostics["schema_changes"])
                        continue
                    tables[category] = {
                        "rows": parsed,
                        "diagnostics": diagnostics,
                    }
                except (TimeoutException, ScraperBrowserError) as error:
                    warnings.append(
                        f"{category} rendered table unavailable: "
                        f"{type(error).__name__}"
                    )
    except ScraperBrowserError as error:
        warnings.append(
            f"Rendered capability browser unavailable: {type(error).__name__}"
        )
    return tables, warnings


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
    category_tables: Optional[Dict[str, Dict[str, Any]]] = None,
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
                "category_ranks": {},
                "rank_evidence": {},
            }
        )

    rendered_categories = set()
    for category, contract in (category_tables or {}).items():
        category_rows = contract.get("rows") or []
        if not category_rows:
            continue
        rendered_categories.add(category)
        for rendered in category_rows:
            existing = next(
                (
                    row
                    for row in parsed
                    if (
                        rendered.get("source_model_id")
                        and row.get("source_model_id")
                        == rendered.get("source_model_id")
                    )
                    or row["source_name"].casefold()
                    == rendered["source_name"].casefold()
                ),
                None,
            )
            if existing is None:
                existing = {
                    "source_name": rendered["source_name"],
                    "source_model_url": rendered["source_model_url"],
                    "source_model_id": rendered.get("source_model_id"),
                    "provider": rendered.get("provider"),
                    "general_score": None,
                    "category_scores": {
                        key: None for key in category_columns
                    },
                    "benchmark_observations": {},
                    "source_details": {},
                    "category_ranks": {},
                    "rank_evidence": {},
                }
                parsed.append(existing)
            existing["category_scores"][category] = rendered["category_score"]
            existing["category_ranks"][category] = rendered["category_rank"]
            existing["rank_evidence"][category] = rendered["rank_evidence"]
            existing["source_model_url"] = (
                existing.get("source_model_url")
                if existing.get("source_model_url") != page_url
                else rendered["source_model_url"]
            )
            existing["source_details"].update(rendered["source_details"])
            for benchmark_id, observation in rendered[
                "benchmark_observations"
            ].items():
                previous = existing["benchmark_observations"].get(benchmark_id)
                if previous:
                    capabilities = sorted(
                        set(previous.get("capabilities") or [])
                        | set(observation.get("capabilities") or [])
                    )
                    indices = {
                        **(previous.get("source_column_indices") or {}),
                        **(observation.get("source_column_indices") or {}),
                    }
                    observation = {
                        **previous,
                        **observation,
                        "capabilities": capabilities,
                        "source_column_indices": indices,
                    }
                existing["benchmark_observations"][benchmark_id] = observation

    # Category pages publish their leading models as server-rendered structured
    # data. Preserve models that are not in the currently visible General table
    # so capability views do not silently lose a source-published leader.
    for category, published in (category_top_models or {}).items():
        if category in rendered_categories:
            continue
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
                    "category_ranks": {},
                    "rank_evidence": {},
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
            category: (
                row.get("category_ranks", {}).get(category)
                if row.get("category_ranks", {}).get(category) is not None
                else ranks_by_category[category][index]
            )
            for category in categories
        }
        evidence = {
            category: row.get("rank_evidence", {}).get(
                category,
                "derived_score_order_visible_server_population",
            )
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
        category_tables, rendered_warnings = _extract_rendered_capability_tables(
            urls,
            timeout=timeout,
            delay_seconds=float(delay or 0),
        )
        warnings.extend(rendered_warnings)
        category_top_models: Dict[str, List[dict]] = {}
        missing_categories: List[str] = []
        for category, url in urls.items():
            if category == "general":
                continue
            if category in category_tables:
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
            category_tables=category_tables,
        )
        schema_changes = list(diagnostics["schema_changes"])
        if missing_categories:
            warnings.append(
                "Rendered table and supplemental top-model payload were "
                "unavailable for: "
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
    "parse_rendered_capability_table",
]
