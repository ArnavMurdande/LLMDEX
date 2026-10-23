"""Build a readable model-family progression dataset from the current index."""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_families import (
    infer_family_brand,
    infer_model_family,
    normalize_release_name,
    release_sort_key,
)

try:
    import requests
except ImportError:  # pragma: no cover - the application requirements include it
    requests = None


def infer_release_date(name: str) -> str | None:
    """Infer a date only when the public model name contains one explicitly."""
    compact = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", name or "")
    if compact:
        candidate = "-".join(compact.groups())
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
        except ValueError:
            pass
    dashed = re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", name or "")
    if dashed:
        candidate = "-".join(dashed.groups())
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
        except ValueError:
            pass
    return None


def extract_release_date(page_text: str) -> str | None:
    """Extract the current model page's published release date."""
    exact = re.search(
        r"released on ([A-Z][a-z]+ \d{1,2}, \d{4})",
        page_text or "",
        re.IGNORECASE,
    )
    if exact:
        try:
            return datetime.strptime(exact.group(1), "%B %d, %Y").date().isoformat()
        except ValueError:
            pass

    structured = re.search(
        r'releaseDate\\?":\\?"(\d{4}-\d{2}-\d{2})',
        page_text or "",
    )
    return structured.group(1) if structured else None


def extract_embedded_model_catalog(page_text: str) -> List[dict]:
    """Read the historical model catalog embedded in a public model page."""
    decoded = (page_text or "").replace('\\"', '"')
    decoder = json.JSONDecoder()
    needle = '{"id":"'
    rows: List[dict] = []
    seen: set[str] = set()
    position = 0
    while True:
        position = decoded.find(needle, position)
        if position < 0:
            break
        try:
            model, end = decoder.raw_decode(decoded, position)
        except (json.JSONDecodeError, TypeError):
            position += len(needle)
            continue
        position = max(end, position + len(needle))
        if not isinstance(model, dict) or "outputModalityVideo" not in model:
            continue
        slug = model.get("slug")
        name = model.get("name")
        release_date = model.get("releaseDate")
        intelligence = model.get("intelligenceIndex")
        if not slug or not name or not release_date or intelligence is None:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        creator = model.get("creator") or {}
        rows.append(
            {
                "canonical_name": name,
                "model_name": name,
                "provider": creator.get("name") or "Unknown",
                "adjusted_performance": float(intelligence),
                "performance_rank": None,
                "model_url": f"https://artificialanalysis.ai/models/{slug}",
                "release_date": release_date,
                "historical": bool(model.get("deprecated")),
            }
        )
    return rows


def resolve_model_meta(
    name: str,
    existing_url: str | None = None,
    release_dates: Dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve canonical model URL and published release date from known release mapping."""
    if not release_dates:
        return existing_url, None

    if existing_url and existing_url in release_dates:
        return existing_url, release_dates[existing_url]

    slug_map = {
        url.split("/")[-1].lower(): (url, date)
        for url, date in release_dates.items()
    }

    if existing_url:
        slug = existing_url.split("/")[-1].lower()
        if slug in slug_map:
            return slug_map[slug]

    clean = (name or "").lower().strip()
    clean = re.sub(r"\s*\([^)]*\)", "", clean).strip()

    s1 = re.sub(r"[\s_]+", "-", clean)
    s1 = re.sub(r"\.", "-", s1)
    s1 = re.sub(r"[^a-z0-9\-]", "", s1)

    s2 = re.sub(r"[\s_]+", "-", clean)
    s2 = re.sub(r"\.", "", s2)
    s2 = re.sub(r"[^a-z0-9\-]", "", s2)

    s3 = re.sub(r"-0$", "", s1)

    candidates = [s1, s2, s3]
    for c in list(candidates):
        candidates.append(c + "-thinking")
        candidates.append(c + "-reasoning")
        candidates.append(c + "-adaptive")

    for c in candidates:
        if c in slug_map:
            return slug_map[c]

    return existing_url, None


def load_historical_catalog(seed_url: str | None, cache_path: Path) -> List[dict]:
    """Refresh the historical catalog from one model page, with disk fallback."""
    existing: List[dict] = []
    if cache_path.exists():
        try:
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_by_key = {
        row.get("model_url")
        or row.get("canonical_name")
        or row.get("model_name"): row
        for row in existing
        if row
    }

    catalog: List[dict] = []
    if requests is not None and seed_url:
        try:
            response = requests.get(
                seed_url,
                headers={
                    "User-Agent": (
                        "LLMDEX/2.0 family-history "
                        "(+https://github.com/ArnavMurdande/LLMDEX)"
                    )
                },
                timeout=25,
            )
            if response.status_code == 200:
                catalog = extract_embedded_model_catalog(response.text)
        except Exception as exc:
            print(f"Historical catalog refresh failed: {exc}")

    for row in catalog:
        key = (
            row.get("model_url")
            or row.get("canonical_name")
            or row.get("model_name")
        )
        if key:
            existing_by_key[key] = row

    merged = list(existing_by_key.values())
    if merged:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _fetch_release_date(url: str) -> tuple[str, str | None]:
    if requests is None:
        return url, None
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "LLMDEX/2.0 family-history "
                    "(+https://github.com/ArnavMurdande/LLMDEX)"
                )
            },
            timeout=18,
        )
        if response.status_code == 200:
            return url, extract_release_date(response.text)
    except Exception as exc:
        print(f"Release-date lookup failed for {url}: {exc}")
    return url, None


def enrich_release_dates(
    history: Dict[str, List[dict]],
    cache_path: Path,
) -> Dict[str, List[dict]]:
    """Cache release dates from the same model pages used by the benchmark."""
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    urls = {
        member.get("model_url")
        for members in history.values()
        for member in members
        if member.get("model_url")
    }
    missing = sorted(url for url in urls if not cache.get(url))
    if missing and requests is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for url, release_date in executor.map(_fetch_release_date, missing):
                if release_date:
                    cache[url] = release_date

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for members in history.values():
        for member in members:
            cached_url = member.get("model_url")
            resolved_url, resolved_date = resolve_model_meta(
                member.get("name", ""),
                cached_url,
                cache,
            )
            if resolved_url and not member.get("model_url"):
                member["model_url"] = resolved_url

            member["release_date"] = (
                cache.get(member.get("model_url"))
                or resolved_date
                or member.get("release_date")
                or infer_release_date(member.get("name", ""))
            )
        members.sort(
            key=lambda item: (
                item.get("release_date") is None,
                item.get("release_date") or "",
                release_sort_key(item.get("name", "")),
            )
        )
        for index, member in enumerate(members):
            previous = members[index - 1] if index else None
            difference = (
                member["performance"] - previous["performance"] if previous else 0.0
            )
            member["improvement_abs"] = round(difference, 2)
            member["improvement_pct"] = round(
                (difference / previous["performance"] * 100)
                if previous and previous["performance"]
                else 0.0,
                2,
            )
            member["predecessor"] = previous["name"] if previous else None
    return history


def build_family_history(data: List[dict]) -> Dict[str, List[dict]]:
    """Collapse effort variants and compute release-to-release improvements."""
    grouped: Dict[str, Dict[str, List[dict]]] = {}

    for row in data:
        name = row.get("canonical_name") or row.get("model_name")
        if not name:
            continue
        performance = row.get("adjusted_performance")
        if performance is None:
            performance = row.get("performance_index")
        if performance is None:
            continue

        family = infer_model_family(name, row.get("provider"))
        release = normalize_release_name(name)
        grouped.setdefault(family, {}).setdefault(release, []).append(
            {
                "name": name,
                "performance": float(performance),
                "rank": row.get("performance_rank"),
                "provider": row.get("provider"),
                "model_url": row.get("model_url"),
                "release_date": row.get("release_date")
                or infer_release_date(name),
            }
        )

    output: Dict[str, List[dict]] = {}
    for family, releases in grouped.items():
        members: List[dict] = []
        for release, variants in releases.items():
            best = max(variants, key=lambda item: item["performance"]).copy()
            best["name"] = release
            best["variants"] = [item["name"] for item in variants]
            best["variant_count"] = len(variants)
            members.append(best)

        members.sort(key=lambda item: release_sort_key(item["name"]))
        for index, member in enumerate(members):
            previous = members[index - 1] if index else None
            difference = (
                member["performance"] - previous["performance"] if previous else 0.0
            )
            member["improvement_abs"] = round(difference, 2)
            member["improvement_pct"] = round(
                (difference / previous["performance"] * 100)
                if previous and previous["performance"]
                else 0.0,
                2,
            )
            member["predecessor"] = previous["name"] if previous else None
        output[family] = members

    # Hide one-off brands, while preserving singleton product lines such as
    # Claude Fable whenever their brand has other releases in the dataset.
    brand_release_counts: Dict[str, int] = {}
    for family, members in output.items():
        brand = infer_family_brand(family)
        brand_release_counts[brand] = brand_release_counts.get(brand, 0) + len(members)

    return {
        family: members
        for family, members in sorted(output.items())
        if brand_release_counts.get(infer_family_brand(family), 0) >= 2
    }


def build_history() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    latest_path = os.path.join(base_dir, "index", "latest.json")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"Index not found: {latest_path}")

    with open(latest_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    history_dir = os.path.join(base_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    release_dates_path = Path(history_dir) / "release_dates.json"
    try:
        release_dates = json.loads(release_dates_path.read_text(encoding="utf-8"))
    except Exception:
        release_dates = {}

    catalog_path = Path(history_dir) / "historical_models.json"
    seed_url = next((row.get("model_url") for row in data if row.get("model_url")), None)
    historical = load_historical_catalog(seed_url, catalog_path)

    combined: Dict[str, dict] = {}

    # 1. Ingest historical snapshots chronologically to recover archived models
    snapshot_files = sorted(
        p
        for p in glob.glob(os.path.join(history_dir, "2026-*.json"))
        if not p.endswith(
            (
                "family_growth.json",
                "historical_models.json",
                "release_dates.json",
                "family_snapshots.json",
            )
        )
    )
    for snapshot_file in snapshot_files:
        try:
            with open(snapshot_file, "r", encoding="utf-8") as file:
                snapshot_rows = json.load(file)
            for row in snapshot_rows:
                name = row.get("canonical_name") or row.get("model_name")
                if not name or re.search(r"-\d{8}", name) or re.match(r"^[a-z]+-[0-9]", name):
                    continue
                perf = row.get("adjusted_performance")
                if perf is None:
                    perf = row.get("performance_index")
                if perf is None:
                    continue
                url, date = resolve_model_meta(name, row.get("model_url"), release_dates)
                if not date and not url:
                    continue
                row_copy = dict(row)
                row_copy["adjusted_performance"] = float(perf)
                provider = row.get("provider")
                row_copy["provider"] = (
                    str(provider)
                    if provider is not None and str(provider).lower() != "nan"
                    else ""
                )
                if url:
                    row_copy["model_url"] = url
                if date:
                    row_copy["release_date"] = date
                key = name
                if key not in combined or row_copy["adjusted_performance"] > (
                    combined[key].get("adjusted_performance") or 0
                ):
                    combined[key] = row_copy
        except Exception as exc:
            print(f"Warning: could not process snapshot {snapshot_file}: {exc}")

    # 2. Ingest historical SSR catalog
    for row in historical:
        name = row.get("canonical_name") or row.get("model_name")
        if not name:
            continue
        row_copy = dict(row)
        url, date = resolve_model_meta(name, row.get("model_url"), release_dates)
        if url:
            row_copy["model_url"] = url
        if date:
            row_copy["release_date"] = date
        key = name
        combined[key] = row_copy

    # 3. Ingest latest active index
    for row in data:
        name = row.get("canonical_name") or row.get("model_name")
        if not name:
            continue
        row_copy = dict(row)
        url, date = resolve_model_meta(name, row.get("model_url"), release_dates)
        if url:
            row_copy["model_url"] = url
        if date:
            row_copy["release_date"] = date
        key = name
        combined[key] = row_copy

    history = build_family_history(list(combined.values()))
    history = enrich_release_dates(
        history,
        release_dates_path,
    )
    output_path = os.path.join(history_dir, "family_growth.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Built {len(history)} model families: {output_path}")


if __name__ == "__main__":
    build_history()
