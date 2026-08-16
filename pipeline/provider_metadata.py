"""Dynamic provider discovery and curated metadata for Power BI exports."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "methodology" / "provider_metadata_registry.json"
LOGO_DIR = ROOT / "website" / "assets" / "providers"
LOGO_BASE_URL = "https://llmdex.pages.dev/assets/providers"
SCHEMA_VERSION = "powerbi-v1"

PROVIDER_COLUMNS = [
    "schema_version", "snapshot_date", "provider_id", "provider_name",
    "parent_company", "provider_group", "provider_aliases", "country",
    "country_code", "region", "hq_city", "latitude", "longitude",
    "website_url", "logo_url", "logo_dark_url", "brand_color",
    "founded_year", "is_active", "metadata_source_url", "metadata_verified_at",
]

ISO2_CODES = set("AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW".split())


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold()).strip("-") or "unknown-provider"


def load_registry(path: Path = REGISTRY_PATH) -> List[Dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("providers", [])
    seen_ids: set[str] = set()
    seen_aliases: Dict[str, str] = {}
    for row in rows:
        provider_id = row.get("provider_id", "")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", provider_id):
            raise ValueError(f"Invalid provider_id in registry: {provider_id!r}")
        if provider_id in seen_ids:
            raise ValueError(f"Duplicate provider_id in registry: {provider_id}")
        seen_ids.add(provider_id)
        for alias in [provider_id, row.get("provider_name", ""), row.get("provider_group", ""), *row.get("aliases", [])]:
            key = _norm(alias)
            if not key:
                continue
            prior = seen_aliases.get(key)
            if prior and prior != provider_id:
                raise ValueError(f"Provider alias collision: {alias!r} -> {prior}/{provider_id}")
            seen_aliases[key] = provider_id
    return rows


def _write_fallback_logo(provider_id: str, provider_name: str) -> None:
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGO_DIR / f"{provider_id}.svg"
    if path.exists():
        return
    initials = "".join(word[0] for word in re.findall(r"[A-Za-z0-9]+", provider_name)[:2]).upper() or "AI"
    label = html.escape(provider_name, quote=True)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" '
        f'aria-label="{label}"><rect width="64" height="64" rx="15" fill="#2563EB"/>'
        f'<text x="32" y="39" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="22" font-weight="700" fill="#fff">{html.escape(initials)}</text></svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def build_provider_metadata(
    provider_values: Iterable[str], snapshot_date: str
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    """Return provider dimension rows, raw-value mapping, and validation report."""
    registry = load_registry()
    by_id = {r["provider_id"]: r for r in registry}
    alias_index: Dict[str, str] = {}
    for row in registry:
        for alias in [row["provider_id"], row.get("provider_name", ""), row.get("provider_group", ""), *row.get("aliases", [])]:
            if _norm(alias):
                alias_index[_norm(alias)] = row["provider_id"]

    observed: Dict[str, set[str]] = {}
    raw_mapping: Dict[str, str] = {}
    for raw in sorted({str(v).strip() for v in provider_values if str(v).strip()}, key=str.casefold):
        provider_id = alias_index.get(_norm(raw), _slug(raw))
        raw_mapping[raw] = provider_id
        observed.setdefault(provider_id, set()).add(raw)

    output: List[Dict[str, Any]] = []
    fully_enriched: List[str] = []
    requiring_metadata: List[str] = []
    required = ("country", "country_code", "region", "hq_city", "latitude", "longitude", "website_url", "metadata_source_url")
    for provider_id in sorted(observed):
        curated = by_id.get(provider_id, {})
        provider_name = curated.get("provider_name") or sorted(observed[provider_id], key=lambda s: (len(s), s.casefold()))[0]
        aliases = sorted(set(curated.get("aliases", [])) | observed[provider_id] | {provider_name}, key=str.casefold)
        _write_fallback_logo(provider_id, provider_name)
        row = {column: "" for column in PROVIDER_COLUMNS}
        row.update({
            "schema_version": SCHEMA_VERSION,
            "snapshot_date": snapshot_date,
            "provider_id": provider_id,
            "provider_name": provider_name,
            "provider_group": curated.get("provider_group") or provider_id,
            "provider_aliases": ";".join(aliases),
            "logo_url": f"{LOGO_BASE_URL}/{provider_id}.svg",
            "is_active": "True",
        })
        for field in PROVIDER_COLUMNS:
            if field in curated and curated[field] is not None:
                row[field] = curated[field]
        if row["country_code"] and row["country_code"] not in ISO2_CODES:
            raise ValueError(f"Invalid ISO-2 country code for {provider_id}: {row['country_code']}")
        for field in ("website_url", "logo_url", "logo_dark_url", "metadata_source_url"):
            if row[field] and not str(row[field]).startswith("https://"):
                raise ValueError(f"Non-HTTPS {field} for {provider_id}: {row[field]}")
        if row["brand_color"] and not re.fullmatch(r"#[0-9A-Fa-f]{6}", str(row["brand_color"])):
            raise ValueError(f"Invalid brand_color for {provider_id}: {row['brand_color']}")
        for field, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
            if row[field] != "" and not low <= float(row[field]) <= high:
                raise ValueError(f"Invalid {field} for {provider_id}: {row[field]}")
        if all(row[field] != "" for field in required):
            fully_enriched.append(provider_id)
        else:
            requiring_metadata.append(provider_id)
        output.append(row)

    report = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
        "providers_discovered": len(output),
        "providers_fully_enriched": len(fully_enriched),
        "providers_requiring_metadata": len(requiring_metadata),
        "unresolved_provider_ids": requiring_metadata,
        "provider_mappings": [
            {"provider_value": raw, "provider_id": provider_id, "provider_name": next(r["provider_name"] for r in output if r["provider_id"] == provider_id)}
            for raw, provider_id in sorted(raw_mapping.items(), key=lambda item: item[0].casefold())
        ],
    }
    return output, raw_mapping, report
