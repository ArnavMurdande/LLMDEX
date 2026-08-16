"""Deterministic family/variant identity contracts for LLMDEX.

The module deliberately separates family normalization from variant parsing.
Fuzzy similarity can create a review candidate, but can never publish a merge.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


MATCHED_STATUSES = {"matched_exact", "matched_family", "matched_manual"}
REVIEW_STATUSES = {"identity_unresolved", "ambiguous"}
VARIANT_WORDS = {
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "min",
    "reasoning",
    "non-reasoning",
    "non reasoning",
    "thinking",
    "non-thinking",
    "non thinking",
    "adaptive",
    "effort",
    "fallback",
    "with fallback",
}
PROVIDER_ALIASES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "google deepmind": "Google",
    "deepmind": "Google",
    "meta": "Meta",
    "meta ai": "Meta",
    "deepseek": "DeepSeek",
    "mistral": "Mistral AI",
    "mistral ai": "Mistral AI",
    "alibaba": "Alibaba",
    "alibaba cloud": "Alibaba",
    "qwen": "Alibaba",
    "zhipu": "Zhipu AI",
    "zhipu ai": "Zhipu AI",
    "xai": "xAI",
    "moonshot": "Moonshot AI",
    "moonshotai": "Moonshot AI",
    "moonshot ai": "Moonshot AI",
    "minimax": "MiniMax",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unknown"


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    """Return a stable provider label without guessing from the model name."""
    text = _clean(provider)
    if not text:
        return None
    key = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return PROVIDER_ALIASES.get(key, text)


def parse_variant_suffixes(name: str) -> Dict[str, Any]:
    """Extract variant/deployment metadata while preserving the source name."""
    original = _clean(name)
    parenthetical = [_clean(value) for value in re.findall(r"\(([^)]*)\)", original)]
    lower = original.casefold()
    effort = None
    for token in ("xhigh", "max", "high", "medium", "low", "min"):
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lower):
            effort = token
            break

    profiles: List[str] = []
    profile_patterns = (
        ("non-reasoning", r"\bnon[-\s]?reasoning\b"),
        ("non-thinking", r"\bnon[-\s]?thinking\b"),
        ("adaptive_reasoning", r"\badaptive\b"),
        ("thinking", r"\bthinking\b"),
        ("reasoning", r"\breasoning\b"),
        ("preview", r"\bpreview\b"),
        ("fallback", r"\bfallback\b"),
    )
    for label, pattern in profile_patterns:
        if re.search(pattern, lower) and label not in profiles:
            profiles.append(label)
    if "non-reasoning" in profiles and "reasoning" in profiles:
        profiles.remove("reasoning")
    if "non-thinking" in profiles and "thinking" in profiles:
        profiles.remove("thinking")

    dates = re.findall(r"(?<!\d)(20\d{6}|20\d{2}[-_/]\d{2}[-_/]\d{2})(?!\d)", original)
    contexts = re.findall(r"(?<!\w)(\d+(?:\.\d+)?[kKmM])(?:\s*(?:ctx|context))?(?!\w)", original)
    parameter_match = re.search(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?:\s*[-/]\s*[aA](\d+(?:\.\d+)?)[bB])?(?!\w)",
        original,
    )
    parameter_count = None
    if parameter_match:
        parameter_count = {
            "total_billions": float(parameter_match.group(1)),
            "active_billions": (
                float(parameter_match.group(2)) if parameter_match.group(2) else None
            ),
        }

    return {
        "source_name": original,
        "reasoning_effort": effort,
        "deployment_profile": profiles,
        "fallback_enabled": True if "fallback" in profiles else False,
        "parenthetical_labels": parenthetical,
        "date_identifiers": dates,
        "context_labels": contexts,
        "parameter_count": parameter_count,
    }


def _is_variant_parenthetical(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return any(word.replace("-", " ") in normalized for word in VARIANT_WORDS)


def normalize_family_name(name: str) -> str:
    """Normalize a family label without discarding unrecognized qualifiers."""
    value = _clean(name)

    def keep_or_remove(match: re.Match[str]) -> str:
        return "" if _is_variant_parenthetical(match.group(1)) else f" ({_clean(match.group(1))})"

    value = re.sub(r"\s*\(([^)]*)\)", keep_or_remove, value)
    value = re.sub(
        r"(?:[-\s]+)(?:non[-\s]?reasoning|non[-\s]?thinking|thinking|adaptive|"
        r"xhigh|max|high|medium|low|min)(?:[-\s]+\d+(?:\.\d+)?[kKmM])?$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Punctuation and whitespace differences should not split a family. Version
    # numbers, preview labels, dates and parameter labels remain in the value.
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip(" -")
    return value or _clean(name)


def _comparison_key(name: str) -> str:
    value = normalize_family_name(name).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_version(name: str) -> Optional[str]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", name)
    if match:
        return match.group(1)
    date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", name)
    return date_match.group(1) if date_match else None


def parse_model_identity(
    name: str,
    provider: Optional[str] = None,
    *,
    source: Optional[str] = None,
    source_model_id: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Create stable family, variant and deployment identifiers."""
    source_name = _clean(name)
    provider_name = normalize_provider(provider)
    family_name = normalize_family_name(source_name)
    family_id = f"{_slug(provider_name or 'unknown')}/{_slug(family_name)}"
    variant = parse_variant_suffixes(source_name)
    variant_tokens = []
    if variant["reasoning_effort"]:
        variant_tokens.append(variant["reasoning_effort"])
    variant_tokens.extend(variant["deployment_profile"])
    variant_key = "-".join(dict.fromkeys(variant_tokens)) or "default"
    variant_id = f"{family_id}:{_slug(variant_key)}"
    deployment_profile_id = f"{variant_id}:deployment"
    return {
        "family_id": family_id,
        "variant_id": variant_id,
        "deployment_profile_id": deployment_profile_id,
        "source_model_id": source_model_id,
        "source_model_url": source_url,
        "source": source,
        "source_name": source_name,
        "provider": provider_name,
        "base_model_name": family_name,
        "canonical_family_name": family_name,
        "version": _extract_version(family_name),
        "reasoning_effort": variant["reasoning_effort"],
        "deployment_profile": variant["deployment_profile"],
        "fallback_enabled": variant["fallback_enabled"],
        "parameter_count": variant["parameter_count"],
        "aliases": [source_name],
        "variant_metadata": variant,
        "comparison_key": _comparison_key(family_name),
    }


@dataclass
class MatchResult:
    source: str
    source_name: str
    source_model_id: Optional[str]
    source_model_url: Optional[str]
    provider: Optional[str]
    candidate_family_id: Optional[str]
    candidate_family_name: Optional[str]
    match_status: str
    match_confidence: float
    match_method: str
    review_reason: Optional[str] = None
    competing_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_family_registry(aa_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build the current registry from AA variants without collapsing them."""
    registry: Dict[str, Dict[str, Any]] = {}
    for row in aa_rows:
        identity = parse_model_identity(
            row.get("canonical_name") or row.get("model_name") or "",
            row.get("provider") or row.get("creator"),
            source="artificial_analysis",
            source_model_id=row.get("source_model_id") or row.get("model_id"),
            source_url=row.get("model_url"),
        )
        family = registry.setdefault(
            identity["family_id"],
            {
                "family_id": identity["family_id"],
                "canonical_family_name": identity["canonical_family_name"],
                "provider": identity["provider"],
                "version": identity["version"],
                "source_aliases": {"artificial_analysis": [], "llmstats": []},
                "source_ids": {},
                "source_urls": {},
                "variants": {},
            },
        )
        family["source_aliases"]["artificial_analysis"].append(identity["source_name"])
        if identity["source_model_id"]:
            family["source_ids"].setdefault("artificial_analysis", []).append(
                identity["source_model_id"]
            )
        if identity["source_model_url"]:
            family["source_urls"].setdefault("artificial_analysis", []).append(
                identity["source_model_url"]
            )
        family["variants"][identity["variant_id"]] = {
            key: identity[key]
            for key in (
                "variant_id",
                "deployment_profile_id",
                "source_name",
                "reasoning_effort",
                "deployment_profile",
                "fallback_enabled",
                "parameter_count",
            )
        }
    for family in registry.values():
        for key in ("source_aliases", "source_ids", "source_urls"):
            for source, values in family[key].items():
                family[key][source] = sorted(set(values))
    return dict(sorted(registry.items()))


def _manual_alias_lookup(overrides: Mapping[str, Any]) -> Dict[tuple[str, str], str]:
    result = {}
    for item in overrides.get("approved_aliases", []):
        key = (str(item.get("source", "")).casefold(), _comparison_key(item.get("source_name", "")))
        result[key] = item.get("family_id")
    return result


def generate_match_candidate(
    observation: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    overrides: Optional[Mapping[str, Any]] = None,
) -> MatchResult:
    """Match in the approved deterministic order.

    Fuzzy similarity is returned only as ``identity_unresolved`` or
    ``ambiguous`` and therefore cannot enter a published consensus.
    """
    source = str(observation.get("source") or "llmstats").casefold()
    name = _clean(observation.get("source_name") or observation.get("model_name"))
    provider = normalize_provider(observation.get("provider"))
    source_id = observation.get("source_model_id")
    source_url = observation.get("source_model_url") or observation.get("model_url")
    parsed = parse_model_identity(name, provider, source=source)

    for family_id, family in registry.items():
        if source_id and any(
            source_id in values for values in family.get("source_ids", {}).values()
        ):
            return MatchResult(
                source, name, source_id, source_url, provider, family_id,
                family.get("canonical_family_name"), "matched_exact", 1.0,
                "exact_source_model_id",
            )
    for family_id, family in registry.items():
        if source_url and any(
            source_url in values for values in family.get("source_urls", {}).values()
        ):
            return MatchResult(
                source, name, source_id, source_url, provider, family_id,
                family.get("canonical_family_name"), "matched_exact", 1.0,
                "exact_source_model_url",
            )

    manual = _manual_alias_lookup(overrides or {})
    manual_family_id = manual.get((source, _comparison_key(name)))
    if manual_family_id in registry:
        family = registry[manual_family_id]
        return MatchResult(
            source, name, source_id, source_url, provider, manual_family_id,
            family.get("canonical_family_name"), "matched_manual", 1.0,
            "approved_alias",
        )

    alias_matches = []
    for family_id, family in registry.items():
        aliases = family.get("source_aliases", {}).get(source, [])
        if any(_comparison_key(alias) == _comparison_key(name) for alias in aliases):
            alias_matches.append((family_id, family))
    if len(alias_matches) == 1:
        family_id, family = alias_matches[0]
        return MatchResult(
            source, name, source_id, source_url, provider, family_id,
            family.get("canonical_family_name"), "matched_exact", 1.0,
            "exact_approved_registry_alias",
        )

    exact_family_matches = []
    for family_id, family in registry.items():
        same_provider = normalize_provider(family.get("provider")) == provider
        same_name = _comparison_key(family.get("canonical_family_name", "")) == parsed["comparison_key"]
        version_ok = (
            not parsed["version"]
            or not family.get("version")
            or parsed["version"] == family.get("version")
        )
        if same_provider and same_name and version_ok:
            exact_family_matches.append((family_id, family))
    if len(exact_family_matches) == 1:
        family_id, family = exact_family_matches[0]
        return MatchResult(
            source, name, source_id, source_url, provider, family_id,
            family.get("canonical_family_name"), "matched_family", 0.95,
            "exact_provider_family_version",
        )
    if len(exact_family_matches) > 1:
        return MatchResult(
            source, name, source_id, source_url, provider, None, None,
            "ambiguous", 0.8, "multiple_exact_family_candidates",
            "Multiple families share the same provider/name/version key.",
            [
                {"family_id": item[0], "family_name": item[1].get("canonical_family_name")}
                for item in exact_family_matches
            ],
        )

    scored = []
    for family_id, family in registry.items():
        if provider and normalize_provider(family.get("provider")) != provider:
            continue
        similarity = SequenceMatcher(
            None,
            parsed["comparison_key"],
            _comparison_key(family.get("canonical_family_name", "")),
        ).ratio()
        if similarity >= 0.72:
            scored.append((similarity, family_id, family))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    if not scored:
        return MatchResult(
            source, name, source_id, source_url, provider, None, None,
            "source_missing", 0.0, "no_candidate",
            "No corresponding family exists in the current AA population.",
        )

    top = scored[0]
    competing = [
        {
            "family_id": family_id,
            "family_name": family.get("canonical_family_name"),
            "similarity": round(similarity, 4),
        }
        for similarity, family_id, family in scored[:5]
    ]
    ambiguous = len(scored) > 1 and (top[0] - scored[1][0]) < 0.03
    return MatchResult(
        source, name, source_id, source_url, provider, top[1],
        top[2].get("canonical_family_name"),
        "ambiguous" if ambiguous else "identity_unresolved",
        min(0.8, round(top[0], 4)),
        "fuzzy_review_candidate",
        "Fuzzy candidates require manual approval and are excluded from scoring.",
        competing,
    )


def enrich_aa_rows(
    aa_rows: Iterable[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Attach identity fields to AA variants and return the family registry."""
    rows = [dict(row) for row in aa_rows]
    registry = build_family_registry(rows)
    for row in rows:
        identity = parse_model_identity(
            row.get("canonical_name") or row.get("model_name") or "",
            row.get("provider") or row.get("creator"),
            source="artificial_analysis",
            source_model_id=row.get("source_model_id") or row.get("model_id"),
            source_url=row.get("model_url"),
        )
        row.update({key: value for key, value in identity.items() if key != "comparison_key"})
        row["match_status"] = "matched_exact"
        row["match_confidence"] = 1.0
        if "aa_coding_proxy_legacy" not in row:
            row["aa_coding_proxy_legacy"] = row.get("coding_score")
        if row.get("aa_official_coding_index") is None:
            values = [
                row.get("terminalbench_v21"),
                row.get("scicode"),
            ]
            available = [float(value) for value in values if value is not None]
            row["aa_official_coding_index"] = (
                sum(available) / len(available) if len(available) == 2 else None
            )
    return rows, registry


def write_identity_audit(
    output_dir: Path,
    registry: Mapping[str, Any],
    matches: Iterable[MatchResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = [match.to_dict() for match in matches]
    unresolved = [
        row
        for row in audit
        if row["match_status"] not in MATCHED_STATUSES
        and row["match_status"] != "source_missing"
    ]
    (output_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "match_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "unresolved_matches.json").write_text(
        json.dumps(unresolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "source",
        "source_name",
        "source_model_id",
        "source_model_url",
        "provider",
        "candidate_family_id",
        "candidate_family_name",
        "match_status",
        "match_confidence",
        "match_method",
        "review_reason",
    ]
    with (output_dir / "match_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(audit)
