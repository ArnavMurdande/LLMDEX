"""
merge_data.py — Data merging, deduplication, and provenance tracking.

SAFETY DESIGN:
    - Uses WEIGHTED MEAN for multi-source aggregation (not max).
    - Keeps per-source provenance so users can audit every number.
    - Never overwrites upstream data — each stage writes its own artifact.
    - Missing values stay None/NaN throughout; never converted to 0.
    - Provider mapping is deterministic and documented.

UPGRADES (v2):
    - Model Identity Registry: resolves aliases to canonical names.
    - Per-source benchmark breakdown preserved in output.
    - Expanded data contract with model_id, aliases, confidence_score.

PIPELINE STAGES written by this module:
    1. raw_snapshot/   — exact copy of scraped data with timestamps
    2. cleaned/        — validated canonical dataset  
    3. index/          — scored and ranked dataset (final output)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from pipeline.scoring import (
    compute_source_weight,
    weighted_mean,
    score_dataset,
    SOURCE_CONFIDENCE_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Model Identity Registry
# ──────────────────────────────────────────────────────────────

_REGISTRY: Optional[Dict[str, Any]] = None
_ALIAS_MAP: Optional[Dict[str, str]] = None


def _load_registry() -> tuple[Dict[str, Any], Dict[str, str]]:
    """Load the model registry and build an alias→canonical_id lookup."""
    global _REGISTRY, _ALIAS_MAP
    if _REGISTRY is not None and _ALIAS_MAP is not None:
        return _REGISTRY, _ALIAS_MAP

    registry_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "model_registry.json"
    )
    registry_path = os.path.abspath(registry_path)

    if not os.path.exists(registry_path):
        logger.warning(f"Model registry not found at {registry_path}")
        _REGISTRY = {}
        _ALIAS_MAP = {}
        return _REGISTRY, _ALIAS_MAP

    with open(registry_path, "r", encoding="utf-8") as f:
        _REGISTRY = json.load(f)

    # Build alias→model_id lookup (normalize aliases for matching)
    _ALIAS_MAP = {}
    for model_id, entry in _REGISTRY.items():
        # Canonical name itself is an alias
        canonical = _normalize_for_matching(entry["canonical_name"])
        _ALIAS_MAP[canonical] = model_id
        for alias in entry.get("aliases", []):
            _ALIAS_MAP[_normalize_for_matching(alias)] = model_id

    logger.info(f"Registry loaded: {len(_REGISTRY)} models, {len(_ALIAS_MAP)} aliases")
    return _REGISTRY, _ALIAS_MAP


def _normalize_for_matching(name: str) -> str:
    """
    Normalize a model name for fuzzy matching against the registry.

    PART 4 UPGRADE: Enhanced normalization pipeline:
    - Case insensitive
    - Remove newlines (scraped artifacts)
    - Strip common suffixes: preview, adaptive, thinking, new, latest, speciale
    - Remove punctuation (except hyphens and dots in version numbers)
    - Collapse whitespace
    """
    name = name.lower().strip()
    # Remove newlines (scraped artifact)
    name = name.replace("\n", " ").replace("\r", "")

    # Strip common suffixes that don't change identity (iteratively)
    # Includes: thinking/reasoning modes, context window sizes (16k/32k/128k),
    # effort levels, date codes, and variant labels
    suffix_pattern = r"\s*[-]?\b(new|adaptive|latest|preview|thinking|reasoning|non-reasoning|non-thinking|speciale|high\s*effort|high|\d+k)\s*$"
    for _ in range(5):  # Multiple passes for stacked suffixes like "thinking-16k"
        cleaned = re.sub(suffix_pattern, "", name, flags=re.IGNORECASE).strip()
        if cleaned == name:
            break
        name = cleaned

    # Remove parenthetical content like "(Adaptive)" or "(high)"
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)

    # Remove punctuation except hyphens and dots (preserve version numbers)
    name = re.sub(r"[^\w\s.\-]", "", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


# Track unmatched aliases for diagnostics
_UNMATCHED_LOG: List[str] = []


def get_unmatched_aliases() -> List[str]:
    """Return list of unmatched model names from last resolution run."""
    return list(_UNMATCHED_LOG)


def resolve_model_identity(
    model_name: str,
) -> tuple[Optional[str], str, Optional[str], List[str]]:
    """
    Resolve a scraped model name to its canonical identity.

    PART 4 UPGRADE: Multi-stage fuzzy matching:
    1. Exact normalized match
    2. Parenthetical removal
    3. Suffix stripping (preview, thinking, etc.)
    4. Substring containment check
    5. Fallback with logging

    Returns: (model_id, canonical_name, provider, aliases)
    If not found in registry, logs the unmatched name.
    Never silently drops rows.
    """
    registry, alias_map = _load_registry()

    normalized = _normalize_for_matching(model_name)

    # Stage 1: Exact alias match
    if normalized in alias_map:
        model_id = alias_map[normalized]
        entry = registry[model_id]
        return (
            model_id,
            entry["canonical_name"],
            entry.get("provider"),
            entry.get("aliases", []),
        )

    # Stage 2: Try removing parenthetical content
    no_parens = re.sub(r"\s*\([^)]*\)\s*", " ", normalized).strip()
    no_parens = re.sub(r"\s+", " ", no_parens)
    if no_parens in alias_map:
        model_id = alias_map[no_parens]
        entry = registry[model_id]
        return (
            model_id,
            entry["canonical_name"],
            entry.get("provider"),
            entry.get("aliases", []),
        )

    # Stage 3: Strip version suffixes and try again
    stripped = re.sub(r"\s*\d{4}\s*$", "", no_parens).strip()  # Remove date codes like "0905"
    if stripped != no_parens and stripped in alias_map:
        model_id = alias_map[stripped]
        entry = registry[model_id]
        return (
            model_id,
            entry["canonical_name"],
            entry.get("provider"),
            entry.get("aliases", []),
        )

    # Stage 4: Fuzzy substring containment (longest match wins)
    best_match = None
    best_len = 0
    for alias_norm, mid in alias_map.items():
        # Check if the normalized name contains a known alias (or vice versa)
        if len(alias_norm) >= 4:  # Avoid matching very short aliases
            if alias_norm in normalized or normalized in alias_norm:
                if len(alias_norm) > best_len:
                    best_len = len(alias_norm)
                    best_match = mid

    if best_match:
        entry = registry[best_match]
        logger.debug(f"Fuzzy substring matched '{model_name}' → '{entry['canonical_name']}'")
        return (
            best_match,
            entry["canonical_name"],
            entry.get("provider"),
            entry.get("aliases", []),
        )

    # Stage 5: Advanced string similarity (difflib)
    import difflib
    # Look for close matches against known normalized aliases
    possible_matches = difflib.get_close_matches(normalized, alias_map.keys(), n=1, cutoff=0.85)
    if possible_matches:
        best_alias = possible_matches[0]
        model_id = alias_map[best_alias]
        entry = registry[model_id]
        logger.debug(f"Difflib fuzzy matched '{model_name}' → '{entry['canonical_name']}'")
        return (
            model_id,
            entry["canonical_name"],
            entry.get("provider"),
            entry.get("aliases", []),
        )

    # Stage 5: Fallback — log unmatched alias, never drop the row
    _UNMATCHED_LOG.append(model_name.replace("\n", " ").strip())
    logger.info(f"Unmatched model alias: '{model_name.replace(chr(10), ' ').strip()}'")
    return None, model_name.replace("\n", " ").strip(), None, []


# ──────────────────────────────────────────────────────────────
# Provider identification
# ──────────────────────────────────────────────────────────────

PROVIDER_PATTERNS: List[tuple[str, List[str]]] = [
    ("OpenAI", ["gpt", "o1-", "o3-", "o4-", "openai", "davinci", "chatgpt"]),
    ("Anthropic", ["claude", "anthropic"]),
    ("Google", ["gemini", "gemma", "google", "palm"]),
    ("Meta", ["llama", "meta-llama"]),
    ("Alibaba", ["qwen", "alibaba"]),
    ("DeepSeek", ["deepseek", "r1"]),
    ("Mistral AI", ["mistral", "mixtral"]),
    ("01.AI", ["yi-"]),
    ("Cohere", ["command", "cohere"]),
    ("Microsoft", ["phi-", "microsoft"]),
    ("Zhipu AI", ["glm", "chatglm"]),
    ("Amazon", ["amazon", "titan", "nova"]),
    ("Databricks", ["databricks", "dbrx"]),
    ("xAI", ["grok"]),
    ("Moonshot", ["kimi", "moonshot"]),
    ("MiniMax", ["minimax"]),
]


def identify_provider(model_name: str) -> Optional[str]:
    """
    Map a model name to its provider.
    Returns None (not "Unknown") if no match found.
    """
    name_lower = model_name.lower()
    for provider, patterns in PROVIDER_PATTERNS:
        if any(p in name_lower for p in patterns):
            return provider
    return None


def slugify_model_name(name: str) -> str:
    """Create a stable, lowercase slug for model deduplication.
    
    Checks cross-source alias table for both space-separated (AA),
    hyphen-separated (LMSYS), and dot-stripped variants.
    """
    slug = name.lower().strip()
    slug = slug.replace("\n", " ").replace("\r", "")
    # Remove parenthetical annotations
    slug = re.sub(r"\s*\([^)]*\)\s*", " ", slug)
    slug_cleaned = re.sub(r"\s+", " ", slug).strip()
    
    # Check alias: space-separated (AA: "gemini 2.5 pro")
    if slug_cleaned in _CROSS_SOURCE_ALIASES:
        return _CROSS_SOURCE_ALIASES[slug_cleaned]
    
    # Check alias: hyphen-separated (LMSYS: "gemini-2.5-pro")
    slug_h = slug_cleaned.replace(" ", "-")
    if slug_h in _CROSS_SOURCE_ALIASES:
        return _CROSS_SOURCE_ALIASES[slug_h]
    
    # Check alias: dots stripped + hyphens (LMSYS after dot-removal: "gemini-25-pro")
    slug_nd = re.sub(r"[^\w\s-]", "", slug_cleaned)
    slug_nd_h = re.sub(r"\s+", "-", slug_nd).strip("-")
    if slug_nd_h in _CROSS_SOURCE_ALIASES:
        return _CROSS_SOURCE_ALIASES[slug_nd_h]
    
    # Default: strip non-word, join with hyphens
    return slug_nd_h if slug_nd_h else slug_h


# ──────────────────────────────────────────────────────────────
# Cross-source aliases: both AA and LMSYS names → shared slug
# ──────────────────────────────────────────────────────────────
_CROSS_SOURCE_ALIASES: Dict[str, str] = {
    # Claude Opus
    "claude opus 4.6": "claude-opus-46",
    "claude-opus-4.6": "claude-opus-46", "claude-opus-4-6": "claude-opus-46",
    "claude-opus-4.6-thinking": "claude-opus-46", "claude-opus-4-6-thinking": "claude-opus-46",
    "claude opus 4.5": "claude-opus-45",
    "claude-opus-4.5-20251101": "claude-opus-45", "claude-opus-4-5-20251101": "claude-opus-45",
    "claude-opus-4.5-20251101-thinking-32k": "claude-opus-45", "claude-opus-4-5-20251101-thinking-32k": "claude-opus-45",
    "claude opus 4.1": "claude-opus-41", "claude 4.1 opus": "claude-opus-41",
    "claude-opus-4.1-20250805": "claude-opus-41", "claude-opus-4-1-20250805": "claude-opus-41",
    "claude-opus-4.1-20250805-thinking-16k": "claude-opus-41", "claude-opus-4-1-20250805-thinking-16k": "claude-opus-41",
    "claude 4.1 opus reasoning": "claude-opus-41",
    # Claude Opus 4 (base, no version suffix)
    "claude 4 opus": "claude-opus-4", "claude opus 4": "claude-opus-4",
    "claude-opus-4-20250514": "claude-opus-4", "claude-opus-4.0-20250514": "claude-opus-4",
    "claude-opus-4-20250514-thinking-16k": "claude-opus-4", "claude-opus-4.0-20250514-thinking-16k": "claude-opus-4",
    "claude 4 opus non-reasoning": "claude-opus-4",
    # Claude Sonnet
    "claude sonnet 4.6": "claude-sonnet-46",
    "claude-sonnet-4.6": "claude-sonnet-46", "claude-sonnet-4-6": "claude-sonnet-46",
    "claude sonnet 4.5": "claude-sonnet-45",
    "claude-sonnet-4.5-20250929": "claude-sonnet-45", "claude-sonnet-4-5-20250929": "claude-sonnet-45",
    "claude-sonnet-4.5-20250929-thinking-32k": "claude-sonnet-45", "claude-sonnet-4-5-20250929-thinking-32k": "claude-sonnet-45",
    # Claude Sonnet 4 (base)
    "claude sonnet 4": "claude-sonnet-4",
    "claude-sonnet-4-20250514": "claude-sonnet-4", "claude-sonnet-4.0-20250514": "claude-sonnet-4",
    "claude-sonnet-4-20250514-thinking-32k": "claude-sonnet-4", "claude-sonnet-4.0-20250514-thinking-32k": "claude-sonnet-4",
    # Claude 3.x Sonnet
    "claude 3.7 sonnet": "claude-37-sonnet",
    "claude-3.7-sonnet-20250219": "claude-37-sonnet", "claude-3-7-sonnet-20250219": "claude-37-sonnet",
    "claude 3.5 sonnet": "claude-35-sonnet",
    "claude-3.5-sonnet-20241022": "claude-35-sonnet", "claude-3-5-sonnet-20241022": "claude-35-sonnet",
    # Claude Haiku
    "claude 4.5 haiku": "claude-45-haiku",
    "claude-haiku-4.5-20251001": "claude-45-haiku", "claude-haiku-4-5-20251001": "claude-45-haiku",
    "claude 3.5 haiku": "claude-35-haiku",
    "claude-3.5-haiku-20241022": "claude-35-haiku", "claude-3-5-haiku-20241022": "claude-35-haiku",
    "claude 3 opus": "claude-3-opus", "claude-3-opus-20240229": "claude-3-opus",
    # Gemini
    "gemini 3 pro": "gemini-3-pro", "gemini-3-pro": "gemini-3-pro",
    "gemini 3 pro preview": "gemini-3-pro",
    "gemini 3.1 pro preview": "gemini-31-pro-preview",
    "gemini 3 flash": "gemini-3-flash", "gemini-3-flash": "gemini-3-flash",
    "gemini 2.5 pro": "gemini-25-pro", "gemini-2.5-pro": "gemini-25-pro", "gemini-25-pro": "gemini-25-pro",
    "gemini 2.5 flash": "gemini-25-flash", "gemini-2.5-flash": "gemini-25-flash", "gemini-25-flash": "gemini-25-flash",
    "gemini-2.5-flash-preview-09-2025": "gemini-25-flash",
    "gemini 2.0 flash": "gemini-20-flash", "gemini-2.0-flash-001": "gemini-20-flash", "gemini-20-flash-001": "gemini-20-flash",
    "gemini 1.5 pro": "gemini-15-pro", "gemini-1.5-pro-002": "gemini-15-pro", "gemini-15-pro-002": "gemini-15-pro",
    "gemini-1.5-pro-001": "gemini-15-pro", "gemini-15-pro-001": "gemini-15-pro",
    "gemini 1.5 flash": "gemini-15-flash", "gemini-1.5-flash-002": "gemini-15-flash", "gemini-15-flash-002": "gemini-15-flash",
    "gemini-1.5-flash-001": "gemini-15-flash", "gemini-15-flash-001": "gemini-15-flash",
    # GPT
    "gpt-5.2": "gpt-52", "gpt-52": "gpt-52", "gpt-5.2-chat-latest-20260210": "gpt-52", "gpt-52-chat-latest-20260210": "gpt-52",
    "gpt-5.2-high": "gpt-52", "gpt-52-high": "gpt-52",
    "gpt-5.1": "gpt-51", "gpt-51": "gpt-51", "gpt-5.1-high": "gpt-51", "gpt-51-high": "gpt-51",
    "gpt-5": "gpt-5", "gpt-5-high": "gpt-5", "gpt-5-chat": "gpt-5",
    "gpt-5 mini": "gpt-5-mini", "gpt-5-mini-high": "gpt-5-mini",
    "gpt-5 nano": "gpt-5-nano", "gpt-5-nano-high": "gpt-5-nano",
    "gpt-4.5 preview": "gpt-45-preview", "gpt-4.5-preview-2025-02-27": "gpt-45-preview", "gpt-45-preview-2025-02-27": "gpt-45-preview",
    "gpt-4.1": "gpt-41", "gpt-4.1-2025-04-14": "gpt-41", "gpt-41-2025-04-14": "gpt-41",
    "gpt-4.1 mini": "gpt-41-mini", "gpt-4.1-mini-2025-04-14": "gpt-41-mini", "gpt-41-mini-2025-04-14": "gpt-41-mini",
    "gpt-4.1 nano": "gpt-41-nano", "gpt-4.1-nano-2025-04-14": "gpt-41-nano", "gpt-41-nano-2025-04-14": "gpt-41-nano",
    "gpt-4o": "gpt-4o", "gpt-4o-2024-05-13": "gpt-4o", "chatgpt-4o-latest-20250326": "gpt-4o",
    "gpt-4o mini": "gpt-4o-mini", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4 turbo": "gpt-4-turbo", "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    # DeepSeek
    "deepseek r1": "deepseek-r1", "deepseek-r1": "deepseek-r1",
    "deepseek-r1-0528": "deepseek-r1",
    "deepseek v3": "deepseek-v3", "deepseek-v3": "deepseek-v3",
    "deepseek v3.1": "deepseek-v31", "deepseek-v3.1": "deepseek-v31", "deepseek-v31": "deepseek-v31",
    "deepseek-v3.1-thinking": "deepseek-v31", "deepseek-v3.1-terminus": "deepseek-v31",
    "deepseek-v3.1-terminus-thinking": "deepseek-v31",
    "deepseek v3.2": "deepseek-v32", "deepseek-v3.2": "deepseek-v32", "deepseek-v32": "deepseek-v32",
    "deepseek-v3.2-thinking": "deepseek-v32",
    "deepseek v3.2 speciale": "deepseek-v32-speciale", "deepseek-v32-exp": "deepseek-v32-speciale",
    "deepseek-v3.2-exp": "deepseek-v32-speciale", "deepseek-v3.2-exp-thinking": "deepseek-v32-speciale",
    # Grok
    "grok 4.1": "grok-41", "grok-4.1": "grok-41", "grok-41": "grok-41",
    "grok-4.1-thinking": "grok-41", "grok-41-thinking": "grok-41",
    "grok-4.1-fast-reasoning": "grok-41-fast", "grok-4.1-fast": "grok-41-fast",
    "grok 4": "grok-4", "grok-4-0709": "grok-4",
    "grok-4-fast-chat": "grok-4-fast", "grok-4-fast-reasoning": "grok-4-fast",
    "grok 3 mini reasoning": "grok-3-mini", "grok-3-mini-high": "grok-3-mini",
    "grok 3": "grok-3", "grok-3-preview-02-24": "grok-3",
    # Qwen
    "qwen3 max": "qwen3-max", "qwen3 max thinking": "qwen3-max", "qwen3-max-preview": "qwen3-max",
    "qwen3-max-2025-09-23": "qwen3-max",
    "qwen3.5-397b-a17b": "qwen35-397b", "qwen35-397b-a17b": "qwen35-397b",
    "qwen3.5 397b a17b": "qwen35-397b",
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b", "qwen3-235b-a22b-instruct": "qwen3-235b",
    "qwen3-235b-a22b-no-thinking": "qwen3-235b",
    "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b", "qwen3-vl-235b-a22b-thinking": "qwen3-vl-235b",
    "qwen3-next-80b-a3b-instruct": "qwen3-next-80b",
    # GLM
    "glm-5": "glm-5", "glm-4.7": "glm-47", "glm-47": "glm-47",
    "glm-4.6": "glm-46", "glm-46": "glm-46",
    "glm-4.5": "glm-45", "glm-45": "glm-45", "glm-4.5-air": "glm-45-air", "glm-45-air": "glm-45-air",
    # Mistral
    "mistral large 3": "mistral-large-3", "mistral-large-3": "mistral-large-3",
    "mistral medium": "mistral-medium", "mistral-medium-2508": "mistral-medium",
    # o-series
    "o3": "o3", "o3-2025-04-16": "o3", "o3 mini": "o3-mini", "o3-mini": "o3-mini",
    "o4 mini": "o4-mini", "o4-mini-2025-04-16": "o4-mini",
    "o1": "o1", "o1-2024-12-17": "o1", "o1 mini": "o1-mini", "o1-mini": "o1-mini",
    "o1 preview": "o1-preview", "o1-preview": "o1-preview",
    # Others
    "llama 4 maverick": "llama-4-maverick", "llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "llama 4 scout": "llama-4-scout", "llama-4-scout-17b-16e-instruct": "llama-4-scout",
    "command a": "command-a", "command-a-03-2025": "command-a",
    "phi-4": "phi-4",
    "minimax-m2.5": "minimax-m25", "minimax-m25": "minimax-m25",
    "nova micro": "nova-micro", "amazon-nova-micro-v10": "nova-micro",
    "nova lite": "nova-lite", "amazon-nova-lite-v10": "nova-lite",
    "nova pro": "nova-pro", "amazon-nova-pro-v10": "nova-pro",
    "amazon-nova-experimental-chat-12-10": "nova-experimental",
    "kimi k2.5": "kimi-k25", "kimi-k2.5-thinking": "kimi-k25", "kimi-k25-thinking": "kimi-k25",
    "kimi k2.5 instant": "kimi-k25-instant", "kimi-k2.5-instant": "kimi-k25-instant", "kimi-k25-instant": "kimi-k25-instant",
    "kimi-k2-thinking-turbo": "kimi-k2-turbo", "kimi-k2-0711-preview": "kimi-k2", "kimi-k2-0905-preview": "kimi-k2",
    "ernie 5.0": "ernie-50", "ernie-5.0-0110": "ernie-50", "ernie-50-0110": "ernie-50",
    "ernie 5.0 preview": "ernie-50-preview", "ernie-5.0-preview-1203": "ernie-50-preview", "ernie-50-preview-1203": "ernie-50-preview",
    "ernie-5.0-preview-1022": "ernie-50-preview",
    "dola-seed-2.0-preview": "dola-seed-20", "dola-seed-20-preview": "dola-seed-20",
    "jamba 1.5 large": "jamba-15-large", "jamba-15-large": "jamba-15-large",
    "intellect-3": "intellect-3",
    # Hunyuan
    "hunyuan-vision-1.5-thinking": "hunyuan-vision-15",
}

# ──────────────────────────────────────────────────────────────
# Model Family Mapping (deterministic, not inferred from runs)
# ──────────────────────────────────────────────────────────────

FAMILY_MAP: Dict[str, tuple] = {
    # Claude Sonnet family
    "claude_sonnet_4":    ("Claude Sonnet", 1),
    "claude_sonnet_4_5":  ("Claude Sonnet", 2),
    "claude_sonnet_4_6":  ("Claude Sonnet", 3),
    # Claude Opus family
    "claude_opus_4_1":    ("Claude Opus", 1),
    "claude_opus_4_5":    ("Claude Opus", 2),
    "claude_opus_4_6":    ("Claude Opus", 3),
    # Claude Haiku family
    "claude_haiku_4_5":   ("Claude Haiku", 1),
    # GPT family
    "gpt_5":              ("GPT", 1),
    "gpt_5_pro":          ("GPT", 2),
    "gpt_5_1":            ("GPT", 3),
    "gpt_5_mini":         ("GPT Mini", 1),
    "gpt_5_medium":       ("GPT Medium", 1),
    "gpt_5_high":         ("GPT High", 1),
    "gpt_5_1_high":       ("GPT High", 2),
    "gpt_5_2_high":       ("GPT High", 3),
    "gpt_5_1_codex":      ("GPT Codex", 1),
    "gpt_5_2_codex":      ("GPT Codex", 2),
    # Gemini family
    "gemini_3_pro":       ("Gemini Pro", 1),
    "gemini_3_flash":     ("Gemini Flash", 1),
    # DeepSeek family
    "deepseek_v3_2":      ("DeepSeek V", 1),
    "deepseek_r1":        ("DeepSeek R", 1),
    # Qwen family
    "qwen_3_5":           ("Qwen", 1),
    # xAI
    "grok_4":             ("Grok", 1),
    # Zhipu
    "glm_4_5":            ("GLM", 1),
    "glm_4_6":            ("GLM", 2),
    "glm_5":              ("GLM", 3),
    # Moonshot
    "kimi_k2":            ("Kimi", 1),
    "kimi_k2_5":          ("Kimi", 2),
    # MiniMax
    "minimax_m2_5":       ("MiniMax", 1),
}


def _assign_model_families(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign model_family and family_order to each row using FAMILY_MAP.

    Uses the model_id field (from registry resolution) to look up
    the deterministic family mapping. Models not in the map get None.
    """
    families = []
    orders = []
    for _, row in df.iterrows():
        model_id = row.get("model_id")
        if model_id and model_id in FAMILY_MAP:
            fam, order = FAMILY_MAP[model_id]
            families.append(fam)
            orders.append(order)
        else:
            families.append(None)
            orders.append(None)
    df["model_family"] = families
    df["family_order"] = orders
    return df


# ──────────────────────────────────────────────────────────────
# Data merging with provenance tracking
# ──────────────────────────────────────────────────────────────

def merge_rows_with_provenance(
    rows: List[dict],
) -> List[dict]:
    """
    Merge rows from multiple sources into one row per model,
    using weighted mean aggregation and preserving provenance.

    UPGRADES (v2):
    1. Uses Model Identity Registry to resolve aliases.
    2. Groups by model_id (from registry) or slug (fallback).
    3. Preserves per-source benchmark_breakdown.
    4. Adds model_id, aliases, confidence_score to output.
    """
    # Group by registry model_id or slug
    groups: Dict[str, List[dict]] = {}
    group_identities: Dict[str, tuple] = {}  # group_key -> (model_id, canonical, provider, aliases)

    for row in rows:
        raw_name = row.get("model_name", "")
        if not raw_name or not raw_name.strip():
            continue

        model_id, canonical, reg_provider, aliases = resolve_model_identity(raw_name)

        if model_id:
            group_key = model_id
        else:
            group_key = slugify_model_name(raw_name)

        groups.setdefault(group_key, []).append(row)

        # Store identity (first match wins for the group)
        if group_key not in group_identities:
            group_identities[group_key] = (model_id, canonical, reg_provider, aliases)

    merged: List[dict] = []

    for group_key, group in groups.items():
        model_id, canonical_name, reg_provider, aliases = group_identities.get(
            group_key, (None, group_key, None, [])
        )

        # ── Provider resolution ──
        providers = [r.get("provider") for r in group if r.get("provider")]
        provider = reg_provider or (providers[0] if providers else identify_provider(canonical_name))

        # ── Name priority: AA names > registry > longest name ──
        # AA names are human-readable (e.g., "Claude Opus 4.6 (max)")
        # LMSYS names are slugs (e.g., "claude-opus-4-6-thinking")
        if not canonical_name or canonical_name == group_key:
            # Prefer AA source name first
            aa_names = [
                r.get("model_name", "").replace("\n", " ").strip()
                for r in group
                if r.get("source") == "Artificial Analysis" and r.get("model_name")
            ]
            if aa_names:
                canonical_name = max(aa_names, key=lambda n: len(n))
            else:
                canonical_name = max(
                    (r.get("model_name", "").replace("\n", " ").strip() for r in group),
                    key=lambda n: len(n),
                )

        # ── Collect all raw names for alias tracking ──
        raw_names = sorted(set(
            r.get("model_name", "").replace("\n", " ").strip()
            for r in group
            if r.get("model_name")
        ))

        # ── Provenance tracking ──
        provenance: List[dict] = []
        for r in group:
            provenance.append({
                "source": r.get("source", "unknown"),
                "scraped_at": r.get("scraped_at"),
                "confidence": r.get("confidence", 1.0),
                "raw_name": r.get("model_name", ""),
                "raw_scores": {
                    k: r.get(k)
                    for k in [
                        "intelligence_score", "coding_score", "reasoning_score",
                        "multimodal_score", "arena_elo", "arena_votes",
                        "input_cost_per_1m", "output_cost_per_1m", "blended_cost_per_1m",
                        "latency_seconds", "latency_first_token",
                        "latency_p5", "latency_p25", "latency_p75", "latency_p95",
                        "total_response_time", "reasoning_time",
                        "tokens_per_second", "speed_p5", "speed_p25", "speed_p75", "speed_p95",
                        "context_window", "gdpval", "terminalbench_hard", "tau2", "lcr", 
                        "omniscience", "omniscience_hallucination",
                        "hle", "gpqa", "scicode", "ifbench", "aime25", "critpt", "mmmu_pro",
                        "livecodebench"
                    ]
                    if r.get(k) is not None
                },
            })

        # ── Build per-source benchmark breakdown ──
        benchmark_breakdown: Dict[str, dict] = {}
        for r in group:
            source = r.get("source", "unknown")
            scores = {}
            for k in [
                "intelligence_score", "coding_score", "reasoning_score",
                "multimodal_score", "arena_elo",
                "gdpval", "terminalbench_hard", "tau2", "lcr", "omniscience", 
                "omniscience_hallucination",
                "hle", "gpqa", "scicode", "ifbench", "aime25", "critpt", "mmmu_pro",
                "livecodebench"
            ]:
                val = r.get(k)
                if val is not None:
                    scores[k] = val
            if scores:
                # If source already seen, merge (keep first)
                if source not in benchmark_breakdown:
                    benchmark_breakdown[source] = scores
                else:
                    for k, v in scores.items():
                        if k not in benchmark_breakdown[source]:
                            benchmark_breakdown[source][k] = v

        # ── Weighted mean aggregation for each numeric field ──
        numeric_fields = [
            "intelligence_score", "coding_score", "reasoning_score",
            "multimodal_score", "arena_elo",
            "input_cost_per_1m", "output_cost_per_1m", "blended_cost_per_1m",
            "latency_seconds", "latency_first_token",
            "latency_p5", "latency_p25", "latency_p75", "latency_p95",
            "total_response_time", "reasoning_time",
            "tokens_per_second", "speed_p5", "speed_p25", "speed_p75", "speed_p95",
            "gdpval", "terminalbench_hard", "tau2", "lcr", "omniscience", 
            "omniscience_hallucination",
            "hle", "gpqa", "scicode", "ifbench", "aime25", "critpt", "mmmu_pro",
            "livecodebench"
        ]

        aggregated = {}
        for field in numeric_fields:
            values = []
            weights = []
            for r in group:
                val = r.get(field)
                if val is not None:
                    src = r.get("source", "unknown")
                    conf = r.get("confidence", 1.0)
                    values.append(float(val))
                    weights.append(compute_source_weight(src, conf))

            result = weighted_mean(values, weights)
            aggregated[field] = round(result, 4) if result is not None else None

        # ── Non-numeric fields: take first non-None ──
        context_windows = [r.get("context_window") for r in group if r.get("context_window") is not None]
        context_window = max(context_windows) if context_windows else None

        open_source_flags = [r.get("open_source") for r in group if r.get("open_source") is not None]
        open_source = open_source_flags[0] if open_source_flags else None

        # Arena votes: take the max
        arena_votes_list = [r.get("arena_votes") for r in group if r.get("arena_votes") is not None]
        arena_votes = max(arena_votes_list) if arena_votes_list else None

        # License: take first non-None
        license_list = [r.get("license_type") for r in group if r.get("license_type")]
        license_type = license_list[0] if license_list else None

        # Creator: take first non-None
        creator_list = [r.get("creator") for r in group if r.get("creator")]
        creator = creator_list[0] if creator_list else None

        sources = sorted(set(r.get("source", "unknown") for r in group))

        # ── Data tier: determines ranking priority ──
        has_aa = "Artificial Analysis" in sources
        has_lmsys = "LMSYS Chatbot Arena" in sources
        if has_aa and has_lmsys:
            data_tier = 1   # Best: cross-referenced
        elif has_aa:
            data_tier = 2   # Good: AA benchmarks + cost
        else:
            data_tier = 3   # Limited: LMSYS ELO only

        # ── Confidence score: mean of per-row confidence weighted by source ──
        conf_values = [r.get("confidence", 1.0) for r in group]
        confidence_score = round(sum(conf_values) / len(conf_values), 4) if conf_values else 1.0

        merged.append({
            "model_id": model_id,
            "model_slug": slugify_model_name(canonical_name),
            "canonical_name": canonical_name,
            "model_name": canonical_name,  # backward compat
            "aliases": aliases or raw_names,
            "provider": provider,
            "context_window": context_window,
            "open_source": open_source,
            "arena_votes": arena_votes,
            "license_type": license_type,
            "creator": creator,
            **aggregated,
            "sources": sources,
            "source_count": len(sources),
            "data_tier": data_tier,
            "aggregation_method": "weighted_mean",
            "provenance": provenance,
            "benchmark_breakdown": benchmark_breakdown,
            "confidence_score": confidence_score,
        })

    return merged


# ──────────────────────────────────────────────────────────────
# Dataset I/O with archival
# ──────────────────────────────────────────────────────────────

def save_dataset_layer(
    data: List[dict] | pd.DataFrame,
    layer: str,
    base_dir: str,
    snapshot_date: str,
) -> str:
    """
    Save a dataset to a named layer directory with date versioning.

    Structure:
        data/{layer}/latest.json
        data/{layer}/latest.csv
        data/{layer}/archive/{snapshot_date}.json

    Returns the path to latest.json.
    """
    layer_dir = os.path.join(base_dir, layer)
    archive_dir = os.path.join(layer_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        df = data
        records = df.to_dict(orient="records")
    else:
        records = data
        df = pd.DataFrame(records)

    # Sanitize NaN → None for JSON safety.
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    records = _sanitize(records)

    # Add metadata to each record
    for r in records:
        r["snapshot_date"] = snapshot_date

    # ── Write latest ──
    json_path = os.path.join(layer_dir, "latest.json")
    csv_path = os.path.join(layer_dir, "latest.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    # For CSV, flatten complex fields
    df_flat = df.copy()
    for col in df_flat.columns:
        if df_flat[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_flat[col] = df_flat[col].apply(lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x)
    df_flat.to_csv(csv_path, index=False)

    # ── Archive ──
    archive_path = os.path.join(archive_dir, f"{snapshot_date}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info(f"Saved {len(records)} rows to {layer}/ (archived as {snapshot_date})")
    return json_path


def process_and_save(
    scrape_results: List[dict],
    snapshot_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Main processing entry point.

    Pipeline stages:
        1. Save raw snapshot
        2. Merge and deduplicate (with registry resolution)
        3. Validate (done externally, data arrives pre-validated)
        4. Score and rank
        5. Save cleaned + index layers

    Returns the final scored DataFrame, or None on failure.
    """
    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    base_dir = os.path.abspath(base_dir)

    if not scrape_results:
        logger.error("No data to process.")
        return None

    # ── Stage 1: Raw snapshot ──
    save_dataset_layer(scrape_results, "raw_snapshot", base_dir, snapshot_date)

    # ── Stage 2: Merge with provenance + registry resolution ──
    merged = merge_rows_with_provenance(scrape_results)
    logger.info(f"Merged into {len(merged)} unique models from {len(scrape_results)} raw rows")

    # ── Stage 3: Save cleaned ──
    save_dataset_layer(merged, "cleaned", base_dir, snapshot_date)

    # ── Stage 4: Score and rank ──
    df = pd.DataFrame(merged)
    df = score_dataset(df)

    # Sort by data_tier first (1=best → 3=limited), then by composite_index
    # This ensures AA models (tier 1-2) always rank above LMSYS-only (tier 3)
    df = df.sort_values(
        by=["data_tier", "composite_index", "performance_index"],
        ascending=[True, False, False],
        na_position="last",
    )

    # ── Stage 4b: Assign model family + generation order ──
    df = _assign_model_families(df)

    # ── Stage 5: Save index ──
    # Drop provenance from index layer (it's in cleaned/)
    index_df = df.drop(columns=["provenance"], errors="ignore")
    index_df["last_updated"] = snapshot_date

    save_dataset_layer(index_df, "index", base_dir, snapshot_date)

    # ── Stage 6 (PART 8): Save history snapshot for time-series ──
    _save_history_snapshot(df, base_dir, snapshot_date)

    # ── Also save to legacy locations for backward compatibility ──
    legacy_csv = os.path.join(base_dir, "models.csv")
    legacy_json = os.path.join(base_dir, "models.json")

    index_flat = index_df.copy()
    for col in index_flat.columns:
        if index_flat[col].apply(lambda x: isinstance(x, (list, dict))).any():
            index_flat[col] = index_flat[col].apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x
            )
    index_flat.to_csv(legacy_csv, index=False)
    index_df.to_json(legacy_json, orient="records", indent=2, default_handler=str)

    # ── Log unmatched aliases ──
    from pipeline.merge_data import get_unmatched_aliases
    unmatched = get_unmatched_aliases()
    if unmatched:
        logger.warning(f"Unmatched aliases ({len(unmatched)}): {unmatched[:10]}...")

    logger.info(
        f"Pipeline complete: {len(df)} models, "
        f"{int(df['composite_index'].notna().sum())} ranked"
    )
    return df


def _save_history_snapshot(df: pd.DataFrame, base_dir: str, snapshot_date: str) -> None:
    """
    PART 8: Save a daily snapshot for historical time-series analysis.

    Stores compact per-model data at:
        data/history/YYYY-MM-DD.json

    Tracked fields:
    - performance_rank, value_rank, efficiency_rank
    - performance_index, adjusted_performance
    - composite_index, efficiency_score
    - input_cost_per_1m, output_cost_per_1m
    - confidence_factor
    """
    history_dir = os.path.join(base_dir, "history")
    os.makedirs(history_dir, exist_ok=True)

    snapshot = []
    for _, row in df.iterrows():
        model_name = row.get("canonical_name") or row.get("model_name") or "Unknown"
        record = {
            "model_name": model_name,
            "model_id": row.get("model_id"),
            "provider": row.get("provider"),
            "performance_index": _safe_val(row.get("performance_index")),
            "adjusted_performance": _safe_val(row.get("adjusted_performance")),
            "composite_index": _safe_val(row.get("composite_index")),
            "efficiency_score": _safe_val(row.get("efficiency_score")),
            "performance_rank": _safe_int(row.get("performance_rank")),
            "value_rank": _safe_int(row.get("value_rank")),
            "efficiency_rank": _safe_int(row.get("efficiency_rank")),
            "input_cost_per_1m": _safe_val(row.get("input_cost_per_1m")),
            "output_cost_per_1m": _safe_val(row.get("output_cost_per_1m")),
            "confidence_factor": _safe_val(row.get("confidence_factor")),
            "snapshot_date": snapshot_date,
        }
        snapshot.append(record)

    snapshot_path = os.path.join(history_dir, f"{snapshot_date}.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info(f"History snapshot saved: {snapshot_path} ({len(snapshot)} models)")


def _safe_val(v):
    """Convert to float or None, handling NaN/NaT."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    """Convert to int or None."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # Smoke test with minimal data
    test_data = [
        {
            "model_name": "GPT-4o",
            "source": "LiveBench",
            "scraped_at": "2025-01-01T00:00:00+00:00",
            "intelligence_score": 88.5,
            "input_cost_per_1m": 2.50,
            "output_cost_per_1m": 10.00,
            "confidence": 0.95,
        },
        {
            "model_name": "gpt-4o",
            "source": "Vellum",
            "scraped_at": "2025-01-01T00:00:00+00:00",
            "input_cost_per_1m": 2.50,
            "output_cost_per_1m": 10.00,
            "latency_seconds": 0.8,
            "confidence": 0.9,
        },
        {
            "model_name": "Claude 3.5 Sonnet",
            "source": "LiveBench",
            "scraped_at": "2025-01-01T00:00:00+00:00",
            "intelligence_score": 85.0,
            "input_cost_per_1m": 3.00,
            "output_cost_per_1m": 15.00,
            "confidence": 0.95,
        },
    ]
    from utils.logger import setup_logging
    setup_logging()
    process_and_save(test_data)