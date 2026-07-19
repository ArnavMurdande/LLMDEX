"""Shared model-family inference for the pipeline and growth explorer.

Artificial Analysis adds new model IDs and reasoning-effort variants frequently.
Family membership therefore has to be inferred from the displayed model name
instead of relying on a hand-maintained ID allow-list.
"""

from __future__ import annotations

import re
from typing import Any, Tuple


_VARIANT_SUFFIXES = re.compile(
    r"\s*\((?=[^)]*(?:fallback|adaptive|thinking|reasoning|effort|"
    r"max|xhigh|high|medium|low|min))[^)]*\)\s*$",
    re.IGNORECASE,
)


def normalize_release_name(name: str) -> str:
    """Collapse effort/fallback variants to one public model release."""
    value = re.sub(r"\s+", " ", (name or "").strip())
    previous = None
    while value and previous != value:
        previous = value
        value = _VARIANT_SUFFIXES.sub("", value).strip()
    return value or (name or "Unknown model").strip()


def infer_model_family(name: str, provider: str | None = None) -> str:
    """Return a stable product family from a current model display name."""
    lower = (name or "").lower()
    provider_lower = (provider or "").lower()

    if "claude" in lower or "anthropic" in provider_lower:
        if "fable" in lower:
            return "Claude Fable"
        if "opus" in lower:
            return "Claude Opus"
        if "sonnet" in lower:
            return "Claude Sonnet"
        if "haiku" in lower:
            return "Claude Haiku"
        return "Claude"

    if "gemini" in lower or "google" in provider_lower:
        if "pro" in lower:
            return "Gemini Pro"
        if "flash" in lower:
            return "Gemini Flash"
        if "ultra" in lower:
            return "Gemini Ultra"
        if "nano" in lower:
            return "Gemini Nano"
        return "Gemini"

    if re.search(r"\bgpt[-\s]", lower) or "openai" in provider_lower:
        if "sol" in lower:
            return "GPT Sol"
        if "terra" in lower:
            return "GPT Terra"
        if "luna" in lower:
            return "GPT Luna"
        if "codex" in lower:
            return "GPT Codex"
        if "mini" in lower:
            return "GPT Mini"
        if re.search(r"\bo[134](?:\b|-)", lower):
            return "OpenAI o-series"
        return "GPT"

    rules = (
        ("deepseek", "DeepSeek"),
        ("qwen", "Qwen"),
        ("llama", "Llama"),
        ("grok", "Grok"),
        ("kimi", "Kimi"),
        ("minimax", "MiniMax"),
        ("glm", "GLM"),
        ("mistral", "Mistral"),
        ("mixtral", "Mistral"),
        ("gemma", "Gemma"),
        ("command", "Cohere Command"),
        ("nova", "Amazon Nova"),
        ("ernie", "ERNIE"),
        ("yi-", "Yi"),
    )
    for token, family in rules:
        if token in lower:
            return family

    if provider and provider_lower not in {"", "unknown", "other"}:
        return provider.strip()
    return (name or "Unknown").split()[0]


def infer_family_brand(family: str) -> str:
    """Group product families under the brand shown in the explorer dropdown."""
    known = (
        "Claude",
        "Gemini",
        "GPT",
        "OpenAI",
        "DeepSeek",
        "Qwen",
        "Llama",
        "Grok",
        "Kimi",
        "MiniMax",
        "GLM",
        "Mistral",
        "Gemma",
        "Amazon",
        "Cohere",
        "ERNIE",
        "Yi",
    )
    for brand in known:
        if family.startswith(brand):
            return "GPT" if brand == "OpenAI" else brand
    return family.split()[0] if family else "Other"


def release_sort_key(name: str) -> Tuple[Any, ...]:
    """Sort releases chronologically by version/date tokens, not benchmark score."""
    clean = normalize_release_name(name).lower()
    numbers = tuple(int(part) for part in re.findall(r"\d+", clean))
    # Names without versions stay stable and sort before numbered generations.
    return (1 if numbers else 0, numbers, clean)


def family_order_value(name: str) -> int:
    """Encode a sortable generation order for dataset rows."""
    numbers = [int(part) for part in re.findall(r"\d+", normalize_release_name(name))]
    value = 0
    for part in numbers[:4]:
        value = value * 10000 + min(part, 9999)
    return value or 1
