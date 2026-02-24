"""
gemini_advisor.py — Gemini-powered data-grounded conversational advisor.

PART 2 UPGRADE: Adds a conversational AI layer on TOP of the existing
deterministic advisor engine. The existing ranking system (advisor_engine.py)
is kept unchanged — this is an ADDITIONAL layer.

SAFETY:
  - Gemini receives ONLY structured data from the scored dataset
  - Gemini CANNOT access the internet
  - All responses must be JSON
  - Temperature is set to 0.2 for analytical accuracy
  - If Gemini fails → user gets a clear error, never a hallucination
  - Rankings are NEVER altered by the advisor

ANTI-HALLUCINATION:
  - Only top-15 models are sent as context (compact snapshot)
  - System prompt explicitly forbids inventing data
  - Only fields present in the dataset are referenced
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Response Cache (in-memory, 10-minute TTL)
# ──────────────────────────────────────────────────────────────

_response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 600  # 10 minutes

# Rate limiting: max 5 requests per user per minute
_rate_limiter: Dict[str, List[float]] = {}
RATE_LIMIT = 5
RATE_WINDOW = 60  # seconds


def _cache_key(query: str) -> str:
    """Generate a cache key from the query."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def _check_cache(query: str) -> Optional[Dict[str, Any]]:
    """Check if a cached response exists and is still valid."""
    key = _cache_key(query)
    if key in _response_cache:
        entry = _response_cache[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            logger.debug(f"Cache hit for advisor query")
            return entry["response"]
        else:
            del _response_cache[key]
    return None


def _store_cache(query: str, response: Dict[str, Any]) -> None:
    """Store a response in cache."""
    key = _cache_key(query)
    _response_cache[key] = {
        "timestamp": time.time(),
        "response": response,
    }
    # Evict old entries if cache grows too large
    if len(_response_cache) > 100:
        oldest_key = min(_response_cache, key=lambda k: _response_cache[k]["timestamp"])
        del _response_cache[oldest_key]


def _check_rate_limit(user_id: str = "default") -> bool:
    """
    Check if user has exceeded rate limit.
    Returns True if allowed, False if rate limited.
    """
    now = time.time()
    if user_id not in _rate_limiter:
        _rate_limiter[user_id] = []

    # Remove old entries
    _rate_limiter[user_id] = [
        t for t in _rate_limiter[user_id] if now - t < RATE_WINDOW
    ]

    if len(_rate_limiter[user_id]) >= RATE_LIMIT:
        return False

    _rate_limiter[user_id].append(now)
    return True


# ──────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the LLMDEX Data-Grounded Advisor.

STRICT RULES:
1. Use ONLY the provided dataset to answer questions. Do NOT invent, estimate, or hallucinate any data.
2. If the data needed to answer a question is missing from the provided dataset, state clearly: "This data is not available in the current dataset."
3. NEVER favor any specific provider or model — be analytically neutral.
4. NEVER alter, question, or override the existing rankings. Rankings are computed by a deterministic algorithm.
5. Answer analytically and concisely. Use data points to support every claim.
6. When comparing models, always reference specific metric values from the dataset.
7. If a user asks about models not in the dataset, say so clearly.
8. Keep responses focused and practical — help users make informed decisions.

You will receive a JSON snapshot of the top models in the LLMDEX index. Each model has:
- model_name: canonical name
- provider: company/org
- performance_rank, value_rank, efficiency_rank: positions in three leaderboards
- adjusted_performance: bias-corrected performance score (0-100)
- input_cost_per_1m, output_cost_per_1m: cost in USD per 1M tokens
- context_window: maximum context length in tokens
- coding_score: software engineering benchmark score
- reasoning_score: mathematical/logical reasoning score
- confidence_factor: data completeness (0-1, higher = more benchmark sources)

Return your response as JSON with this exact structure:
{
  "answer": "Your analytical response as a string",
  "referenced_models": ["model1", "model2"],
  "data_points_used": ["adjusted_performance", "input_cost_per_1m"]
}"""


# ──────────────────────────────────────────────────────────────
# Data Extraction
# ──────────────────────────────────────────────────────────────

def _load_dataset(index_path: Optional[str] = None) -> List[dict]:
    """Load the latest index data."""
    if index_path is None:
        index_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "index", "latest.json"
        )

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load index data: {e}")
        return []


def _extract_compact_snapshot(dataset: List[dict], top_n: int = 15) -> List[dict]:
    """
    Extract a compact top-N dataset snapshot for Gemini context.
    
    Only includes fields that Gemini needs — no raw benchmark breakdowns,
    no internal IDs, no data that could confuse the model.
    """
    # Sort by performance rank (best first)
    ranked = sorted(
        [d for d in dataset if d.get("performance_rank") is not None],
        key=lambda d: d.get("performance_rank", 9999),
    )[:top_n]

    snapshot = []
    for m in ranked:
        entry = {
            "model_name": m.get("canonical_name") or m.get("model_name") or "Unknown",
            "provider": m.get("provider") or "Unknown",
            "performance_rank": m.get("performance_rank"),
            "value_rank": m.get("value_rank"),
            "efficiency_rank": m.get("efficiency_rank"),
            "adjusted_performance": _round_safe(m.get("adjusted_performance")),
            "input_cost_per_1m": _round_safe(m.get("input_cost_per_1m")),
            "output_cost_per_1m": _round_safe(m.get("output_cost_per_1m")),
            "context_window": m.get("context_window"),
            "coding_score": _round_safe(m.get("coding_score")),
            "reasoning_score": _round_safe(m.get("reasoning_score")),
            "confidence_factor": _round_safe(m.get("confidence_factor")),
        }
        snapshot.append(entry)

    return snapshot


def _round_safe(val, decimals=2):
    """Round a value safely, handling None."""
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def generate_advisor_response(
    user_query: str,
    user_id: str = "default",
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a data-grounded AI advisor response.

    Args:
        user_query: The user's question about models
        user_id: User identifier for rate limiting
        index_path: Optional path to index data

    Returns:
        Dict with keys: answer, referenced_models, data_points_used, source
        On failure: Dict with error message
    """
    # Rate limit check
    if not _check_rate_limit(user_id):
        return {
            "answer": "Rate limit exceeded. Please wait a moment before asking another question. Maximum 5 queries per minute.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "rate_limit",
        }

    # Check cache
    cached = _check_cache(user_query)
    if cached:
        cached["source"] = "cache"
        return cached

    # Load and extract data
    dataset = _load_dataset(index_path)
    if not dataset:
        return {
            "answer": "Unable to load model data. The dataset may not be available yet.",
            "referenced_models": [],
            "data_points_used": [],
            "source": "error",
        }

    snapshot = _extract_compact_snapshot(dataset, top_n=15)

    # Build prompt with embedded data
    prompt = f"""USER QUESTION: {user_query}

DATASET SNAPSHOT (Top {len(snapshot)} models from LLMDEX):
{json.dumps(snapshot, indent=2)}

Answer the user's question using ONLY the data above. Follow all system rules."""

    # Call Gemini
    try:
        from utils.gemini_client import call_gemini
    except ImportError:
        logger.error("gemini_client not available")
        return _fallback_response()

    result = call_gemini(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        pool_type="advisor",
        temperature=0.2,
    )

    if result is None:
        return _fallback_response()

    # Validate response structure
    response = {
        "answer": result.get("answer", "Unable to generate a response."),
        "referenced_models": result.get("referenced_models", []),
        "data_points_used": result.get("data_points_used", []),
        "source": "gemini",
    }

    # Store in cache
    _store_cache(user_query, response)

    return response


def _fallback_response() -> Dict[str, Any]:
    """Return fallback when Gemini is unavailable."""
    return {
        "answer": "AI advisor temporarily unavailable. Please use the ranking filters and priority selector below to find the best models for your needs.",
        "referenced_models": [],
        "data_points_used": [],
        "source": "fallback",
    }
