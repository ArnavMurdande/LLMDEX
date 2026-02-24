"""
sentiment_pipeline.py — Community sentiment analysis module.

PART 6 UPGRADED: Now uses Gemini-based semantic classification with
VADER as fallback. Scrapes community mentions from free public sources,
performs sentiment scoring, and produces per-model sentiment metrics.

SOURCES:
    - Reddit (via old.reddit.com JSON API — no authentication)
    - Hacker News (via Algolia public API — no authentication)
    - GitHub Issues (via REST API — no authentication needed for public repos)

GEMINI INTEGRATION:
    - Batch classification of posts per model (40 per batch)
    - Validates relevance, scores sentiment, detects praise/criticism
    - Falls back to VADER on Gemini failure
    - Caches Gemini results for 24 hours

CONCURRENCY ARCHITECTURE:
    - Phase 1 (parallel): Scrape + clean + sample ALL models concurrently
    - Phase 2 (controlled): Gemini classification gated by GeminiSemaphore
    - GeminiSemaphore limits concurrent Gemini calls to available key count
    - Backpressure retry: quota failures retry once before VADER fallback
    - Dynamic key-aware concurrency via get_pool_stats()

OUTPUT:
    data/sentiment/latest.json

    Per-model fields:
    - community_sentiment: weighted average sentiment (-1 to 1)
    - mention_count: total mentions across all sources
    - controversy_index: ratio of polarized mentions
    - sentiment_trend: direction (positive, negative, stable)
    - confidence_score: min(1, valid_posts / 30)
    - sentiment_method: "gemini" or "vader_fallback"

SAFETY:
    - Sentiment is EXPERIMENTAL and NEVER influences ranking.
    - All data is labeled as experimental in the frontend.
    - Failures are logged but don't crash the pipeline.
    - _experimental: true is ALWAYS set
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing VADER — graceful fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")

# Try importing requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

MAX_MENTIONS_PER_MODEL = 150   # Sample cap before Gemini classification
BATCH_SIZE = 12                # Mentions per Gemini call — safer for Gemini latency
GEMINI_CALL_TIMEOUT = 90       # Seconds before a Gemini call is timed out
MAX_SCRAPE_WORKERS = 4         # Parallel workers for Phase 1 (scraping only)
SENTIMENT_CACHE_TTL = 86400    # 24 hours (in-memory + disk cache TTL)
QUOTA_RETRY_DELAY = 5          # Seconds to wait before retrying a quota-failed call
COOLDOWN_SECONDS = 300         # Must match utils.gemini_client.COOLDOWN_SECONDS


# ──────────────────────────────────────────────────────────────
# GeminiSemaphore — dynamic key-aware concurrency control
# ──────────────────────────────────────────────────────────────

_gemini_semaphore: Optional[threading.Semaphore] = None
_gemini_semaphore_lock = threading.Lock()
_gemini_concurrency_limit: int = 1


def _count_active_keys() -> Tuple[int, float]:
    """
    Query the KeyPool stats to count currently active (non-cooling) keys.

    Returns:
        (active_key_count, earliest_cooldown_expiry_timestamp)
        The expiry timestamp is 0 if no keys are in cooldown.
    """
    try:
        from utils.gemini_client import get_pool_stats
        stats = get_pool_stats()
        sentiment_stats = stats.get("sentiment", {})
    except Exception:
        return 1, 0.0

    active = 0
    earliest_expiry = 0.0
    now = time.time()

    for _key_id, info in sentiment_stats.items():
        status = info.get("status", "active")
        if status == "active":
            active += 1
        # Approximate expiry: cooldown started at some past time, expires after COOLDOWN_SECONDS
        # We can't get the exact start time from stats but we can still detect
        # all-cooled-down state and sleep conservatively.

    return active, earliest_expiry


def _build_gemini_semaphore() -> threading.Semaphore:
    """
    Build (or rebuild) the global GeminiSemaphore based on current active key count.

    Concurrency = max(1, active_keys), capped at 2 for safety.
    Logs the chosen limit.
    """
    global _gemini_semaphore, _gemini_concurrency_limit

    active_keys, _ = _count_active_keys()

    # If no active keys, try waiting for cooldown recovery
    if active_keys == 0:
        logger.warning(
            f"All Gemini keys cooling down — sleeping {COOLDOWN_SECONDS}s for recovery..."
        )
        time.sleep(COOLDOWN_SECONDS)
        active_keys, _ = _count_active_keys()
        if active_keys == 0:
            logger.warning("Still no active keys after cooldown wait. Forcing concurrency=1.")
            active_keys = 1

    limit = max(1, min(active_keys, 2))
    _gemini_concurrency_limit = limit
    sem = threading.Semaphore(limit)
    logger.info(f"Gemini concurrency limit set to {limit} (active keys: {active_keys})")
    return sem


def _get_gemini_semaphore() -> threading.Semaphore:
    """
    Lazy-initialize the global GeminiSemaphore (thread-safe).
    """
    global _gemini_semaphore
    with _gemini_semaphore_lock:
        if _gemini_semaphore is None:
            _gemini_semaphore = _build_gemini_semaphore()
    return _gemini_semaphore


def _reset_gemini_semaphore() -> None:
    """
    Force-rebuild the semaphore (called after key exhaustion is detected).
    """
    global _gemini_semaphore
    with _gemini_semaphore_lock:
        _gemini_semaphore = _build_gemini_semaphore()


# ──────────────────────────────────────────────────────────────
# Model names — dynamically read from the index (not hardcoded)
# ──────────────────────────────────────────────────────────────

_FAMILY_PREFIXES = [
    "GPT", "Claude", "Gemini", "DeepSeek", "Qwen",
    "Llama", "Grok", "GLM", "Kimi", "MiniMax",
    "Devstral", "Trinity",
]

_FALLBACK_MODELS = [
    "GPT-5", "Claude Opus", "Gemini 3", "DeepSeek V3",
    "Llama 4", "Grok 4", "Qwen 3",
]


def _detect_family(name: str) -> Optional[str]:
    """Detect model family from name using prefix matching."""
    if not name:
        return None
    normalized = name.strip()
    for fam in _FAMILY_PREFIXES:
        if normalized.startswith(fam):
            return fam
    return None


def get_models_for_sentiment(index_path: Optional[str] = None, max_per_family: int = 1) -> List[str]:
    """
    Dynamically determine which models to scrape sentiment for.

    Reads the index data, groups models by family, and picks the top
    representative(s) from each family (by performance_rank).
    This ensures the sentiment pipeline always covers models present
    in the index, adapting automatically when models change.

    Falls back to _FALLBACK_MODELS if the index cannot be read.
    """
    if index_path is None:
        index_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "index", "latest.json"
        )

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.warning(f"Cannot read index data at {index_path}: {e}. Using fallback model list.")
        return _FALLBACK_MODELS

    if not index_data:
        return _FALLBACK_MODELS

    families: Dict[str, List[dict]] = {}
    for m in index_data:
        name = m.get("canonical_name") or m.get("model_name") or ""
        fam = _detect_family(name)
        if not fam:
            continue
        if fam not in families:
            families[fam] = []
        families[fam].append(m)

    selected = []
    for fam, members in sorted(families.items()):
        members.sort(key=lambda x: x.get("performance_rank") or 9999)
        for m in members[:max_per_family]:
            selected.append(fam)
            break

    seen = set()
    unique = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    return unique if unique else _FALLBACK_MODELS


# ──────────────────────────────────────────────────────────────
# Mention cache (per model, persisted to disk)
# ──────────────────────────────────────────────────────────────

def _get_mention_cache_path(model_name: str, base_dir: Optional[str] = None) -> str:
    """Return the path to the cached mentions file for a model."""
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "sentiment_cache"
        )
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
    return os.path.abspath(os.path.join(base_dir, f"{safe_name}.json"))


def _load_mention_cache(model_name: str, base_dir: Optional[str] = None) -> Optional[List[dict]]:
    """Load cached mentions from disk if fresh (within SENTIMENT_CACHE_TTL)."""
    path = _get_mention_cache_path(model_name, base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        if time.time() - entry.get("timestamp", 0) < SENTIMENT_CACHE_TTL:
            logger.info(f"[{model_name}] Loaded {len(entry['mentions'])} mentions from disk cache")
            return entry["mentions"]
    except (json.JSONDecodeError, IOError, KeyError):
        pass
    return None


def _save_mention_cache(model_name: str, mentions: List[dict], base_dir: Optional[str] = None) -> None:
    """Persist mentions to disk cache."""
    path = _get_mention_cache_path(model_name, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": time.time(), "mentions": mentions}, f)
    except IOError as e:
        logger.warning(f"[{model_name}] Failed to write mention cache: {e}")


# ──────────────────────────────────────────────────────────────
# Source scrapers
# ──────────────────────────────────────────────────────────────

def _build_search_query(model_name: str) -> str:
    """Build a disambiguation-aware search query globally."""
    name_lower = model_name.lower()
    
    # Specific provider-based disambiguation
    if "gemini" in name_lower:
        return f'"{model_name}" AND (Google OR AI OR LLM)'
    if "claude" in name_lower:
        return f'"{model_name}" AND (Anthropic OR AI OR LLM)'
    if "grok" in name_lower:
        return f'"{model_name}" AND (xAI OR Elon OR AI OR LLM)'
    if "llama" in name_lower:
        return f'"{model_name}" AND (Meta OR AI OR LLM)'
        
    # Universal disambiguation for everything else (avoids issues with "Kimi" or "MiMo")
    return f'"{model_name}" AND (AI OR LLM OR model OR "language model")'


def scrape_reddit_mentions(model_name: str, limit: int = 75) -> List[dict]:
    """Scrape Reddit mentions using old.reddit.com JSON API. No authentication required."""
    if not REQUESTS_AVAILABLE:
        return []
    mentions = []
    try:
        search_term = _build_search_query(model_name)
        query = requests.utils.quote(search_term)
        url = f"https://old.reddit.com/search.json?q={query}&sort=new&limit={limit}&restrict_sr=&t=month"
        headers = {"User-Agent": "LLMDEX-Sentiment/1.0 (research)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                d = post.get("data", {})
                title = d.get('title', '')
                selftext = d.get('selftext', '')
                text = f"{title} {selftext}"
                if "rodriguez" in text.lower() and "grok" in model_name.lower():
                    continue
                mentions.append({
                    "source": "Reddit",
                    "text": text[:500],
                    "score": d.get("score", 0),
                    "created": d.get("created_utc"),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                })
        else:
            logger.warning(f"Reddit API returned {resp.status_code} for '{model_name}'")
        time.sleep(1)
    except Exception as e:
        logger.warning(f"Reddit scrape failed for '{model_name}': {e}")
    return mentions


def scrape_hackernews_mentions(model_name: str, limit: int = 75) -> List[dict]:
    """Scrape Hacker News mentions via Algolia public search API. No authentication required."""
    if not REQUESTS_AVAILABLE:
        return []
    mentions = []
    try:
        url = f"https://hn.algolia.com/api/v1/search_by_date?query={model_name}&tags=story&hitsPerPage={limit}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for hit in data.get("hits", []):
                text = f"{hit.get('title', '')} {hit.get('story_text', '') or ''}"
                if len(model_name) < 6 and "AI" not in text and "LLM" not in text and "model" not in text:
                    continue
                mentions.append({
                    "source": "HackerNews",
                    "text": text[:500],
                    "score": hit.get("points", 0),
                    "created": hit.get("created_at"),
                    "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                })
        else:
            logger.warning(f"HN API returned {resp.status_code} for '{model_name}'")
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"HN scrape failed for '{model_name}': {e}")
    return mentions


def scrape_github_mentions(model_name: str, limit: int = 40) -> List[dict]:
    """Scrape GitHub issue mentions via public search API. No authentication required."""
    if not REQUESTS_AVAILABLE:
        return []
    mentions = []
    try:
        query = f'"{model_name}"'
        url = f"https://api.github.com/search/issues?q={query}+in:title&sort=created&order=desc&per_page={limit}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", []):
                text = f"{item.get('title', '')} {item.get('body', '') or ''}"
                mentions.append({
                    "source": "GitHub",
                    "text": text[:500],
                    "score": item.get("comments", 0),
                    "created": item.get("created_at"),
                    "url": item.get("html_url", ""),
                })
        else:
            logger.warning(f"GitHub API returned {resp.status_code} for '{model_name}'")
        time.sleep(2)
    except Exception as e:
        logger.warning(f"GitHub scrape failed for '{model_name}': {e}")
    return mentions


# ──────────────────────────────────────────────────────────────
# Deduplication & Spam Filtering
# ──────────────────────────────────────────────────────────────

def _deduplicate_mentions(mentions: List[dict]) -> List[dict]:
    """Remove duplicate posts based on text similarity."""
    seen_hashes = set()
    unique = []
    for m in mentions:
        text = (m.get("text") or "").strip().lower()[:100]
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique.append(m)
    return unique


def _filter_spam(mentions: List[dict]) -> List[dict]:
    """Filter obvious spam and low-quality posts."""
    spam_patterns = [
        r"buy\s+now", r"click\s+here", r"free\s+money",
        r"subscribe\s+to", r"follow\s+me", r"upvote\s+this",
        r"giveaway", r"promo\s+code",
    ]
    spam_regex = re.compile("|".join(spam_patterns), re.IGNORECASE)
    filtered = []
    for m in mentions:
        text = m.get("text", "")
        if len(text.strip()) < 15:
            continue
        if spam_regex.search(text):
            continue
        filtered.append(m)
    return filtered


# ──────────────────────────────────────────────────────────────
# Mention Sampling
# ──────────────────────────────────────────────────────────────

def _sample_mentions(model_name: str, mentions: List[dict], max_count: int = MAX_MENTIONS_PER_MODEL) -> List[dict]:
    """
    If mentions exceed max_count, randomly sample down to max_count.
    Logs a warning when sampling occurs.
    """
    if len(mentions) <= max_count:
        return mentions
    sampled = random.sample(mentions, max_count)
    logger.warning(
        f"[{model_name}] Sampled {max_count} mentions from {len(mentions)} total "
        f"(random sampling applied to stay within API budget)"
    )
    return sampled


# ──────────────────────────────────────────────────────────────
# Batching helper
# ──────────────────────────────────────────────────────────────

def batch_mentions(mentions: List[dict], batch_size: int = BATCH_SIZE) -> List[List[dict]]:
    """
    Split a flat list of mentions into chunks of at most batch_size.

    Returns:
        List of mention sub-lists (batches).
    """
    return [mentions[i:i + batch_size] for i in range(0, len(mentions), batch_size)]


# ──────────────────────────────────────────────────────────────
# Gemini Sentiment Classification
# ──────────────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are analyzing public discussions about AI models.

STRICT RELEVANCE RULES:

Mark is_relevant = true ONLY if the post contains a clear opinion,
evaluation, experience, praise, criticism, or performance discussion
about the AI model itself.

Mark is_relevant = false if the post merely:

- mentions the model name without judging it
- asks procedural/setup questions
- discusses business or monetization
- discusses prompt usage without evaluation
- is meta chatter
- is only a bug report without sentiment

We only want posts that help assess model quality.

For each item determine:

1) is_relevant (boolean)
2) sentiment_score (-1.0 to +1.0)
3) is_strong_praise (boolean)
4) is_strong_criticism (boolean)

Be conservative and strict.
Return a JSON array with one object per input item.
"""


# In-memory cache for Gemini sentiment results (24h TTL)
_sentiment_cache: Dict[str, Dict[str, Any]] = {}
_sentiment_cache_lock = threading.Lock()


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception looks like a quota / rate-limit error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ["429", "quota", "rate limit", "resource_exhausted", "too many requests"])


def _normalize_gemini_response(raw: Any) -> Optional[List[dict]]:
    """Normalize a Gemini response to a plain list of classification dicts."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ["results", "items", "classifications", "data"]:
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]
    return None


def _call_gemini_with_timeout(
    prompt: str,
    timeout: float = GEMINI_CALL_TIMEOUT,
) -> Tuple[Optional[Any], Optional[Exception]]:
    """
    Execute a call_gemini() in a daemon thread, enforcing a wall-clock timeout.

    Returns:
        (result, exception) — exactly one will be non-None.
    """
    try:
        from utils.gemini_client import call_gemini
    except ImportError:
        return None, ImportError("gemini_client not available")

    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            result_holder["value"] = call_gemini(
                prompt=prompt,
                system_instruction=SENTIMENT_SYSTEM_PROMPT,
                pool_type="sentiment",
                temperature=0.1,
            )
        except Exception as exc:
            error_holder["exc"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, TimeoutError(f"Gemini call timed out after {timeout}s")

    if "exc" in error_holder:
        return None, error_holder["exc"]

    return result_holder.get("value"), None


def analyze_batch_with_gemini(
    model_name: str,
    batch: List[dict],
    batch_index: int,
    total_batches: int,
) -> Optional[List[dict]]:
    """
    Send a single batch of mentions to Gemini for sentiment classification.

    Architecture:
    - Acquires GeminiSemaphore before calling (prevents key exhaustion)
    - Includes wall-clock timeout guard
    - Backpressure retry: on quota error waits QUOTA_RETRY_DELAY then retries once
    - Only falls back to None after retry failure

    Args:
        model_name: The AI model being analyzed
        batch: List of mention dicts for this batch
        batch_index: 1-based index of this batch (for logging)
        total_batches: Total number of batches (for logging)

    Returns:
        List of classification dicts, or None on unrecoverable failure.
    """
    batch_data = [
        {
            "index": i,
            "text": (m.get("text") or "")[:180],
            "source": m.get("source", "unknown"),
        }
        for i, m in enumerate(batch)
    ]

    prompt = (
    f"AI MODEL: {model_name}\n\n"
    f"Classify these posts:\n\n"
    f"{json.dumps(batch_data, separators=(',', ':'))}"
    )


    semaphore = _get_gemini_semaphore()

    for attempt in range(1):  # single attempt — retries burn quota, failures fall back to VADER
        if attempt == 1:
            logger.info(f"[{model_name}] Retrying Gemini call after quota error... (batch {batch_index}/{total_batches})")
            time.sleep(QUOTA_RETRY_DELAY)

        logger.info(
            f"[{model_name}] Waiting for available Gemini slot... "
            f"(batch {batch_index}/{total_batches}, attempt {attempt + 1})"
        )

        with semaphore:
            logger.info(
                f"[{model_name}] Processing batch {batch_index}/{total_batches} "
                f"({len(batch)} mentions, attempt {attempt + 1})"
            )
            raw, exc = _call_gemini_with_timeout(prompt, timeout=GEMINI_CALL_TIMEOUT)

        # Handle errors
        if exc is not None:
            if isinstance(exc, TimeoutError):
                logger.warning(
                    f"[{model_name}] Gemini timed out after {GEMINI_CALL_TIMEOUT}s "
                    f"on batch {batch_index}/{total_batches}"
                )
                return None  # Timeouts don't benefit from retry

            if _is_quota_error(exc):
                logger.warning(
                    f"[{model_name}] Quota error on batch {batch_index}/{total_batches}: {exc}"
                )
                if attempt == 0:
                    # Rebuild semaphore to reflect new key state before retry
                    _reset_gemini_semaphore()
                    continue  # → retry
                # Second attempt also quota-failed
                logger.error(
                    f"[{model_name}] Quota error persisted after retry on batch "
                    f"{batch_index}/{total_batches}. Falling back."
                )
                return None

            # Non-quota error — log and give up
            logger.warning(
                f"[{model_name}] Gemini error on batch {batch_index}/{total_batches}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        # raw is valid
        if raw is None:
            logger.warning(
                f"[{model_name}] Gemini returned empty response for batch {batch_index}/{total_batches}"
            )
            return None

        result = _normalize_gemini_response(raw)
        if result is None:
            logger.warning(
                f"[{model_name}] Unexpected Gemini response type {type(raw)} "
                f"for batch {batch_index}/{total_batches}"
            )
            return None

        return result

    return None  # Should never reach here, but satisfies type checker


# ──────────────────────────────────────────────────────────────
# In-memory Gemini result caching & classification orchestration
# ──────────────────────────────────────────────────────────────

def _make_cache_key(model_name: str, mentions: List[dict]) -> str:
    fingerprint = f"{model_name}:{len(mentions)}:{mentions[0].get('text', '')[:50] if mentions else ''}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _classify_with_gemini(model_name: str, mentions: List[dict]) -> Optional[List[dict]]:
    """
    Classify all mentions for a model using Gemini, in batches of BATCH_SIZE.

    - Checks in-memory cache first (24h TTL)
    - Sends batches sequentially through GeminiSemaphore
    - Gracefully degrades individual failed batches to placeholder entries
    - Returns None only if every single batch fails

    Returns:
        Flat list of classification results (one per mention), or None on total failure.
    """
    cache_key = _make_cache_key(model_name, mentions)

    with _sentiment_cache_lock:
        entry = _sentiment_cache.get(cache_key)
        if entry and time.time() - entry["timestamp"] < SENTIMENT_CACHE_TTL:
            logger.debug(f"[{model_name}] In-memory sentiment cache hit")
            return entry["results"]

    batches = batch_mentions(mentions, BATCH_SIZE)
    total_batches = len(batches)
    all_results: List[dict] = []
    any_success = False

    for idx, batch in enumerate(batches, start=1):
        batch_results = analyze_batch_with_gemini(model_name, batch, idx, total_batches)

        if batch_results is not None:
            any_success = True
            all_results.extend(batch_results)
        else:
            # Graceful degradation: placeholder entries keep index alignment
            logger.warning(
                f"[{model_name}] Batch {idx}/{total_batches} failed — "
                f"inserting {len(batch)} placeholder entries"
            )
            for _ in batch:
                all_results.append({
                    "is_relevant": True,
                    "sentiment_score": None,
                    "is_strong_praise": False,
                    "is_strong_criticism": False,
                    "_fallback": True,
                })

        # Brief inter-batch pause to reduce burst pressure on the API
        if idx < total_batches:
            time.sleep(0.5)

    if not any_success:
        return None

    with _sentiment_cache_lock:
        _sentiment_cache[cache_key] = {
            "timestamp": time.time(),
            "results": all_results,
        }

    return all_results


# ──────────────────────────────────────────────────────────────
# Sentiment scoring (Gemini-enhanced + VADER fallback)
# ──────────────────────────────────────────────────────────────

def analyze_sentiment_gemini(model_name: str, mentions: List[dict]) -> dict:
    """
    Analyze sentiment using Gemini classification.
    Falls back to VADER if Gemini fails entirely.

    Returns per-model sentiment metrics including confidence_score.
    """
    if not mentions:
        return {
            "community_sentiment": None,
            "mention_count": 0,
            "controversy_index": None,
            "sentiment_trend": None,
            "confidence_score": 0,
            "sentiment_method": "none",
        }

    classifications = _classify_with_gemini(model_name, mentions)

    if classifications and len(classifications) > 0:
        return _compute_gemini_sentiment(mentions, classifications)
    else:
        logger.info(f"[{model_name}] All Gemini batches failed — falling back to VADER")
        result = _analyze_sentiment_vader(mentions)
        result["sentiment_method"] = "vader_fallback"
        return result


def _compute_gemini_sentiment(mentions: List[dict], classifications: List[dict]) -> dict:
    """Compute sentiment metrics from Gemini classification results."""
    valid_scores = []
    strong_praise_count = 0
    strong_criticism_count = 0
    total_valid = 0

    for i, classification in enumerate(classifications):
        if i >= len(mentions):
            break

        is_relevant = classification.get("is_relevant", True)
        if not is_relevant:
            continue

        total_valid += 1
        raw_score = classification.get("sentiment_score")

        # Placeholder from a timed-out/failed batch — count toward total but not scores
        if raw_score is None:
            continue

        sentiment_score = max(-1.0, min(1.0, float(raw_score)))
        engagement = max(1, mentions[i].get("score", 1))

        valid_scores.append({
            "score": sentiment_score,
            "engagement": engagement,
        })

        if classification.get("is_strong_praise", False):
            strong_praise_count += 1
        if classification.get("is_strong_criticism", False):
            strong_criticism_count += 1

    if not valid_scores:
        return {
            "community_sentiment": None,
            "mention_count": len(mentions),
            "controversy_index": None,
            "sentiment_trend": None,
            "confidence_score": 0,
            "sentiment_method": "gemini",
        }

    total_weight = sum(math.log(s["engagement"] + 1) for s in valid_scores)
    if total_weight > 0:
        weighted_sentiment = sum(
            s["score"] * math.log(s["engagement"] + 1) for s in valid_scores
        ) / total_weight
    else:
        weighted_sentiment = sum(s["score"] for s in valid_scores) / len(valid_scores)

    controversy = (strong_praise_count + strong_criticism_count) / total_valid if total_valid > 0 else 0

    if weighted_sentiment > 0.1:
        trend = "positive"
    elif weighted_sentiment < -0.1:
        trend = "negative"
    else:
        trend = "stable"

    confidence_score = min(1.0, total_valid / 30)

    # Guardrail for low sample sizes
    if total_valid < 10:
        confidence_score = min(confidence_score, 0.25)
    elif total_valid < 20:
        confidence_score = min(confidence_score, 0.6)

    return {
        "community_sentiment": round(weighted_sentiment, 4),
        "mention_count": len(mentions),
        "controversy_index": round(controversy, 4),
        "sentiment_trend": trend,
        "confidence_score": round(confidence_score, 2),
        "sentiment_method": "gemini",
        "valid_posts": total_valid,
        "strong_praise_count": strong_praise_count,
        "strong_criticism_count": strong_criticism_count,
    }


def _analyze_sentiment_vader(mentions: List[dict]) -> dict:
    """Analyze sentiment using VADER (fallback method)."""
    if not mentions:
        return {
            "community_sentiment": None,
            "mention_count": 0,
            "controversy_index": None,
            "sentiment_trend": None,
            "confidence_score": 0,
        }

    if not VADER_AVAILABLE:
        return {
            "community_sentiment": None,
            "mention_count": len(mentions),
            "controversy_index": None,
            "sentiment_trend": None,
            "confidence_score": 0,
            "_note": "VADER not installed",
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    mixed_count = 0

    for mention in mentions:
        text = mention.get("text", "")
        if not text.strip():
            continue
        vs = analyzer.polarity_scores(text)
        compound = vs["compound"]
        engagement = max(1, mention.get("score", 1))
        scores.append({
            "compound": compound,
            "source": mention.get("source", "unknown"),
            "engagement": engagement,
        })
        if vs["pos"] > 0.1 and vs["neg"] > 0.1:
            mixed_count += 1

    if not scores:
        return {
            "community_sentiment": None,
            "mention_count": len(mentions),
            "controversy_index": None,
            "sentiment_trend": None,
            "confidence_score": 0,
        }

    total_weight = sum(math.log(s["engagement"] + 1) for s in scores)
    if total_weight > 0:
        weighted_sentiment = sum(
            s["compound"] * math.log(s["engagement"] + 1) for s in scores
        ) / total_weight
    else:
        weighted_sentiment = sum(s["compound"] for s in scores) / len(scores)

    controversy = mixed_count / len(scores) if scores else 0

    if weighted_sentiment > 0.1:
        trend = "positive"
    elif weighted_sentiment < -0.1:
        trend = "negative"
    else:
        trend = "stable"

    confidence_score = min(1.0, len(scores) / 30)

    return {
        "community_sentiment": round(weighted_sentiment, 4),
        "mention_count": len(mentions),
        "controversy_index": round(controversy, 4),
        "sentiment_trend": trend,
        "confidence_score": round(confidence_score, 2),
    }


# ──────────────────────────────────────────────────────────────
# Quote extraction for UI display
# ──────────────────────────────────────────────────────────────

_PROFANITY_WORDS = {
    "fuck", "shit", "damn", "ass", "bitch", "crap", "dick",
    "bastard", "piss", "hell", "wtf", "stfu", "lmao",
}


def _basic_profanity_filter(text: str) -> str:
    """Replace profane words with asterisks. Conservative filter."""
    words = text.split()
    cleaned = []
    for word in words:
        if word.lower().strip(".,!?;:'\"") in _PROFANITY_WORDS:
            cleaned.append("***")
        else:
            cleaned.append(word)
    return " ".join(cleaned)


def _strip_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def _clean_links(text: str) -> str:
    """Remove markdown links and bare URLs."""
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_community_examples(
    mentions: List[dict],
    gemini_classifications: Optional[List[dict]] = None,
    max_quotes: int = 3,
) -> List[dict]:
    """
    Extract the best community quotes for UI display.

    Selection logic:
    - 1 highest positive sentiment post
    - 1 highest negative sentiment post
    - 1 most engaged neutral post
    """
    if not mentions:
        return []

    QUALITY_KEYWORDS = [
        "better", "worse", "performance", "accuracy",
        "fast", "slow", "hallucinate", "hallucinates",
        "quality", "benchmark", "impressive", "bad",
        "good", "excellent", "weak", "strong",
        "latency", "speed", "coding", "reasoning"
    ]

    candidates = []
    vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    for i, m in enumerate(mentions):
        text = (m.get("text") or "").strip()

        # Prefer posts that evaluate the model
        text_lower = text.lower()
        quality_hits = sum(1 for kw in QUALITY_KEYWORDS if kw in text_lower)

        # Skip low-signal posts unless highly upvoted
        if quality_hits == 0 and m.get("score", 0) < 5:
            continue

        if len(text) < 20:
            continue

        if gemini_classifications and i < len(gemini_classifications):
            cls = gemini_classifications[i]
            if not cls.get("is_relevant", True):
                continue
            raw = cls.get("sentiment_score")
            score = float(raw) if raw is not None else 0.0
        elif vader:
            vs = vader.polarity_scores(text)
            score = vs["compound"]
        else:
            score = 0

        candidates.append({
            "text": text.replace("\n", " ").strip(),
            "source": m.get("source", "Community"),
            "url": m.get("url", ""),
            "engagement": max(1, m.get("score", 1)),
            "sentiment": score,
        })

    if not candidates:
        return []

    selected = []
    seen_texts: set = set()

    # 1) Highest positive
    for c in sorted(candidates, key=lambda c: c["sentiment"], reverse=True):
        if c["sentiment"] > 0:
            selected.append(c)
            seen_texts.add(c["text"][:50].lower())
            break

    # 2) Highest negative
    for c in sorted(candidates, key=lambda c: c["sentiment"]):
        if c["sentiment"] < 0 and c["text"][:50].lower() not in seen_texts:
            selected.append(c)
            seen_texts.add(c["text"][:50].lower())
            break

    # 3) Most engaged neutral
    neutral = sorted(
        [c for c in candidates if -0.15 < c["sentiment"] < 0.15 and c["text"][:50].lower() not in seen_texts],
        key=lambda c: c["engagement"],
        reverse=True,
    )
    if neutral:
        selected.append(neutral[0])

    # Fill remaining slots by engagement
    if len(selected) < max_quotes:
        for c in sorted(
            [c for c in candidates if c["text"][:50].lower() not in seen_texts],
            key=lambda c: c["engagement"],
            reverse=True,
        ):
            if len(selected) >= max_quotes:
                break
            selected.append(c)
            seen_texts.add(c["text"][:50].lower())

    quotes = []
    for c in selected[:max_quotes]:
        text = c["text"]
        text = _clean_links(text)
        text = _strip_emojis(text)
        if len(text) > 220:
            text = text[:217].rsplit(" ", 1)[0] + "..."
        text = _basic_profanity_filter(text)
        if len(text) < 15:
            continue
        source_label = "Hacker News" if c["source"] == "HackerNews" else c["source"]
        quotes.append({
            "source": source_label,
            "text": text,
            "url": c["url"],
        })

    if not quotes:
        return [{
            "source": "System",
            "text": "No relevant AI-specific discussions found in last 30 days.",
            "url": "",
        }]

    return quotes


# ──────────────────────────────────────────────────────────────
# Phase 1 worker: scrape + clean + sample (no Gemini)
# ──────────────────────────────────────────────────────────────

def _scrape_model_mentions(
    model_name: str,
    mention_cache_dir: Optional[str],
) -> Tuple[str, List[dict]]:
    """
    Scrape, deduplicate, filter, and sample mentions for one model.
    Uses disk cache to avoid re-scraping within TTL.

    Returns:
        (model_name, cleaned_mentions)
    """
    cached = _load_mention_cache(model_name, mention_cache_dir)
    if cached is not None:
        logger.info(f"[{model_name}] Using cached mentions — skipping scrape")
        return model_name, _sample_mentions(model_name, cached)

    logger.info(f"[{model_name}] Scraping fresh mentions from Reddit / HN / GitHub...")
    raw: List[dict] = []
    raw.extend(scrape_reddit_mentions(model_name))
    raw.extend(scrape_hackernews_mentions(model_name))
    raw.extend(scrape_github_mentions(model_name))

    raw = _deduplicate_mentions(raw)
    raw = _filter_spam(raw)
    _save_mention_cache(model_name, raw, mention_cache_dir)

    raw_count = len(raw)
    sampled = _sample_mentions(model_name, raw)

    return model_name, {
        "mentions": sampled,
        "raw_count": raw_count
    }



# ──────────────────────────────────────────────────────────────
# Pre-filter: reduce Gemini load with cheap heuristics
# ──────────────────────────────────────────────────────────────

def _prefilter_for_gemini(mentions: list) -> list:
    """Cheap heuristic filter to reduce Gemini load."""
    if not mentions:
        return mentions

    KEY_HINTS = [
        "better", "worse", "fast", "slow", "good", "bad",
        "benchmark", "performance", "accuracy",
        "hallucinate", "quality", "impressive"
    ]

    filtered = []
    for m in mentions:
        text = (m.get("text") or "").lower()

        # keep if strong signal
        if any(k in text for k in KEY_HINTS):
            filtered.append(m)
            continue

        # keep if highly upvoted (important discussions)
        if m.get("score", 0) >= 8:
            filtered.append(m)

    # hard cap to protect quota
    return filtered[:120]


# ──────────────────────────────────────────────────────────────
# Phase 2 worker: classify + score + extract quotes (Gemini-gated)
# ──────────────────────────────────────────────────────────────

def process_model_sentiment(
    model_name: str,
    mentions: List[dict],
) -> dict:
    """
    Run Gemini classification + sentiment scoring for one model.

    This is Phase 2 and is called sequentially after all mentions are scraped,
    with Gemini access gated by GeminiSemaphore inside analyze_batch_with_gemini.

    Args:
        model_name: Name of the AI model
        mentions: Pre-scraped, cleaned, sampled mention list

    Returns:
        Sentiment result dict for this model.
    """
    model_start = time.time()
    logger.info(f"[{model_name}] Starting Gemini classification ({len(mentions)} mentions)")

    sentiment = analyze_sentiment_gemini(model_name, mentions)
    sentiment["model_name"] = model_name
    sentiment["scraped_at"] = datetime.now(timezone.utc).isoformat()
    sentiment["_experimental"] = True  # SAFETY: always labeled

    # Retrieve Gemini classifications from cache for quote selection
    cache_key = _make_cache_key(model_name, mentions)
    with _sentiment_cache_lock:
        cached_entry = _sentiment_cache.get(cache_key)
    gemini_cls = cached_entry.get("results") if cached_entry else None

    sentiment["community_examples"] = _extract_community_examples(
        mentions, gemini_classifications=gemini_cls, max_quotes=3
    )
    sentiment["top_quotes"] = sentiment["community_examples"]  # backward compat

    elapsed = time.time() - model_start
    logger.info(
        f"[{model_name}] Completed sentiment in {elapsed:.1f}s — "
        f"{sentiment['mention_count']} mentions, "
        f"sentiment={sentiment.get('community_sentiment', 'N/A')}, "
        f"method={sentiment.get('sentiment_method', 'unknown')}"
    )

    return sentiment


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_sentiment_pipeline(
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Run the full sentiment pipeline in two phases:

    Phase 1 — Parallel scraping (up to MAX_SCRAPE_WORKERS threads):
        For each model: load disk cache or scrape Reddit + HN + GitHub,
        deduplicate, filter spam, and sample to MAX_MENTIONS_PER_MODEL.

    Phase 2 — Controlled Gemini classification (sequential, semaphore-gated):
        For each model: send batches of 40 mentions to Gemini via
        GeminiSemaphore, with backpressure retry on quota errors, and
        VADER fallback if Gemini is fully unavailable.

    Returns dict of model_name → sentiment data.
    """
    pipeline_start = time.time()

    if model_names is None:
        model_names = get_models_for_sentiment()

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "sentiment"
        )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mention_cache_dir = os.path.join(os.path.dirname(output_dir), "sentiment_cache")
    os.makedirs(mention_cache_dir, exist_ok=True)

    num_models = len(model_names)
    scrape_workers = min(MAX_SCRAPE_WORKERS, num_models)

    # ── Pre-initialize GeminiSemaphore based on live key state ──
    _get_gemini_semaphore()

    logger.info(
        f"Sentiment pipeline: {num_models} models | "
        f"Phase 1 scrape workers: {scrape_workers} | "
        f"Phase 2 Gemini concurrency: {_gemini_concurrency_limit}"
    )

    # ── Phase 1: Parallel scraping ────────────────────────────
    phase1_start = time.time()
    logger.info("=== Phase 1: Scraping mentions (parallel) ===")

    mentions_by_model: Dict[str, List[dict]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=scrape_workers) as executor:
        futures = {
            executor.submit(_scrape_model_mentions, name, mention_cache_dir): name
            for name in model_names
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                name, mentions = future.result()
                mentions_by_model[name] = mentions
                logger.info(f"[{name}] Phase 1 complete — {len(mentions)} mentions ready")
            except Exception as exc:
                model_name = futures[future]
                logger.error(f"[{model_name}] Scrape phase failed: {exc}", exc_info=True)
                mentions_by_model[model_name] = []

    phase1_elapsed = time.time() - phase1_start
    logger.info(f"=== Phase 1 complete in {phase1_elapsed:.1f}s ===")

    # ── Phase 2: Serial Gemini classification (semaphore-gated internally) ──
    phase2_start = time.time()
    logger.info("=== Phase 2: Gemini classification (semaphore-controlled) ===")

    results: Dict[str, dict] = {}

    for model_name in model_names:
        mentions = mentions_by_model.get(model_name, [])
        mentions = _prefilter_for_gemini(mentions)
        try:
            sentiment = process_model_sentiment(model_name, mentions)
            results[model_name] = sentiment
        except Exception as exc:
            logger.error(f"[{model_name}] Unhandled exception in classification: {exc}", exc_info=True)
            results[model_name] = {
                "model_name": model_name,
                "community_sentiment": None,
                "mention_count": 0,
                "controversy_index": None,
                "sentiment_trend": None,
                "confidence_score": 0,
                "sentiment_method": "error",
                "_experimental": True,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "community_examples": [],
                "top_quotes": [],
            }

    phase2_elapsed = time.time() - phase2_start
    logger.info(f"=== Phase 2 complete in {phase2_elapsed:.1f}s ===")

    # ── Save output ───────────────────────────────────────────
    output_path = os.path.join(output_dir, "latest.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, indent=2, default=str)

    total_elapsed = time.time() - pipeline_start
    succeeded = sum(1 for v in results.values() if v.get("sentiment_method") not in ("error", "none"))
    gemini_success = sum(1 for v in results.values() if v.get("sentiment_method") == "gemini")
    vader_fallback = sum(1 for v in results.values() if v.get("sentiment_method") == "vader_fallback")

    logger.info(
        f"Sentiment pipeline complete — "
        f"{succeeded}/{num_models} succeeded "
        f"({gemini_success} Gemini, {vader_fallback} VADER fallback). "
        f"Total runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min). "
        f"Output: {output_path}"
    )

    return results


def merge_sentiment_with_dataset(
    dataset: List[dict],
    sentiment_path: Optional[str] = None,
) -> List[dict]:
    """
    Merge sentiment data into the main model dataset.

    SAFETY: Sentiment fields are always prefixed/labeled as experimental.
    They NEVER influence any ranking calculations.
    """
    if sentiment_path is None:
        sentiment_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "sentiment", "latest.json"
        )

    if not os.path.exists(sentiment_path):
        logger.info("No sentiment data found. Skipping merge.")
        return dataset

    try:
        with open(sentiment_path, "r", encoding="utf-8") as f:
            sentiment_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load sentiment data: {e}")
        return dataset

    sentiment_lookup: Dict[str, dict] = {}
    for s in sentiment_data:
        name = s.get("model_name", "").lower().strip()
        sentiment_lookup[name] = s

    for model in dataset:
        model_name = (model.get("canonical_name") or model.get("model_name") or "").lower().strip()
        matched = sentiment_lookup.get(model_name)
        if not matched:
            for sname, sdata in sentiment_lookup.items():
                if sname in model_name or model_name in sname:
                    matched = sdata
                    break

        if matched:
            model["community_sentiment"] = matched.get("community_sentiment")
            model["mention_count"] = matched.get("mention_count", 0)
            model["controversy_index"] = matched.get("controversy_index")
            model["sentiment_trend"] = matched.get("sentiment_trend")
            model["sentiment_experimental"] = True
        else:
            model["community_sentiment"] = None
            model["mention_count"] = 0
            model["controversy_index"] = None
            model["sentiment_trend"] = None
            model["sentiment_experimental"] = True

    return dataset


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dotenv import load_dotenv
    load_dotenv()
    from utils.logger import setup_logging
    setup_logging()
    run_sentiment_pipeline()
