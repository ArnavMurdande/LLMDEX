"""
gemini_client.py — Gemini API key rotation and management.

Provides:
  - Separate key pools for Advisor and Sentiment workloads
  - Round-robin key selection
  - Automatic retry on 429 / quota errors with key switching
  - 5-minute cooldown on failed keys
  - Safe logging (keys are never printed)
  - Graceful failure when all keys are exhausted

SAFETY:
  - Keys are read from environment variables only
  - Actual key values are NEVER logged or printed
  - Failed keys are automatically cooled down
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

COOLDOWN_SECONDS = 300  # 5 minutes
MODEL_NAME = "gemini-3-flash-preview"  # High-speed model with better free tier quotas

# Key pools — read from environment variables
# Advisor keys:  GEMINI_ADVISOR_KEY_1, GEMINI_ADVISOR_KEY_2, GEMINI_ADVISOR_KEY_3
# Sentiment keys: GEMINI_SENTIMENT_KEY_1, GEMINI_SENTIMENT_KEY_2


def _load_keys(prefix: str, max_keys: int = 5) -> List[str]:
    """Load API keys from environment variables with given prefix."""
    keys = []
    for i in range(1, max_keys + 1):
        key = os.environ.get(f"{prefix}_{i}", "").strip()
        if key:
            keys.append(key)
    # Also check a single non-numbered key as fallback
    single = os.environ.get(prefix, "").strip()
    if single and single not in keys:
        keys.insert(0, single)
    return keys


# ──────────────────────────────────────────────────────────────
# Key Pool Manager
# ──────────────────────────────────────────────────────────────

class KeyPool:
    """
    Manages a pool of API keys with round-robin rotation and cooldown.
    
    Usage tracking:
      - Each key tracks total uses and last failure time
      - Failed keys are cooled down for COOLDOWN_SECONDS
      - Round-robin selection skips cooled-down keys
    """

    def __init__(self, pool_name: str, keys: List[str]):
        self.pool_name = pool_name
        self._keys = keys
        self._current_index = 0
        self._usage_count: Dict[int, int] = {i: 0 for i in range(len(keys))}
        self._failure_time: Dict[int, float] = {}  # key_index → timestamp
        logger.info(
            f"KeyPool '{pool_name}': initialized with {len(keys)} key(s)"
        )

    @property
    def size(self) -> int:
        return len(self._keys)

    def _is_cooled_down(self, index: int) -> bool:
        """Check if a key is currently in cooldown."""
        fail_time = self._failure_time.get(index)
        if fail_time is None:
            return False
        elapsed = time.time() - fail_time
        return elapsed < COOLDOWN_SECONDS

    def get_next_key(self) -> Optional[str]:
        """
        Get the next available key using round-robin.
        Skips keys that are in cooldown.
        Returns None if all keys are exhausted.
        """
        if not self._keys:
            return None

        n = len(self._keys)
        for _ in range(n):
            idx = self._current_index % n
            self._current_index = (self._current_index + 1) % n

            if not self._is_cooled_down(idx):
                self._usage_count[idx] = self._usage_count.get(idx, 0) + 1
                logger.debug(
                    f"KeyPool '{self.pool_name}': using key #{idx + 1} "
                    f"(usage count: {self._usage_count[idx]})"
                )
                return self._keys[idx]

        logger.warning(
            f"KeyPool '{self.pool_name}': all {n} keys are in cooldown!"
        )
        return None

    def mark_failed(self, key: str) -> None:
        """Mark a key as failed and start its cooldown."""
        try:
            idx = self._keys.index(key)
            self._failure_time[idx] = time.time()
            logger.warning(
                f"KeyPool '{self.pool_name}': key #{idx + 1} marked as failed, "
                f"cooldown for {COOLDOWN_SECONDS}s"
            )
        except ValueError:
            pass  # Key not in pool, ignore

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics (safe — no actual keys exposed)."""
        stats = {}
        for i in range(len(self._keys)):
            status = "cooled_down" if self._is_cooled_down(i) else "active"
            stats[f"key_{i + 1}"] = {
                "status": status,
                "usage_count": self._usage_count.get(i, 0),
            }
        return stats


# ──────────────────────────────────────────────────────────────
# Global Key Pools (lazy init)
# ──────────────────────────────────────────────────────────────

_advisor_pool: Optional[KeyPool] = None
_sentiment_pool: Optional[KeyPool] = None


def _get_advisor_pool() -> KeyPool:
    global _advisor_pool
    if _advisor_pool is None:
        keys = _load_keys("GEMINI_ADVISOR_KEY")
        # Also accept generic GEMINI_API_KEY as fallback
        if not keys:
            generic = os.environ.get("GEMINI_API_KEY", "").strip()
            if generic:
                keys = [generic]
        _advisor_pool = KeyPool("advisor", keys)
    return _advisor_pool


def _get_sentiment_pool() -> KeyPool:
    global _sentiment_pool
    if _sentiment_pool is None:
        keys = _load_keys("GEMINI_SENTIMENT_KEY")
        # Fallback to generic key
        if not keys:
            generic = os.environ.get("GEMINI_API_KEY", "").strip()
            if generic:
                keys = [generic]
        _sentiment_pool = KeyPool("sentiment", keys)
    return _sentiment_pool


# ──────────────────────────────────────────────────────────────
# Gemini API Call with Rotation
# ──────────────────────────────────────────────────────────────

def _is_quota_error(error: Exception) -> bool:
    """Check if an error is a 429 / quota error."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "429", "quota", "rate limit", "resource_exhausted",
        "too many requests", "rate_limit",
    ])



def call_gemini(
    prompt: str,
    system_instruction: str,
    pool_type: str = "advisor",
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini API with automatic key rotation using google-genai SDK.

    Args:
        prompt: User prompt / data to send
        system_instruction: System instruction for the model
        pool_type: "advisor" or "sentiment"
        temperature: Generation temperature
        max_retries: Maximum number of retries across different keys

    Returns:
        Parsed JSON response dict, or None on failure
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error(
            "google-genai not installed. "
            "Run: pip install google-genai"
        )
        return None

    pool = _get_advisor_pool() if pool_type == "advisor" else _get_sentiment_pool()

    if pool.size == 0:
        logger.warning(
            f"No Gemini API keys configured for '{pool_type}'. "
            f"Set GEMINI_ADVISOR_KEY_1 or GEMINI_API_KEY environment variable."
        )
        return None

    last_error = None

    for attempt in range(max_retries):
        key = pool.get_next_key()
        if key is None:
            logger.error(
                f"Gemini call failed: all keys exhausted for '{pool_type}' "
                f"after {attempt} attempts"
            )
            return None

        try:
            client = genai.Client(api_key=key)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                )
            )

            # Parse JSON response
            if response and response.text:
                try:
                    result = json.loads(response.text)
                    logger.debug(
                        f"Gemini '{pool_type}' call succeeded on attempt {attempt + 1}"
                    )
                    return result
                except json.JSONDecodeError as je:
                    logger.warning(
                        f"Gemini returned invalid JSON: {je}. "
                        f"Raw text length: {len(response.text)}"
                    )
                    # Try to extract JSON from markdown code blocks
                    text = response.text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            pass
                    last_error = je
            else:
                logger.warning(f"Gemini returned empty response on attempt {attempt + 1}")
                last_error = ValueError("Empty response")

        except Exception as e:
            last_error = e
            if _is_quota_error(e):
                logger.warning(
                    f"Gemini quota error on attempt {attempt + 1}: "
                    f"rotating key for '{pool_type}'"
                )
                pool.mark_failed(key)
            else:
                logger.error(
                    f"Gemini API error on attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                # For non-quota errors, still try next key
                pool.mark_failed(key)

    logger.error(
        f"Gemini call failed after {max_retries} attempts for '{pool_type}': "
        f"{type(last_error).__name__ if last_error else 'Unknown'}"
    )
    return None


def call_gemini_batch(
    prompts: List[str],
    system_instruction: str,
    pool_type: str = "sentiment",
    temperature: float = 0.1,
    delay_between: float = 0.5,
) -> List[Optional[Dict[str, Any]]]:
    """
    Call Gemini for multiple prompts sequentially with rate limiting.

    Args:
        prompts: List of prompts
        system_instruction: Shared system instruction
        pool_type: Key pool to use
        temperature: Generation temperature
        delay_between: Seconds between calls

    Returns:
        List of results (None for failed calls)
    """
    results = []
    for i, prompt in enumerate(prompts):
        result = call_gemini(
            prompt=prompt,
            system_instruction=system_instruction,
            pool_type=pool_type,
            temperature=temperature,
        )
        results.append(result)

        # Rate limiting
        if i < len(prompts) - 1:
            time.sleep(delay_between)

    return results


def get_pool_stats() -> Dict[str, Any]:
    """Get stats for all pools (for monitoring, no key exposure)."""
    return {
        "advisor": _get_advisor_pool().get_stats(),
        "sentiment": _get_sentiment_pool().get_stats(),
    }
