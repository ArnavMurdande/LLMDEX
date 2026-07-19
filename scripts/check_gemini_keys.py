"""Validate configured Gemini API keys without printing secret values."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"


def _load_local_dotenv() -> None:
    """Load .env for local diagnostics; GitHub Actions supplies real env vars."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except ImportError:
        pass

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _configured_keys(pool: str) -> list[tuple[str, str]]:
    prefix = f"GEMINI_{pool.upper()}_KEY"
    keys: list[tuple[str, str]] = []
    for index in range(1, 6):
        name = f"{prefix}_{index}"
        value = os.environ.get(name, "").strip()
        if value:
            keys.append((name, value))

    single = os.environ.get(prefix, "").strip()
    if single:
        keys.insert(0, (prefix, single))

    generic = os.environ.get("GEMINI_API_KEY", "").strip()
    if not keys and generic:
        keys.append(("GEMINI_API_KEY", generic))
    return keys


def _check_key(key: str, timeout: float) -> tuple[str, str]:
    request = urllib.request.Request(
        MODELS_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMDEX-key-check/1.0",
            "x-goog-api-key": key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and isinstance(payload.get("models"), list):
                return "working", "authenticated"
            return "invalid", f"unexpected HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return "quota_exhausted", "authenticated but quota-limited"
        if exc.code in (401, 403):
            return "invalid", f"authentication rejected (HTTP {exc.code})"
        return "error", f"Gemini API returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return "error", f"network check failed: {type(reason).__name__}"
    except (ValueError, json.JSONDecodeError):
        return "error", "Gemini API returned an unreadable response"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=("advisor", "sentiment"), required=True)
    parser.add_argument("--require-any", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    _load_local_dotenv()
    configured = _configured_keys(args.pool)
    if not configured:
        print(f"{args.pool}: no API keys configured")
        return 1 if args.require_any else 0

    usable = 0
    for name, key in configured:
        status, detail = _check_key(key, args.timeout)
        if status in {"working", "quota_exhausted"}:
            usable += 1
        print(f"{name}: {status} ({detail})")

    print(
        f"{args.pool}: {usable}/{len(configured)} keys authenticated "
        "(secret values were not printed)"
    )
    return 0 if usable or not args.require_any else 1


if __name__ == "__main__":
    sys.exit(main())
