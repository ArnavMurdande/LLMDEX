"""
api_server.py — Lightweight API server for the LLMDEX AI Advisor.

Serves:
  POST /api/advisor  → Gemini-powered advisor responses
  GET  /api/health   → Key pool health check

Also serves the static website files.

Usage:
  python api_server.py
  → Starts on http://localhost:8080

SAFETY:
  - API CORS restricted to approved production and local origins
  - Rate limiting enforced server-side
  - API keys never exposed to frontend
  - Gemini cannot access internet
"""
from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Static dashboard serving does not require python-dotenv.
    pass

import gzip
import csv
import json
import logging
import mimetypes
import os
import sys
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 8080))
WEBSITE_DIR = os.path.join(os.path.dirname(__file__), "website")
DATA_DIR = Path(__file__).resolve().parent / "data"
_DATA_CACHE = {}
_DATA_CACHE_LOCK = threading.Lock()
_DEFAULT_API_CORS_ORIGINS = frozenset(
    {
        "https://llmdex.onrender.com",
        "https://llmdex.pages.dev",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    }
)


def _api_cors_origins():
    """Return the exact browser origins allowed to call the API."""
    configured = os.environ.get("LLMDEX_CORS_ALLOWED_ORIGINS", "")
    extra_origins = {
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip() and origin.strip() != "*"
    }
    return _DEFAULT_API_CORS_ORIGINS | extra_origins


class LLMDEXHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves API endpoints + static files."""

    def __init__(self, *args, **kwargs):
        # Serve from the website directory
        self._cache_control_sent = False
        super().__init__(*args, directory=WEBSITE_DIR, **kwargs)

    def end_headers(self):
        if not self._cache_control_sent:
            if self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-store")
            else:
                self.send_header(
                    "Cache-Control",
                    "public, max-age=3600, stale-while-revalidate=86400",
                )
        super().end_headers()

    def do_POST(self):
        """Handle POST requests for API endpoints."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/advisor":
            if not self._is_api_origin_allowed():
                logger.warning(
                    "Rejected Advisor request from unapproved origin: %s",
                    self.headers.get("Origin"),
                )
                self._json_response({"error": "Origin not allowed"}, 403)
                return
            self._handle_advisor()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        """Handle GET requests — API routes + static files."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._handle_health()
        elif parsed.path == "/api/leaderboards/general":
            self._handle_general_leaderboard(parse_qs(parsed.query))
        elif parsed.path.startswith("/api/leaderboards/capabilities/"):
            capability = parsed.path.rsplit("/", 1)[-1]
            self._handle_capability_leaderboard(capability)
        elif parsed.path == "/api/data-quality":
            self._serve_json_contract(DATA_DIR / "quality" / "latest.json")
        elif parsed.path == "/api/methodology":
            self._handle_methodology()
        elif parsed.path.startswith("/api/models/"):
            self._handle_model_route(parsed.path)
        elif parsed.path.startswith("/data/"):
            # Serve data files from the project root
            self._serve_data_file(parsed.path)
        else:
            # Serve static files from website/
            super().do_GET()

    @staticmethod
    def _read_json(path: Path, default=None):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    def _publication_metadata(self):
        manifest = self._read_json(
            DATA_DIR / "methodology" / "publication_manifest.json", {}
        )
        return {
            "generated_at": manifest.get("generated_at"),
            "methodology_version": manifest.get("methodology_version"),
            "source_updated_at": manifest.get("source_updated_at", {}),
            "source_health": manifest.get("source_health", {}),
        }

    def _serve_json_contract(self, path: Path):
        payload = self._read_json(path)
        if payload is None:
            self._json_response({"error": "Contract unavailable"}, 404)
            return
        if isinstance(payload, dict):
            payload = {**self._publication_metadata(), **payload}
        self._json_response(payload)

    def _handle_general_leaderboard(self, query):
        rows = self._read_json(DATA_DIR / "index" / "latest.json", [])
        if not isinstance(rows, list):
            self._json_response({"error": "General leaderboard unavailable"}, 503)
            return
        sort_mode = (query.get("sort") or ["aa"])[0].casefold()
        if sort_mode == "llmdex":
            rows = sorted(
                rows,
                key=lambda row: (
                    row.get("llmdex_score") is None,
                    -(row.get("llmdex_score") or 0),
                    row.get("performance_rank") or 10**9,
                    str(row.get("canonical_name") or "").casefold(),
                ),
            )
        else:
            rows = sorted(
                rows,
                key=lambda row: (
                    row.get("performance_rank") or 10**9,
                    str(row.get("canonical_name") or "").casefold(),
                ),
            )
        self._json_response(
            {
                **self._publication_metadata(),
                "source": "Artificial Analysis",
                "sort": sort_mode,
                "unranked_consensus_count": sum(
                    row.get("llmdex_score") is None for row in rows
                ),
                "rows": rows,
                "attribution": (
                    "General intelligence, pricing and API performance data "
                    "from Artificial Analysis."
                ),
            }
        )

    def _handle_capability_leaderboard(self, capability):
        allowed = {
            "coding",
            "math",
            "reasoning",
            "writing",
            "research",
            "long_context",
            "tool_calling",
        }
        if capability not in allowed:
            self._json_response({"error": "Unknown capability"}, 404)
            return
        self._serve_json_contract(
            DATA_DIR / "capabilities" / f"{capability}.json"
        )

    def _handle_model_route(self, path):
        suffix = unquote(path.removeprefix("/api/models/")).strip("/")
        wants_history = suffix.endswith("/history")
        family_id = (
            suffix[: -len("/history")].strip("/") if wants_history else suffix
        )
        if not family_id:
            self._json_response({"error": "Family ID required"}, 400)
            return
        if wants_history:
            history_path = DATA_DIR / "history" / "family_snapshots.csv"
            rows = []
            if history_path.exists():
                with history_path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    rows = [
                        row
                        for row in csv.DictReader(handle)
                        if row.get("family_id") == family_id
                    ]
            self._json_response(
                {
                    **self._publication_metadata(),
                    "family_id": family_id,
                    "rows": rows,
                }
            )
            return

        families_contract = self._read_json(
            DATA_DIR / "families" / "latest.json", {}
        )
        family = next(
            (
                row
                for row in families_contract.get("rows", [])
                if row.get("family_id") == family_id
            ),
            None,
        )
        if not family:
            self._json_response({"error": "Model family not found"}, 404)
            return
        aa_rows = self._read_json(DATA_DIR / "index" / "latest.json", [])
        llmstats_contract = self._read_json(
            DATA_DIR / "cleaned" / "llmstats" / "latest.json", {}
        )
        llmstats_rows = (
            llmstats_contract.get("rows", [])
            if isinstance(llmstats_contract, dict)
            else llmstats_contract
        )
        self._json_response(
            {
                **self._publication_metadata(),
                "family": family,
                "artificial_analysis_variants": [
                    row for row in aa_rows if row.get("family_id") == family_id
                ],
                "llmstats_observations": [
                    row
                    for row in llmstats_rows
                    if row.get("family_id") == family_id
                    or row.get("matched_aa_family_id") == family_id
                ],
                "match_explanation": (
                    "LLMStats publishes a family-level result using its own "
                    "naming. LLMDEX links it to the corresponding Artificial "
                    "Analysis family; this does not prove LLMStats tested the "
                    "highlighted AA reasoning or deployment profile."
                ),
            }
        )

    def _handle_methodology(self):
        self._json_response(
            {
                **self._publication_metadata(),
                "sources": self._read_json(
                    DATA_DIR / "methodology" / "source_config.json", {}
                ),
                "scores": self._read_json(
                    DATA_DIR / "methodology" / "score_versions.json", {}
                ),
                "benchmarks": self._read_json(
                    DATA_DIR / "methodology" / "benchmark_registry.json", {}
                ),
            }
        )

    def _handle_advisor(self):
        """Handle POST /api/advisor — Gemini-powered advisor."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            query = data.get("query", "").strip()
            if not query:
                self._json_response({"error": "No query provided"}, 400)
                return

            if len(query) > 500:
                self._json_response({"error": "Query too long (max 500 chars)"}, 400)
                return

            # Call the Gemini advisor
            try:
                from pipeline.gemini_advisor import generate_advisor_response
                response = generate_advisor_response(
                    user_query=query,
                    user_id=self.client_address[0],  # IP-based rate limiting
                )
                self._json_response(response)
            except ImportError:
                self._json_response({
                    "answer": "AI advisor module not available. Using local data analysis instead.",
                    "referenced_models": [],
                    "data_points_used": [],
                    "source": "fallback",
                }, 200)
            except Exception as e:
                logger.error(f"Advisor error: {e}")
                self._json_response({
                    "answer": "AI advisor temporarily unavailable. Please use the ranking filters and priority selector below.",
                    "referenced_models": [],
                    "data_points_used": [],
                    "source": "fallback",
                }, 200)

        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error(f"API error: {e}")
            self._json_response({"error": "Internal server error"}, 500)

    def _handle_health(self):
        """Handle GET /api/health — pool stats."""
        try:
            from utils.gemini_client import get_client_health, get_pool_stats
            stats = get_pool_stats()
            client = get_client_health()
            self._json_response({
                "status": "ok" if client["ready"] else "degraded",
                "advisor": client,
                "pools": stats,
            })
        except ImportError:
            self._json_response({
                "status": "degraded",
                "advisor": {
                    "ready": False,
                    "sdk_available": False,
                    "advisor_key_count": 0,
                },
                "pools": {},
                "note": "gemini_client not available",
            })

    def _serve_data_file(self, path):
        """Serve data files from the project root (not website dir)."""
        relative_path = path.removeprefix("/data/").lstrip("/")
        file_path = (DATA_DIR / relative_path).resolve()
        if file_path != DATA_DIR and DATA_DIR not in file_path.parents:
            self.send_error(403, "Forbidden")
            return

        if not file_path.is_file():
            self.send_error(404, "File not found")
            return

        stat = file_path.stat()
        etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header(
                "Cache-Control",
                "public, max-age=300, stale-while-revalidate=86400",
            )
            self._cache_control_sent = True
            self.end_headers()
            return

        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        use_gzip = accepts_gzip and stat.st_size >= 1024
        cache_key = (str(file_path), stat.st_mtime_ns, stat.st_size, use_gzip)

        with _DATA_CACHE_LOCK:
            body = _DATA_CACHE.get(cache_key)
        if body is None:
            body = file_path.read_bytes()
            if use_gzip:
                body = gzip.compress(body, compresslevel=6)
            with _DATA_CACHE_LOCK:
                stale_keys = [
                    key for key in _DATA_CACHE if key[0] == str(file_path)
                ]
                for key in stale_keys:
                    _DATA_CACHE.pop(key, None)
                _DATA_CACHE[cache_key] = body

        content_type = mimetypes.guess_type(file_path.name)[0]
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type == "application/json":
            content_type = "application/json; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Cache-Control",
            "public, max-age=300, stale-while-revalidate=86400",
        )
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self._cache_control_sent = True
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, data: dict, status: int = 200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_api_cors_headers()
        self.send_header("Cache-Control", "no-store")
        self._cache_control_sent = True
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _request_origin(self):
        return (self.headers.get("Origin") or "").strip().rstrip("/")

    def _is_api_origin_allowed(self):
        origin = self._request_origin()
        return not origin or origin in _api_cors_origins()

    def _send_api_cors_headers(self):
        origin = self._request_origin()
        if not origin or origin not in _api_cors_origins():
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404, "Not Found")
            return
        if not self._is_api_origin_allowed():
            self._json_response({"error": "Origin not allowed"}, 403)
            return
        self.send_response(204)
        self._send_api_cors_headers()
        self.send_header("Cache-Control", "no-store")
        self._cache_control_sent = True
        self.end_headers()

    def log_message(self, format, *args):
        """Override to use Python logger instead of stderr."""
        if "/api/" in str(args[0]):
            logger.info(f"{self.client_address[0]} - {args[0]}")


def main():
    setup_logging()
    logger.info(f"Starting LLMDEX API server on http://localhost:{PORT}")
    logger.info(f"Serving website from: {WEBSITE_DIR}")

    # Check for API keys
    has_advisor_keys = any(
        (os.environ.get(f"GEMINI_ADVISOR_KEY_{i}") or "").strip()
        or (os.environ.get("GEMINI_API_KEY") or "").strip()
        for i in range(1, 6)
    )
    if has_advisor_keys:
        logger.info("✓ Gemini API keys detected — AI advisor will use Gemini")
    else:
        logger.info("⚠ No Gemini API keys found — AI advisor will use client-side fallback")
        logger.info("  Set GEMINI_API_KEY environment variable to enable Gemini responses")

    server = ThreadingHTTPServer(("", PORT), LLMDEXHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
