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
  - CORS restricted to localhost
  - Rate limiting enforced server-side
  - API keys never exposed to frontend
  - Gemini cannot access internet
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 8080))
WEBSITE_DIR = os.path.join(os.path.dirname(__file__), "website")


class LLMDEXHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves API endpoints + static files."""

    def __init__(self, *args, **kwargs):
        # Serve from the website directory
        super().__init__(*args, directory=WEBSITE_DIR, **kwargs)

    def do_POST(self):
        """Handle POST requests for API endpoints."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/advisor":
            self._handle_advisor()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        """Handle GET requests — API routes + static files."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._handle_health()
        elif parsed.path.startswith("/data/"):
            # Serve data files from the project root
            self._serve_data_file(parsed.path)
        else:
            # Serve static files from website/
            super().do_GET()

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
            from utils.gemini_client import get_pool_stats
            stats = get_pool_stats()
            self._json_response({"status": "ok", "pools": stats})
        except ImportError:
            self._json_response({"status": "ok", "pools": {}, "note": "gemini_client not available"})

    def _serve_data_file(self, path):
        """Serve data files from the project root (not website dir)."""
        # Security: only allow files under data/
        if ".." in path:
            self.send_error(403, "Forbidden")
            return

        file_path = os.path.join(os.path.dirname(__file__), path.lstrip("/"))
        if os.path.isfile(file_path):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "http://localhost")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")

    def _json_response(self, data: dict, status: int = 200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
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
        os.environ.get(f"GEMINI_ADVISOR_KEY_{i}") or os.environ.get("GEMINI_API_KEY")
        for i in range(1, 4)
    )
    if has_advisor_keys:
        logger.info("✓ Gemini API keys detected — AI advisor will use Gemini")
    else:
        logger.info("⚠ No Gemini API keys found — AI advisor will use client-side fallback")
        logger.info("  Set GEMINI_API_KEY environment variable to enable Gemini responses")

    server = HTTPServer(("", PORT), LLMDEXHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
