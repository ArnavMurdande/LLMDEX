import os
import unittest
from unittest.mock import patch

from api_server import LLMDEXHandler, _api_cors_origins


class ApiCorsTests(unittest.TestCase):
    def test_production_and_same_origin_local_hosts_are_allowed(self):
        with patch.dict(os.environ, {}, clear=True):
            origins = _api_cors_origins()

        self.assertIn("https://llmdex.onrender.com", origins)
        self.assertIn("https://llmdex.pages.dev", origins)
        self.assertIn("http://localhost:8080", origins)
        self.assertNotIn("*", origins)

    def test_configured_origins_are_normalized_and_wildcard_is_ignored(self):
        with patch.dict(
            os.environ,
            {
                "LLMDEX_CORS_ALLOWED_ORIGINS": (
                    "https://preview.example.com/, *, "
                    "http://localhost:5173"
                )
            },
            clear=True,
        ):
            origins = _api_cors_origins()

        self.assertIn("https://preview.example.com", origins)
        self.assertIn("http://localhost:5173", origins)
        self.assertNotIn("*", origins)

    def test_handler_rejects_unknown_browser_origin(self):
        handler = object.__new__(LLMDEXHandler)
        handler.headers = {"Origin": "https://attacker.example"}

        self.assertFalse(handler._is_api_origin_allowed())

    def test_handler_accepts_production_and_originless_requests(self):
        handler = object.__new__(LLMDEXHandler)
        handler.headers = {"Origin": "https://llmdex.pages.dev"}
        self.assertTrue(handler._is_api_origin_allowed())
        handler.headers = {}
        self.assertTrue(handler._is_api_origin_allowed())



if __name__ == "__main__":
    unittest.main()
