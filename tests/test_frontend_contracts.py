from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.capabilities = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if "capability-pill" in values.get("class", "").split():
            self.capabilities.append(values.get("data-capability"))


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "website" / "index.html").read_text(encoding="utf-8")
        cls.js = (ROOT / "website" / "app.js").read_text(encoding="utf-8")
        cls.css = (ROOT / "website" / "styles.css").read_text(encoding="utf-8")
        cls.parser = _IdCollector()
        cls.parser.feed(cls.html)

    def test_general_is_default_and_all_capability_pills_exist(self):
        self.assertIn(
            '<button class="capability-pill active"',
            self.html,
        )
        self.assertEqual(
            self.parser.capabilities,
            [
                "general",
                "coding",
                "math",
                "reasoning",
                "writing",
                "research",
                "long_context",
                "tool_calling",
            ],
        )

    def test_details_quality_and_data_views_exist(self):
        for element_id in (
            "model-drawer",
            "quality-grid",
            "stale-data-banner",
            "data-section",
            "consensus-trend-chart",
        ):
            self.assertIn(element_id, self.parser.ids)
        self.assertIn("showModelDetailsDrawer", self.js)
        self.assertIn("renderDataQuality", self.js)

    def test_mobile_pills_scroll_and_keyboard_navigation_is_bound(self):
        self.assertIn("overflow-x: auto", self.css)
        self.assertIn('event.key === "ArrowLeft"', self.js)
        self.assertIn('event.key === "ArrowRight"', self.js)

    def test_capability_publications_use_llmstats_names(self):
        for capability in (
            "coding",
            "math",
            "reasoning",
            "writing",
            "research",
            "long_context",
            "tool_calling",
        ):
            path = ROOT / "data" / "capabilities" / f"{capability}.json"
            contract = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(contract["source"], "LLMStats")
            self.assertGreater(len(contract["rows"]), 0)
            for row in contract["rows"]:
                self.assertTrue(row["source_name"])
                self.assertNotIn("(max)", row["source_name"].casefold())
                self.assertNotIn("(xhigh)", row["source_name"].casefold())

    def test_single_source_status_and_stale_state_are_rendered(self):
        self.assertIn("llmstats_only", self.js)
        self.assertIn("aa_only", self.js)
        self.assertIn("identity_review", self.js)
        self.assertIn("Data quality notice", self.js)


if __name__ == "__main__":
    unittest.main()
