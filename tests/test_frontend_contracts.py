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
        self._in_capability_selector = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "select" and values.get("id") == "capability-selector":
            self._in_capability_selector = True
        elif tag == "option" and self._in_capability_selector:
            self.capabilities.append(values.get("value"))

    def handle_endtag(self, tag):
        if tag == "select" and self._in_capability_selector:
            self._in_capability_selector = False


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "website" / "index.html").read_text(encoding="utf-8")
        cls.js = (ROOT / "website" / "app.js").read_text(encoding="utf-8")
        cls.css = (ROOT / "website" / "styles.css").read_text(encoding="utf-8")
        cls.parser = _IdCollector()
        cls.parser.feed(cls.html)

    def test_general_is_default_and_all_capability_options_exist(self):
        self.assertIn('id="capability-selector"', self.html)
        self.assertIn('<option value="general">General</option>', self.html)
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
        ):
            self.assertIn(element_id, self.parser.ids)
        self.assertIn("showModelDetailsDrawer", self.js)
        self.assertIn("renderDataQuality", self.js)
        self.assertNotIn('id="benchmark-modal"', self.html)
        self.assertNotIn('id="status-filter"', self.html)
        self.assertIn('id="table-scrollbar-spacer"', self.html)
        self.assertIn('id="table-scrollbar-bottom-spacer"', self.html)

    def test_capability_dropdown_and_mirrored_scrollbars_are_bound(self):
        self.assertIn('enhanceCustomSelect(selector, "capability")', self.js)
        self.assertIn('selector.addEventListener("change"', self.js)
        self.assertIn("table-scrollbar-bottom", self.js)
        self.assertIn('id="leaderboard-source-viewer"', self.html)
        self.assertIn("syncLeaderboardSourceViewer", self.js)
        self.assertIn("Artificial Analysis based", self.js)
        self.assertIn("LLMStats based", self.js)

    def test_shared_visual_effect_layer_is_loaded(self):
        effects = (ROOT / "website" / "effects.js").read_text(encoding="utf-8")
        self.assertIn('src="effects.js?v=3"', self.html)
        self.assertIn("installSpecularButtons", effects)
        self.assertIn("installPixelBlast", effects)
        self.assertIn('variant: "square"', effects)
        self.assertIn("pixelSize: 3", effects)
        self.assertIn("color: [56, 189, 248]", effects)
        self.assertIn("patternScale: 4", effects)
        self.assertIn("patternDensity: 0.8", effects)
        self.assertIn("particle.x +=", effects)

    def test_advisor_discloses_when_gemini_is_not_connected(self):
        self.assertIn("Gemini not connected", self.js)
        self.assertIn("Deterministic dataset analysis", self.js)

    def test_mobile_table_reveals_scrolled_metrics(self):
        self.assertIn("position: static !important", self.css)
        self.assertIn("touch-action: pan-x pan-y", self.css)

    def test_fable_identity_and_pending_score_copy_are_normalized(self):
        self.assertIn("normalizePublishedModels", self.js)
        self.assertIn('canonical_name: "Claude Fable 5"', self.js)
        self.assertNotIn(">Pending match<", self.js)

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

    def test_capability_ui_keeps_every_published_benchmark_column(self):
        self.assertNotIn(".slice(0, 4)", self.js)
        self.assertIn("sourcePopulation", self.js)
        self.assertIn("benchmark-coverage", self.js)

    def test_single_source_status_and_stale_state_are_rendered(self):
        self.assertIn("llmstats_only", self.js)
        self.assertIn("aa_only", self.js)
        self.assertIn("identity_review", self.js)
        self.assertIn("Data quality notice", self.js)

    def test_static_advisor_uses_configured_api_or_local_mode(self):
        self.assertIn("resolveAdvisorApiBase", self.js)
        self.assertIn("window.LLMDEX_API_BASE", self.js)
        self.assertIn('meta[name="llmdex-api-base"]', self.js)
        self.assertIn("if (!advisorHealthUrl || !advisorResponseUrl)", self.js)
        self.assertNotIn('fetch("/api/health"', self.js)
        self.assertNotIn('fetch("/api/advisor"', self.js)


if __name__ == "__main__":
    unittest.main()
