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
        self.assertIn("Source: Artificial Analysis", self.js)
        self.assertIn("Source: LLMStats", self.js)

    def test_transparency_terminology_and_attribution(self):
        # A. Attribution exists
        self.assertIn("https://artificialanalysis.ai/", self.html)
        self.assertIn("https://llm-stats.com/", self.html)
        self.assertIn("not affiliated", self.html.casefold())
        self.assertIn("endorsed", self.html.casefold())

        # B. New frontend elements exist
        self.assertIn("metric-info-btn", self.html)
        self.assertIn("data-metric-help", self.html)
        self.assertIn("METRIC_HELP", self.js)
        self.assertIn("setupMetricHelp", self.js)
        self.assertIn("source-credits-section", self.html)
        self.assertIn("metric-source", self.html)

        # C. Misleading terminology removed from website/index.html
        self.assertNotIn("financial-grade analytics index", self.html)
        self.assertNotIn("only verified benchmark data", self.html)
        self.assertNotIn("Free models receive maximum efficiency", self.html)
        self.assertNotIn("Models Tracked</h3>", self.html)
        self.assertNotIn("Open SOTA</h3>", self.html)

        # D. New terminology exists
        self.assertIn("Source-transparent model comparison", self.html)
        self.assertIn("Latest successfully processed snapshot", self.html)
        self.assertIn("Model Variants Tracked", self.html)
        self.assertIn("Open-Weights SOTA", self.html)
        self.assertIn("published, source-linked benchmark observations", self.html)

        # E. Existing IDs preserved
        for required_id in [
            "top-model",
            "val-model",
            "eff-model",
            "total-models",
            "top-llmdex-model",
            "open-sota-model",
            "matched-families",
            "last-updated",
        ]:
            self.assertIn(required_id, self.parser.ids)

    def test_shared_visual_effect_layer_is_loaded(self):
        effects = (ROOT / "website" / "effects.js").read_text(encoding="utf-8")
        pixelblast = (ROOT / "frontend-effects" / "PixelBlast.jsx").read_text(
            encoding="utf-8"
        )
        pixelblast_entry = (
            ROOT / "frontend-effects" / "index.jsx"
        ).read_text(encoding="utf-8")
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        self.assertIn('id="pixel-blast-root"', self.html)
        self.assertIn('src="pixelblast.bundle.js?v=1"', self.html)
        self.assertIn('href="pixelblast.bundle.css?v=1"', self.html)
        self.assertIn('src="effects.js?v=4"', self.html)
        self.assertIn("installSpecularButtons", effects)
        self.assertNotIn("installPixelBlast", effects)
        self.assertIn("THREE.WebGLRenderer", pixelblast)
        self.assertIn("THREE.ShaderMaterial", pixelblast)
        self.assertIn("EffectComposer", pixelblast)
        self.assertIn("#define FBM_OCTAVES 5", pixelblast)
        self.assertIn('variant="square"', pixelblast_entry)
        self.assertIn('color={light ? "#000000" : "#ffffff"}', pixelblast_entry)
        self.assertIn("patternDensity={1.4}", pixelblast_entry)
        self.assertTrue((ROOT / "website" / "pixelblast.bundle.js").is_file())
        self.assertTrue((ROOT / "website" / "pixelblast.bundle.css").is_file())
        for dependency in ("react", "react-dom", "three", "postprocessing"):
            self.assertIn(dependency, package["dependencies"])

    def test_advisor_discloses_when_gemini_is_not_connected(self):
        self.assertIn("Gemini not connected", self.js)
        self.assertIn("Deterministic dataset analysis", self.js)
        self.assertIn('name="llmdex-api-base"', self.html)
        self.assertIn('content="https://llmdex.onrender.com"', self.html)
        self.assertIn("ensureAdvisorHealth", self.js)
        self.assertIn("controller.abort(), 60000", self.js)
        self.assertIn("Waking Gemini Advisor", self.js)

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

    def test_active_source_update_lifecycle(self):
        self.assertIn("updateActiveSourceCount", self.js)
        self.assertIn("updateActiveSourceCount(allDataRef || [])", self.js)
        self.assertIn("active source", self.js)

    def test_shared_tooltip_manager(self):
        self.assertIn("explanationTooltipState", self.js)
        self.assertIn("showExplanationTooltip", self.js)
        self.assertIn("hideExplanationTooltip", self.js)
        self.assertIn("activeTrigger:", self.js)
        self.assertIn("activeMode:", self.js)
        self.assertNotIn("activeMetricHelpBtn", self.js)
        self.assertNotIn("let activeIcon = null;", self.js)

    def test_tooltip_accessibility_attributes(self):
        self.assertIn('role="dialog"', self.html)
        self.assertIn('aria-labelledby="tooltip-title"', self.html)
        self.assertIn('aria-modal="false"', self.html)
        self.assertIn('type="button"', self.html)
        self.assertIn('aria-label="Close explanation"', self.html)

    def test_power_bi_wording_consistency(self):
        self.assertIn("Power BI wide schema", self.html)
        self.assertIn("Download processed datasets for the dashboard, Power BI, and external analysis.", self.html)
        self.assertNotIn("Power BI-ready General index", self.html)
        self.assertNotIn("Download the same processed datasets used by this dashboard and Power BI", self.html)

    def test_documentation_wording_consistency(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        methodology = (ROOT / "METHODOLOGY.md").read_text(encoding="utf-8")
        data_sources = (ROOT / "docs" / "DATA_SOURCES.md").read_text(encoding="utf-8")

        self.assertIn("Open-Weights SOTA", readme)
        self.assertIn("Open-Weights SOTA", methodology)
        self.assertNotIn("OPEN SOTA", readme)
        self.assertNotIn("OPEN SOTA", methodology)

        self.assertIn("publicly rendered server-side pages", data_sources)
        self.assertNotIn("adheres to public server contracts", data_sources)

    def test_general_header_information_buttons(self):
        self.assertIn("column-info-btn", self.js)
        self.assertIn('type="button"', self.js)
        self.assertIn('aria-expanded="false"', self.js)
        self.assertIn('aria-label="Explain', self.js)
        self.assertIn('closestElement(e.target, ".info-icon")', self.js)

    def test_capability_header_information_buttons(self):
        self.assertIn("data-capability-help", self.js)
        self.assertIn("__capabilityHelpMap", self.js)
        self.assertIn("cap_cat_", self.js)
        self.assertIn("cap_bm_", self.js)


if __name__ == "__main__":
    unittest.main()

