from __future__ import annotations

import unittest
from pathlib import Path

from scraper.scrape_llmstats import (
    _extract_top_models,
    _infer_provider,
    parse_llmstats_table,
    parse_rendered_capability_table,
)


FIXTURES = Path(__file__).parent / "fixtures"


class LLMStatsFixtureTests(unittest.TestCase):
    def test_missing_provider_uses_unambiguous_series_prefix(self):
        self.assertEqual(
            _infer_provider("DeepSeek-V4-Pro-Max", "deepseek-v4-pro-max"),
            "DeepSeek",
        )

    def test_table_schema_and_source_native_names(self):
        general = (FIXTURES / "llmstats_general.html").read_text(encoding="utf-8")
        coding = (FIXTURES / "llmstats_coding.html").read_text(encoding="utf-8")
        rows, diagnostics = parse_llmstats_table(
            general,
            "https://llm-stats.com/leaderboards/llm-leaderboard",
            category_top_models={"coding": _extract_top_models(coding)},
        )
        self.assertEqual(diagnostics["schema_changes"], [])
        self.assertEqual([row.source_name for row in rows], ["GPT-5.6 Sol", "Claude Fable 5"])
        self.assertEqual(rows[0].category_ranks["coding"], 1.0)
        self.assertEqual(
            rows[0].rank_evidence["coding"],
            "source_published_top_models_order",
        )
        self.assertEqual(
            rows[0].benchmark_observations["swe_bench_verified"]["value"],
            76.5,
        )

    def test_missing_required_columns_fails_closed(self):
        rows, diagnostics = parse_llmstats_table(
            "<table><tr><th>Model</th></tr><tr><td>Example</td></tr></table>",
            "https://llm-stats.com/leaderboards/llm-leaderboard",
        )
        self.assertEqual(rows, [])
        self.assertTrue(diagnostics["schema_changes"])

    def test_empty_page_returns_no_rows(self):
        rows, diagnostics = parse_llmstats_table(
            "<html><body></body></html>",
            "https://llm-stats.com/leaderboards/llm-leaderboard",
        )
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["headers"], [])

    def test_duplicate_source_ids_are_reported(self):
        html = """
        <table><tr><th>Model</th><th>LLM Stats</th><th>Organization</th></tr>
        <tr><td><a href='/models/same'>One</a></td><td>10</td><td>Lab</td></tr>
        <tr><td><a href='/models/same'>Two</a></td><td>9</td><td>Lab</td></tr>
        </table>
        """
        rows, diagnostics = parse_llmstats_table(
            html, "https://llm-stats.com/leaderboards/llm-leaderboard"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(diagnostics["duplicate_source_ids"], ["same"])

    def test_rendered_capability_table_preserves_all_benchmark_columns(self):
        table = {
            "headers": [
                {"text": "MODEL"},
                {"text": "Rating\nconservative"},
                {"text": "Price\n$/M"},
                {"text": "Context window"},
                {"text": "Speed\nchars/s"},
                {"text": "TTFT\nlatency"},
                {"text": "LICENSE"},
                {
                    "text": "SWE-Bench Verified\n104 models",
                    "source_url": "https://llm-stats.com/benchmarks/swe-bench",
                },
                {
                    "text": "LiveCodeBench\n73 models",
                    "source_url": "https://llm-stats.com/benchmarks/livecodebench",
                },
            ],
            "rows": [
                {
                    "model_name": "GPT-5.6 Sol",
                    "model_url": "https://llm-stats.com/models/gpt-5.6-sol",
                    "cells": [
                        {"text": "#1\nGPT-5.6 Sol"},
                        {"text": "49.5"},
                        {"text": "$2.00"},
                        {"text": "1M"},
                        {"text": "120"},
                        {"text": "0.2"},
                        {"text": "Closed"},
                        {"text": "76.5"},
                        {"text": "81.2"},
                    ],
                }
            ],
        }
        rows, diagnostics = parse_rendered_capability_table(
            table,
            "coding",
            "https://llm-stats.com/leaderboards/best-ai-for-coding",
        )
        self.assertEqual(diagnostics["schema_changes"], [])
        self.assertEqual(len(diagnostics["benchmark_columns"]), 2)
        self.assertEqual(rows[0]["source_details"]["License"], "Closed")
        self.assertEqual(rows[0]["category_rank"], 1.0)
        self.assertEqual(
            rows[0]["benchmark_observations"]["swe_bench_verified"][
                "capabilities"
            ],
            ["coding"],
        )
        self.assertEqual(
            rows[0]["benchmark_observations"]["llmstats__livecodebench"]["value"],
            81.2,
        )

    def test_rendered_general_table_is_accepted_when_server_html_has_no_table(self):
        rendered = {
            "headers": [
                {"text": ""},
                {"text": "Model"},
                {"text": "LLM Stats"},
                {"text": "Coding"},
                {"text": "Organization"},
            ],
            "rows": [
                {
                    "model_name": "Claude Opus 5",
                    "model_url": "https://llm-stats.com/models/claude-opus-5",
                    "cells": [
                        {"text": ""},
                        {"text": "Claude Opus 5"},
                        {"text": "56.3"},
                        {"text": "42.7"},
                        {"text": "Anthropic"},
                    ],
                },
                {
                    "model_name": "Gemini 3.7 Flash",
                    "model_url": "https://llm-stats.com/models/gemini-3-7-flash",
                    "cells": [
                        {"text": ""},
                        {"text": "Gemini 3.7 Flash"},
                        {"text": "44.2"},
                        {"text": "35.1"},
                        {"text": "Google"},
                    ],
                },
            ],
        }
        rows, diagnostics = parse_llmstats_table(
            "<html><body>client rendered</body></html>",
            "https://llm-stats.com/leaderboards/llm-leaderboard",
            rendered_general_table=rendered,
        )
        self.assertEqual(diagnostics["schema_changes"], [])
        self.assertEqual([row.source_name for row in rows], ["Claude Opus 5", "Gemini 3.7 Flash"])
        self.assertEqual(rows[0].provider, "Anthropic")
        self.assertEqual(rows[1].category_scores["coding"], 35.1)


if __name__ == "__main__":
    unittest.main()
