from __future__ import annotations

import unittest
from pathlib import Path

from scraper.scrape_llmstats import (
    _extract_top_models,
    parse_llmstats_table,
)


FIXTURES = Path(__file__).parent / "fixtures"


class LLMStatsFixtureTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
