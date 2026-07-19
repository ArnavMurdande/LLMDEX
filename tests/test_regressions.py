from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.scoring import score_dataset
from pipeline.sentiment_pipeline import _scrape_model_mentions
from pipeline.gemini_advisor import _extract_compact_snapshot
from pipeline.merge_data import merge_rows_with_provenance
from scraper.scrape_artificialanalysis import _model_to_scraped_row, _number


class SentimentPipelineRegressionTests(unittest.TestCase):
    def test_fresh_scrape_returns_same_list_contract_as_cache_hit(self):
        mention = {
            "source": "HackerNews",
            "text": "This AI model has impressive benchmark performance",
            "score": 10,
            "url": "https://example.test/item",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "pipeline.sentiment_pipeline._load_mention_cache",
                    return_value=None,
                ),
                patch(
                    "pipeline.sentiment_pipeline.scrape_reddit_mentions",
                    return_value=[],
                ),
                patch(
                    "pipeline.sentiment_pipeline.scrape_hackernews_mentions",
                    return_value=[mention],
                ),
                patch(
                    "pipeline.sentiment_pipeline.scrape_github_mentions",
                    return_value=[],
                ),
                patch("pipeline.sentiment_pipeline._save_mention_cache"),
            ):
                model_name, mentions = _scrape_model_mentions(
                    "Example Model", str(Path(temp_dir))
                )

        self.assertEqual(model_name, "Example Model")
        self.assertIsInstance(mentions, list)
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0]["source"], "HackerNews")


class RankingRegressionTests(unittest.TestCase):
    def test_stronger_tier_two_model_can_outrank_weaker_tier_one_model(self):
        rows = pd.DataFrame(
            [
                {
                    "model_name": "Cross Referenced But Weaker",
                    "data_tier": 1,
                    "intelligence_score": 35.0,
                    "coding_score": 35.0,
                    "gpqa": 35.0,
                    "arena_elo": 1300,
                    "blended_cost_per_1m": 2.0,
                },
                {
                    "model_name": "Newer Strong Model",
                    "data_tier": 2,
                    "intelligence_score": 60.0,
                    "coding_score": 60.0,
                    "gpqa": 80.0,
                    "blended_cost_per_1m": 2.0,
                },
            ]
        )

        scored = score_dataset(rows).set_index("model_name")

        self.assertGreater(
            scored.loc["Newer Strong Model", "adjusted_performance"],
            scored.loc["Cross Referenced But Weaker", "adjusted_performance"],
        )
        self.assertEqual(scored.loc["Newer Strong Model", "performance_rank"], 1)
        self.assertEqual(
            scored.loc["Cross Referenced But Weaker", "performance_rank"], 2
        )

    def test_performance_rank_follows_artificial_analysis_intelligence(self):
        rows = pd.DataFrame(
            [
                {
                    "model_name": "Claude Fable 5 (with fallback)",
                    "intelligence_score": 60.0,
                    "source_rank": 1,
                },
                {
                    "model_name": "GPT-5.6 Sol (max)",
                    "intelligence_score": 59.0,
                    "source_rank": 2,
                },
                {
                    "model_name": "Gemini 3.1 Pro",
                    "intelligence_score": 46.0,
                    "source_rank": 30,
                },
            ]
        )

        scored = score_dataset(rows).set_index("model_name")
        self.assertEqual(scored.loc["Claude Fable 5 (with fallback)", "performance_rank"], 1)
        self.assertEqual(scored.loc["GPT-5.6 Sol (max)", "performance_rank"], 2)
        self.assertEqual(scored.loc["Gemini 3.1 Pro", "performance_rank"], 3)


class ArtificialAnalysisScraperRegressionTests(unittest.TestCase):
    def test_expanded_fields_are_typed_and_raw_values_are_preserved(self):
        details = {
            "Model": "GPT-5.6 Sol (max)",
            "Context Window": "1M",
            "Creator": "OpenAI",
            "License": "Proprietary",
            "Artificial Analysis Intelligence Index": "59",
            "Artificial Analysis Omniscience Index": "22",
            "Terminal-Bench Hard Agentic Coding & Terminal Use": "66%",
            "Terminal-Bench v2.1 Agentic Coding & Terminal Use": "88%",
            "SciCode Coding": "56%",
            "ITBench-AA Kubernetes Incident Root-Cause Analysis": "56%",
            "Blended USD/1M Tokens": "$4.35",
            "Input Price USD/1M Tokens": "$5.00",
            "Output Price USD/1M Tokens": "$30.00",
            "Median Tokens/s": "53",
            "P95 Tokens/s": "81",
            "Latency First Chunk (s)": "130.34",
            "Total Response (s)": "139.75",
        }
        row = _model_to_scraped_row(
            {
                "name": details["Model"],
                "source_rank": 2,
                "model_url": "https://artificialanalysis.ai/models/gpt-5-6-sol",
                "providers_url": "https://artificialanalysis.ai/models/gpt-5-6-sol/providers",
                "source_details": details,
            }
        )

        self.assertEqual(row.intelligence_score, 59.0)
        self.assertEqual(row.coding_score, 66.5)
        self.assertEqual(row.output_cost_per_1m, 30.0)
        self.assertEqual(row.source_details, details)
        self.assertEqual(_number("−10"), -10.0)

    def test_reasoning_effort_variants_are_not_collapsed(self):
        base = {
            "source": "Artificial Analysis",
            "scraped_at": "2026-07-19T00:00:00+00:00",
            "provider": "OpenAI",
            "confidence": 1.0,
        }
        merged = merge_rows_with_provenance(
            [
                {
                    **base,
                    "model_name": "GPT-5.6 Sol (max)",
                    "intelligence_score": 59.0,
                    "source_rank": 2,
                },
                {
                    **base,
                    "model_name": "GPT-5.6 Sol (xhigh)",
                    "intelligence_score": 58.0,
                    "source_rank": 3,
                },
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual({row["source_rank"] for row in merged}, {2, 3})


class AdvisorRegressionTests(unittest.TestCase):
    def test_query_match_outside_top_models_is_included_in_context(self):
        rows = [
            {
                "canonical_name": f"Model {index}",
                "provider": "Example",
                "performance_rank": index,
                "value_rank": index,
                "efficiency_rank": index,
            }
            for index in range(1, 80)
        ]
        rows[-1]["canonical_name"] = "Special Comparison Model"
        rows[-1]["aliases"] = ["special-model"]

        snapshot = _extract_compact_snapshot(
            rows,
            user_query="Compare special-model with the performance leader",
        )

        names = {row["model_name"] for row in snapshot}
        self.assertIn("Special Comparison Model", names)
        self.assertIn("Model 1", names)
        self.assertLessEqual(len(snapshot), 50)


if __name__ == "__main__":
    unittest.main()
