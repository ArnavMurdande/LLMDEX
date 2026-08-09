from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.scoring import score_dataset
from pipeline.sentiment_pipeline import (
    _extract_community_examples,
    _filter_model_relevance,
    _sample_mentions,
    _scrape_model_mentions,
    get_models_for_sentiment,
)
from pipeline.build_family_history import (
    build_family_history,
    extract_embedded_model_catalog,
    extract_release_date,
    infer_release_date,
)
from pipeline.gemini_advisor import _extract_compact_snapshot, _load_dataset
from pipeline.merge_data import merge_rows_with_provenance
from scraper.scrape_artificialanalysis import _model_to_scraped_row, _number


class SentimentPipelineRegressionTests(unittest.TestCase):
    def test_fresh_scrape_returns_same_list_contract_as_cache_hit(self):
        mention = {
            "source": "HackerNews",
            "text": "Example Model has impressive benchmark performance",
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
                patch(
                    "pipeline.sentiment_pipeline.scrape_web_mentions",
                    return_value=[],
                ),
                patch(
                    "pipeline.sentiment_pipeline.scrape_x_mentions",
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

    def test_sentiment_tracks_top_ten_distinct_current_models(self):
        rows = [
            {
                "canonical_name": "GPT-5.6 Sol (max)",
                "performance_rank": 1,
            },
            {
                "canonical_name": "GPT-5.6 Sol (xhigh)",
                "performance_rank": 2,
            },
        ]
        rows.extend(
            {
                "canonical_name": f"Current Model {rank}",
                "performance_rank": rank,
            }
            for rank in range(3, 14)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "latest.json"
            index_path.write_text(json.dumps(rows), encoding="utf-8")
            selected = get_models_for_sentiment(str(index_path))

        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[0], "GPT-5.6 Sol")
        self.assertNotIn("GPT-5.6 Sol (xhigh)", selected)

    def test_sentiment_rejects_broad_family_false_positives(self):
        mentions = [
            {"text": "Claude Sonnet 5 is much better for this workflow."},
            {"text": "Claude usage limits made me switch to another model."},
            {"text": "Sonnet 4.5 remains affordable."},
        ]

        relevant = _filter_model_relevance("Claude Sonnet 5", mentions)

        self.assertEqual(len(relevant), 1)
        self.assertIn("Sonnet 5", relevant[0]["text"])

    def test_sentiment_samples_only_balanced_community_sources(self):
        mentions = [
            {
                "source": "GitHub",
                "text": "Files changed by Claude Sonnet 5",
                "score": 99,
            },
            {
                "source": "HackerNews",
                "text": "Claude Sonnet 5 feels slow for coding",
                "score": 2,
            },
            {
                "source": "Reddit",
                "text": "I switched to Claude Sonnet 5 because it is more reliable",
                "score": 1,
            },
        ]

        sampled = _sample_mentions("Claude Sonnet 5", mentions)

        self.assertEqual({item["source"] for item in sampled}, {"HackerNews", "Reddit"})
        self.assertEqual(sampled[0]["source"], "Reddit")

    def test_community_examples_reject_developer_artifacts(self):
        mentions = [
            {
                "source": "GitHub",
                "text": "Updated to Claude Sonnet 5. Files changed: src/agent/prompt.rs",
                "score": 100,
            },
            {
                "source": "Reddit",
                "text": "I tried Claude Sonnet 5 for coding and it is much more reliable.",
                "score": 4,
            },
        ]

        examples = _extract_community_examples(mentions, max_quotes=3)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["source"], "Reddit")


class FamilyGrowthRegressionTests(unittest.TestCase):
    def test_fable_is_preserved_and_effort_variants_are_collapsed(self):
        rows = [
            {
                "canonical_name": "Claude Fable 5 (with fallback)",
                "provider": "Anthropic",
                "adjusted_performance": 60,
                "performance_rank": 1,
            },
            {
                "canonical_name": "Claude Opus 4.8 (max)",
                "provider": "Anthropic",
                "adjusted_performance": 58,
                "performance_rank": 2,
            },
            {
                "canonical_name": "Claude Opus 4.8 (xhigh)",
                "provider": "Anthropic",
                "adjusted_performance": 57,
                "performance_rank": 3,
            },
        ]

        history = build_family_history(rows)

        self.assertIn("Claude Fable", history)
        self.assertEqual(history["Claude Fable"][0]["name"], "Claude Fable 5")
        self.assertEqual(len(history["Claude Opus"]), 1)
        self.assertEqual(history["Claude Opus"][0]["variant_count"], 2)

    def test_release_dates_are_extracted_from_model_records_and_names(self):
        page = "When was Example released? It was released on July 9, 2026."

        self.assertEqual(extract_release_date(page), "2026-07-09")
        self.assertEqual(infer_release_date("Model Preview 20250805"), "2025-08-05")

    def test_embedded_historical_catalog_is_parsed(self):
        page = (
            r'prefix {\"id\":\"1\",\"slug\":\"claude-3-opus\",'
            r'\"name\":\"Claude 3 Opus\",\"shortName\":\"Opus\",'
            r'\"releaseDate\":\"2024-03-04\",\"intelligenceIndex\":11.8,'
            r'\"deprecated\":true,\"creator\":{\"name\":\"Anthropic\"},'
            r'\"outputModalityVideo\":false} suffix'
        )

        rows = extract_embedded_model_catalog(page)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_name"], "Claude 3 Opus")
        self.assertEqual(rows[0]["release_date"], "2024-03-04")


class RankingRegressionTests(unittest.TestCase):
    def test_blended_cost_is_published_from_input_and_output_prices(self):
        scored = score_dataset(
            pd.DataFrame(
                [
                    {
                        "model_name": "Priced Model",
                        "intelligence_score": 50,
                        "input_cost_per_1m": 2,
                        "output_cost_per_1m": 8,
                    }
                ]
            )
        )
        self.assertAlmostEqual(scored.iloc[0]["blended_cost_per_1m"], 4.4)

    def test_value_requires_price_and_only_redistributes_missing_speed(self):
        scored = score_dataset(
            pd.DataFrame(
                [
                    {
                        "model_name": "Unpriced Model",
                        "intelligence_score": 50,
                        "tokens_per_second": 100,
                    },
                    {
                        "model_name": "Priced Model Without Speed",
                        "intelligence_score": 50,
                        "blended_cost_per_1m": 2,
                    },
                ]
            )
        ).set_index("model_name")

        self.assertTrue(pd.isna(scored.loc["Unpriced Model", "composite_index"]))
        self.assertTrue(pd.isna(scored.loc["Unpriced Model", "value_rank"]))

        priced = scored.loc["Priced Model Without Speed"]
        expected = round(
            (
                priced["adjusted_performance"] * 0.50
                + priced["cost_index"] * 0.30
            )
            / 0.80,
            2,
        )
        self.assertEqual(priced["composite_index"], expected)
        self.assertEqual(priced["value_rank"], 1)

    def test_efficiency_percentiles_use_only_the_eligible_population(self):
        scored = score_dataset(
            pd.DataFrame(
                [
                    {
                        "model_name": "Excluded Low Performance",
                        "intelligence_score": 10,
                        "blended_cost_per_1m": 0.1,
                    },
                    {
                        "model_name": "Eligible Lower Efficiency",
                        "intelligence_score": 25,
                        "blended_cost_per_1m": 10,
                    },
                    {
                        "model_name": "Eligible Higher Efficiency",
                        "intelligence_score": 50,
                        "blended_cost_per_1m": 1,
                    },
                    {
                        "model_name": "Eligible Higher Efficiency Twin",
                        "intelligence_score": 50,
                        "blended_cost_per_1m": 1,
                    },
                ]
            )
        ).set_index("model_name")

        excluded = scored.loc["Excluded Low Performance"]
        self.assertTrue(pd.isna(excluded["efficiency_score"]))
        self.assertTrue(pd.isna(excluded["efficiency_rank"]))
        self.assertEqual(
            scored.loc["Eligible Lower Efficiency", "efficiency_score"], 0.0
        )
        self.assertEqual(
            scored.loc["Eligible Higher Efficiency", "efficiency_score"], 100.0
        )
        self.assertEqual(
            scored.loc["Eligible Higher Efficiency Twin", "efficiency_score"],
            100.0,
        )
        self.assertEqual(
            scored.loc["Eligible Higher Efficiency", "efficiency_rank"], 1
        )

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
    def test_family_contract_is_loaded_and_consensus_fields_reach_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "families.json"
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "canonical_family_name": "Consensus Leader",
                                "provider": "Example",
                                "llmdex_rank": 1,
                                "llmdex_score": 100,
                                "score_status": "consensus",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = _load_dataset(str(path))

        snapshot = _extract_compact_snapshot(rows)
        self.assertEqual(snapshot[0]["model_name"], "Consensus Leader")
        self.assertEqual(snapshot[0]["llmdex_score"], 100.0)
        self.assertEqual(snapshot[0]["score_status"], "consensus")

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
