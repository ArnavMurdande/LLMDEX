from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.consensus import (
    build_consensus,
    descending_average_ranks,
    rank_to_percentile,
    select_aa_representatives,
)
from pipeline.identity import (
    build_family_registry,
    generate_match_candidate,
    normalize_family_name,
    parse_model_identity,
    parse_variant_suffixes,
)


class IdentityContractTests(unittest.TestCase):
    def setUp(self):
        self.aa_rows = [
            {
                "canonical_name": "GPT-5.6 Sol (max)",
                "provider": "OpenAI",
                "model_url": "https://artificialanalysis.ai/models/gpt-5-6-sol",
                "model_id": "aa-sol-max",
                "performance_rank": 1,
                "intelligence_score": 59.0,
            },
            {
                "canonical_name": "GPT-5.6 Sol (xhigh)",
                "provider": "OpenAI",
                "model_url": "https://artificialanalysis.ai/models/gpt-5-6-sol-xhigh",
                "model_id": "aa-sol-xhigh",
                "performance_rank": 2,
                "intelligence_score": 58.0,
            },
            {
                "canonical_name": "Claude Fable 5 (adaptive, with fallback)",
                "provider": "Anthropic",
                "performance_rank": 3,
                "intelligence_score": 57.0,
            },
        ]
        self.registry = build_family_registry(self.aa_rows)

    def test_variant_metadata_is_preserved(self):
        parsed = parse_model_identity(
            "GPT-5.6 Sol (max, thinking, 128k)", "OpenAI"
        )
        self.assertEqual(parsed["canonical_family_name"], "GPT-5.6 Sol")
        self.assertEqual(parsed["reasoning_effort"], "max")
        self.assertIn("thinking", parsed["deployment_profile"])
        self.assertIn("128k", [item.lower() for item in parsed["variant_metadata"]["context_labels"]])
        self.assertNotEqual(parsed["variant_id"], parsed["family_id"])

    def test_adaptive_fallback_and_parenthetical_are_retained(self):
        metadata = parse_variant_suffixes(
            "Claude Fable 5 (adaptive reasoning, with fallback)"
        )
        self.assertIn("adaptive_reasoning", metadata["deployment_profile"])
        self.assertTrue(metadata["fallback_enabled"])
        self.assertEqual(
            metadata["parenthetical_labels"],
            ["adaptive reasoning, with fallback"],
        )

    def test_dates_preview_and_parameter_labels_are_not_destroyed(self):
        name = "Qwen3.5-397B-A17B Preview 20250718"
        parsed = parse_model_identity(name, "Alibaba")
        self.assertIn("Preview", parsed["canonical_family_name"])
        self.assertIn("20250718", parsed["canonical_family_name"])
        self.assertEqual(parsed["parameter_count"]["total_billions"], 397.0)
        self.assertEqual(parsed["parameter_count"]["active_billions"], 17.0)

    def test_exact_source_id_match(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Different display",
                "provider": "OpenAI",
                "source_model_id": "aa-sol-max",
            },
            self.registry,
        )
        self.assertEqual(result.match_status, "matched_exact")
        self.assertEqual(result.match_method, "exact_source_model_id")

    def test_exact_url_match(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Different display",
                "provider": "OpenAI",
                "source_model_url": "https://artificialanalysis.ai/models/gpt-5-6-sol",
            },
            self.registry,
        )
        self.assertEqual(result.match_status, "matched_exact")

    def test_approved_alias_match(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Sol",
                "provider": "OpenAI",
            },
            self.registry,
            {
                "approved_aliases": [
                    {
                        "source": "llmstats",
                        "source_name": "Sol",
                        "family_id": "openai/gpt-5-6-sol",
                    }
                ]
            },
        )
        self.assertEqual(result.match_status, "matched_manual")
        self.assertEqual(result.match_confidence, 1.0)

    def test_provider_family_version_match(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "GPT 5.6 Sol",
                "provider": "OpenAI",
            },
            self.registry,
        )
        self.assertEqual(result.match_status, "matched_family")
        self.assertEqual(result.match_confidence, 0.95)

    def test_manual_provider_alias_can_link_same_named_family(self):
        registry = build_family_registry(
            [{"canonical_name": "Kimi K3", "provider": "Kimi"}]
        )
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Kimi K3",
                "provider": "MoonshotAI",
            },
            registry,
            overrides={
                "approved_aliases": [
                    {
                        "source": "llmstats",
                        "source_name": "Kimi K3",
                        "family_id": "kimi/kimi-k3",
                    }
                ]
            },
        )
        self.assertEqual(result.match_status, "matched_manual")
        self.assertEqual(result.candidate_family_id, "kimi/kimi-k3")

    def test_fuzzy_match_is_review_only(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "GPT-5.6 Solar",
                "provider": "OpenAI",
            },
            self.registry,
        )
        self.assertIn(result.match_status, {"identity_unresolved", "ambiguous"})
        self.assertLessEqual(result.match_confidence, 0.8)

    def test_unmatched_model_is_retained_as_source_missing(self):
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Entirely New Model 1",
                "provider": "New Lab",
            },
            self.registry,
        )
        self.assertEqual(result.match_status, "source_missing")
        self.assertIsNone(result.candidate_family_id)

    def test_qwen_separator_differences_match_exact_family(self):
        registry = build_family_registry(
            [{"canonical_name": "Qwen3.5-397B-A17B", "provider": "Alibaba"}]
        )
        result = generate_match_candidate(
            {
                "source": "llmstats",
                "source_name": "Qwen3.5 397B A17B",
                "provider": "Alibaba",
            },
            registry,
        )
        self.assertIn(result.match_status, {"matched_exact", "matched_family"})


class ConsensusScoreTests(unittest.TestCase):
    def test_tie_aware_ranks_and_percentiles(self):
        self.assertEqual(descending_average_ranks([10, 10, 5]), [1.5, 1.5, 3.0])
        self.assertEqual(rank_to_percentile(1.5, 3), 75.0)
        self.assertEqual(rank_to_percentile(1.0, 1), 100.0)

    def test_representative_selection_is_deterministic(self):
        rows = [
            {
                "family_id": "lab/model",
                "variant_id": "lab/model:xhigh",
                "source_name": "Model (xhigh)",
                "performance_rank": 2,
                "intelligence_score": 60,
            },
            {
                "family_id": "lab/model",
                "variant_id": "lab/model:max",
                "source_name": "Model (max)",
                "performance_rank": 1,
                "intelligence_score": 59,
            },
        ]
        selected = select_aa_representatives(
            rows, selected_at="2026-07-23T00:00:00+00:00"
        )
        self.assertEqual(
            selected["lab/model"]["aa_representative_variant_id"],
            "lab/model:max",
        )

    def test_missing_source_and_identity_review_never_become_zero(self):
        aa = [
            {
                "canonical_name": "Model A (max)",
                "model_name": "Model A (max)",
                "provider": "Lab",
                "performance_rank": 1,
                "source_rank": 1,
                "intelligence_score": 60.0,
                "aa_official_coding_index": 50.0,
            },
            {
                "canonical_name": "Model A (high)",
                "model_name": "Model A (high)",
                "provider": "Lab",
                "performance_rank": 2,
                "source_rank": 2,
                "intelligence_score": 59.0,
                "aa_official_coding_index": 49.0,
            },
            {
                "canonical_name": "Model B",
                "model_name": "Model B",
                "provider": "Lab",
                "performance_rank": 3,
                "source_rank": 3,
                "intelligence_score": 58.0,
            },
        ]
        from pipeline.identity import enrich_aa_rows

        enriched, registry = enrich_aa_rows(aa)
        llmstats = [
            {
                "source": "llmstats",
                "source_name": "Model A",
                "provider": "Lab",
                "source_model_id": "model-a",
                "source_model_url": "https://example.test/model-a",
                "general_score": 55.0,
                "general_rank": 1.0,
                "category_scores": {"coding": 45.0},
                "category_ranks": {"coding": 1.0},
            },
            {
                "source": "llmstats",
                "source_name": "Model Bee",
                "provider": "Lab",
                "source_model_id": "model-bee",
                "source_model_url": "https://example.test/model-bee",
                "general_score": 54.0,
                "general_rank": 2.0,
                "category_scores": {"coding": 44.0},
                "category_ranks": {"coding": 2.0},
            },
        ]
        output = build_consensus(
            enriched,
            llmstats,
            registry,
            generated_at="2026-07-23T00:00:00+00:00",
        )
        variants = {
            row["canonical_name"]: row for row in output["aa_rows"]
        }
        self.assertEqual(variants["Model A (max)"]["llmdex_score"], 100.0)
        self.assertEqual(variants["Model A (high)"]["llmdex_score"], 100.0)
        self.assertEqual(
            variants["Model A (high)"]["score_status"],
            "consensus",
        )
        self.assertIsNone(variants["Model B"]["llmdex_score"])
        self.assertEqual(variants["Model B"]["performance_rank"], 3)
        self.assertNotEqual(variants["Model B"].get("llmdex_score"), 0)

    def test_agreement_does_not_change_score_or_rank(self):
        # The formula is defined only by the two percentiles.
        left, right = 100.0, 0.0
        score = (left + right) / 2
        agreement = 100 - abs(left - right)
        self.assertEqual(score, 50.0)
        self.assertEqual(agreement, 0.0)


if __name__ == "__main__":
    unittest.main()
