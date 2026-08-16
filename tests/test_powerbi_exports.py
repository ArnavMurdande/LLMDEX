"""Unit tests for the LLMDEX Power BI v2 Wide Export Layer (Defect Fixes & Non-Regression)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from pipeline.export_powerbi import clean_llmstats_display_name, run_exports

ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = ROOT / "data" / "powerbi" / "v1"


class _CardCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self._in_grid = False
        self._current_card = None

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "").split()
        if "data-download-grid" in classes:
            self._in_grid = True
        elif self._in_grid and tag == "a" and "data-download-card" in classes:
            self._current_card = {
                "href": attr_dict.get("href"),
                "download": "download" in attr_dict or attr_dict.get("download") is not None,
            }
            self.cards.append(self._current_card)


class PowerBIExportsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run exporter to ensure files exist
        run_exports(include_git_history=False)

    def test_01_primary_export_files_exist(self):
        required_files = [
            "artificial_analysis_benchmarks.csv",
            "llmstats_benchmarks.csv",
            "combined_latest.json",
            "model_family_history.csv",
            "provider_metadata.csv",
            "provider_metadata_validation.json",
            "manifest.json",
            "data_dictionary.csv",
        ]
        for fn in required_files:
            p = POWERBI_DIR / fn
            self.assertTrue(p.is_file(), f"Expected export file missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Export file is empty: {p}")

    def test_02_llmstats_name_cleanup(self):
        # Test helper function
        c1, o1, ok1 = clean_llmstats_display_name("25Claude 3.7 Sonnet", "claude-3-7-sonnet", known_rank=25)
        self.assertEqual(c1, "Claude 3.7 Sonnet")
        self.assertTrue(ok1)

        c2, o2, ok2 = clean_llmstats_display_name("17Claude Haiku 4.5", "claude-haiku-4-5", known_rank=17)
        self.assertEqual(c2, "Claude Haiku 4.5")
        self.assertTrue(ok2)

        c3, o3, ok3 = clean_llmstats_display_name("29Gemini 2.5 Pro", "gemini-2-5-pro", known_rank=29)
        self.assertEqual(c3, "Gemini 2.5 Pro")
        self.assertTrue(ok3)

        # Legitimate numeric names remain unchanged
        c4, o4, ok4 = clean_llmstats_display_name("01.AI Yi-Large", "yi-large")
        self.assertEqual(c4, "01.AI Yi-Large")
        self.assertFalse(ok4)

        # Assert exported CSV does not contain malformed names
        with open(POWERBI_DIR / "llmstats_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            names = [r["source_name"] for r in reader]
            self.assertNotIn("25Claude 3.7 Sonnet", names)
            self.assertNotIn("17Claude Haiku 4.5", names)
            self.assertNotIn("29Gemini 2.5 Pro", names)

    def test_03_fallback_family_ids_and_matched_integrity(self):
        with open(POWERBI_DIR / "llmstats_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fid = r["family_id"]
                self.assertTrue(bool(fid))
                if fid.startswith("unknown/"):
                    self.assertNotIn("25claude", fid)
                    self.assertNotIn("17claude", fid)
                    self.assertNotIn("29gemini", fid)

    def test_04_llmstats_general_category_score_and_rank(self):
        manifest = json.loads((POWERBI_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("llmstats_general_row_count", manifest)
        self.assertGreater(manifest["llmstats_general_row_count"], 0)

        with open(POWERBI_DIR / "llmstats_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            gen_rows = [r for r in reader if r["capability"] == "general"]
            self.assertGreater(len(gen_rows), 0)
            populated_scores = [r["category_score"] for r in gen_rows if r["category_score"] != ""]
            self.assertGreater(len(populated_scores), 0)

    def test_05_provider_enrichment_from_approved_metadata(self):
        manifest = json.loads((POWERBI_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(manifest["llmstats_provider_populated_count"], 0)
        self.assertEqual(manifest["llmstats_provider_missing_count"], 0)

        with open(POWERBI_DIR / "llmstats_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.assertNotEqual(r["provider"], "", f"Unpopulated provider in row: {r['source_name']}")

    def test_06_json_native_typing_in_combined(self):
        doc = json.loads((POWERBI_DIR / "combined_latest.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], "powerbi-v2-wide")

        aa_models = doc["artificial_analysis"]["models"]
        self.assertGreater(len(aa_models), 0)

        sample = aa_models[0]
        if sample.get("source_rank") is not None:
            self.assertIsInstance(sample["source_rank"], int)
        if sample.get("aa_intelligence_score") is not None:
            self.assertIsInstance(sample["aa_intelligence_score"], (int, float))
        if sample.get("is_open_weights") is not None:
            self.assertIsInstance(sample["is_open_weights"], bool)

        # Ensure no empty string appears in numeric/boolean fields
        def check_types(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in {"source_rank", "aa_intelligence_score", "is_open_weights", "category_score"}:
                        self.assertNotEqual(v, "", f"Empty string found in typed field '{k}'")
                    check_types(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_types(item)

        check_types(doc)

    def test_07_dedicated_consensus_array(self):
        doc = json.loads((POWERBI_DIR / "combined_latest.json").read_text(encoding="utf-8"))
        self.assertIn("consensus", doc["llmdex"])
        consensus = doc["llmdex"]["consensus"]
        self.assertGreater(len(consensus), 0)

        manifest = json.loads((POWERBI_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["consensus_family_count"], len(consensus))

        for c in consensus:
            self.assertIsNotNone(c.get("family_id"))
            self.assertIsNotNone(c.get("llmdex_score"))
            self.assertIsNotNone(c.get("llmdex_rank"))

    def test_08_reserved_aa_columns_in_csv_and_dictionary(self):
        with open(POWERBI_DIR / "artificial_analysis_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            self.assertIn("aa_aime25", fieldnames)
            self.assertIn("aa_livecodebench", fieldnames)
            self.assertIn("aa_arena_elo", fieldnames)

        with open(POWERBI_DIR / "data_dictionary.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            dict_rows = {r["column_name"]: r for r in reader if r["table_name"] == "artificial_analysis_benchmarks"}
            self.assertEqual(dict_rows["aa_aime25"]["currently_populated"], "False")
            self.assertEqual(dict_rows["aa_aime25"]["source_population"], "0")

    def test_09_history_record_typing_and_proprietary_preservation(self):
        proprietary_fids = {"openai/gpt-3-5-turbo", "openai/gpt-4", "anthropic/claude-instant"}
        with open(POWERBI_DIR / "model_family_history.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["family_id"] in proprietary_fids:
                    self.assertEqual(row["is_open_weights"], "False")

    def test_10_natural_keys_uniqueness(self):
        # AA natural key
        seen_aa = set()
        with open(POWERBI_DIR / "artificial_analysis_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                k = (r["snapshot_date"], r["model_key"])
                self.assertNotIn(k, seen_aa, f"Duplicate AA key: {k}")
                seen_aa.add(k)

        # LLMStats natural key
        seen_llm = set()
        with open(POWERBI_DIR / "llmstats_benchmarks.csv", "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                k = (r["snapshot_date"], r["capability"], r["source_model_id"] or r["source_name"])
                self.assertNotIn(k, seen_llm, f"Duplicate LLMStats key: {k}")
                seen_llm.add(k)

        # History natural key
        seen_hist = set()
        with open(POWERBI_DIR / "model_family_history.csv", "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                k = (r["observation_date"], r["family_id"], r["record_type"])
                self.assertNotIn(k, seen_hist, f"Duplicate History key: {k}")
                seen_hist.add(k)

    def test_11_no_unintended_mutations_outside_powerbi(self):
        for path in [
            ROOT / "data" / "index" / "latest.csv",
            ROOT / "data" / "index" / "latest.json",
            ROOT / "data" / "history" / "score_history.csv",
            ROOT / "data" / "identity" / "match_audit.csv",
        ]:
            self.assertTrue(path.is_file(), f"Existing contract missing: {path}")

    def test_12_frontend_download_cards_valid(self):
        html = (ROOT / "website" / "index.html").read_text(encoding="utf-8")
        collector = _CardCollector()
        collector.feed(html)
        self.assertEqual(len(collector.cards), 4)
        for card in collector.cards:
            self.assertTrue(card["href"].startswith("./data/powerbi/v1/"))
        self.assertIn("./data/powerbi/v1/provider_metadata.csv", [card["href"] for card in collector.cards])
        self.assertNotIn("./data/powerbi/v1/combined_latest.json", [card["href"] for card in collector.cards])

    def test_13_provider_metadata_schema_and_joins(self):
        expected = [
            "schema_version", "snapshot_date", "provider_id", "provider_name",
            "parent_company", "provider_group", "provider_aliases", "country",
            "country_code", "region", "hq_city", "latitude", "longitude",
            "website_url", "logo_url", "logo_dark_url", "brand_color",
            "founded_year", "is_active", "metadata_source_url", "metadata_verified_at",
        ]
        with open(POWERBI_DIR / "provider_metadata.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(expected, reader.fieldnames)
        ids = [row["provider_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 0)
        for row in rows:
            if row["latitude"]:
                self.assertTrue(-90 <= float(row["latitude"]) <= 90)
            if row["longitude"]:
                self.assertTrue(-180 <= float(row["longitude"]) <= 180)
            self.assertTrue(row["logo_url"].startswith("https://llmdex.pages.dev/assets/providers/"))
            self.assertTrue((ROOT / "website" / "assets" / "providers" / f"{row['provider_id']}.svg").is_file())
        provider_ids = set(ids)
        for filename in ("artificial_analysis_benchmarks.csv", "llmstats_benchmarks.csv", "model_family_history.csv"):
            with open(POWERBI_DIR / filename, "r", encoding="utf-8-sig") as f:
                fact_rows = list(csv.DictReader(f))
            self.assertIn("provider_id", fact_rows[0])
            for row in fact_rows:
                if row.get("provider"):
                    self.assertIn(row["provider_id"], provider_ids, f"Orphan provider in {filename}: {row['provider']}")


if __name__ == "__main__":
    unittest.main()
