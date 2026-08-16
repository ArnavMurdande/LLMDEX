"""Power BI Export Layer (v2 Wide Schema) for LLMDEX.

Reads processed contracts in data/ and outputs Power BI-friendly wide CSV/JSON contracts
under data/powerbi/v1/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = ROOT / "data" / "powerbi" / "v1"

# Schema version constant
SCHEMA_VERSION = "powerbi-v2-wide"

# Empty reserved AA columns
EMPTY_RESERVED_AA_COLS = [
    "aa_aime25",
    "aa_livecodebench",
    "aa_arena_elo",
    "aa_reasoning_score",
    "aa_multimodal_score",
    "aa_cost_per_task_usd",
    "aa_cache_read_cost_usd_per_1m",
    "aa_cache_write_cost_usd_per_1m",
]

# Field typing rules for JSON coercion
INT_FIELDS = {
    "source_rank", "category_rank", "aa_rank", "llmstats_general_rank", "llmdex_rank",
    "global_rank", "value_rank", "efficiency_rank", "performance_rank", "source_population",
    "context_window", "context_window_tokens", "aa_context_window_tokens", "perf_source_count",
    "llmdex_performance_component_count", "llmstats_general_row_count",
    "llmstats_general_missing_score_count", "llmstats_general_missing_rank_count",
    "llmstats_general_missing_provider_count", "llmstats_provider_populated_count",
    "llmstats_provider_missing_count", "llmstats_availability_unknown_count",
    "malformed_name_count_detected", "malformed_name_count_corrected",
    "unresolved_malformed_name_count", "consensus_family_count", "catalog_event_row_count",
    "observed_snapshot_row_count", "number_of_benchmark_columns"
}

BOOL_FIELDS = {
    "is_open_weights", "is_family_representative", "is_current", "is_sota", "is_open_sota",
    "higher_is_better", "nullable", "currently_populated", "fallback_enabled"
}

# Known model series to provider map for exact un-indexed models
KNOWN_MODEL_PROVIDERS = {
    "claude": "Anthropic",
    "gpt": "OpenAI",
    "o3": "OpenAI",
    "o1": "OpenAI",
    "gemini": "Google",
    "gemma": "Google",
    "qwen": "Alibaba",
    "llama": "Meta",
    "mistral": "Mistral",
    "glm": "Z AI",
    "kimi": "Moonshot AI",
    "minimax": "MiniMax",
    "mimo": "Xiaomi",
    "seed": "ByteDance",
    "muse": "Baidu",
    "mai": "Microsoft",
    "longcat": "Meituan",
    "nova": "Amazon",
    "hermes": "Nous Research",
    "nemotron": "NVIDIA",
    "deepseek": "DeepSeek",
    "grok": "xAI",
    "sakana": "Sakana AI",
}


def clean_num(val: Any) -> Optional[str]:
    """Convert numeric value to string without currency symbols, commas, NaN, or Infinity."""
    if val is None or val == "" or val == "N/A" or val == "--":
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if val != val or val == float("inf") or val == float("-inf"):
            return None
        if isinstance(val, int) or val.is_integer():
            return str(int(val))
        return f"{val:.6g}"
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace("%", "").replace(",", "").strip()
        if not cleaned or cleaned in {"N/A", "--", "nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"}:
            return None
        try:
            f = float(cleaned)
            if f != f or f == float("inf") or f == float("-inf"):
                return None
            if f.is_integer():
                return str(int(f))
            return f"{f:.6g}"
        except ValueError:
            return None
    return None


def bool_str(val: Any) -> str:
    """Format boolean value cleanly as 'True' or 'False'."""
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if val_lower in {"true", "1", "yes"}:
            return "True"
        if val_lower in {"false", "0", "no"}:
            return "False"
        return ""
    return "True" if bool(val) else "False"


def derive_is_open_weights(availability_class: Any) -> str:
    """Derive is_open_weights strictly from approved availability_class."""
    avail = str(availability_class or "").lower().strip()
    if avail == "open_weights":
        return "True"
    if avail == "proprietary":
        return "False"
    return ""


def clean_llmstats_display_name(
    source_name: str,
    source_model_id: str = "",
    source_model_url: str = "",
    known_rank: Optional[Any] = None,
) -> Tuple[str, str, bool]:
    """Clean malformed LLMStats display names (e.g., '25Claude 3.7 Sonnet' -> 'Claude 3.7 Sonnet').

    Returns (cleaned_display_name, original_source_name, is_corrected).
    """
    if not source_name or not isinstance(source_name, str):
        return source_name or "", source_name or "", False

    raw_name = source_name.strip()
    match = re.match(r"^(\d+)\s*([A-Za-z].*)$", raw_name)
    if not match:
        return raw_name, raw_name, False

    num_prefix_str, remaining_text = match.group(1), match.group(2).strip()

    rem_norm = re.sub(r"[^a-z0-9]+", "", remaining_text.lower())
    smid_norm = re.sub(r"[^a-z0-9]+", "", (source_model_id or "").lower())

    url_slug = ""
    if source_model_url and isinstance(source_model_url, str):
        url_slug = source_model_url.rstrip("/").rsplit("/", 1)[-1]
    url_norm = re.sub(r"[^a-z0-9]+", "", url_slug.lower())
    raw_norm = re.sub(r"[^a-z0-9]+", "", raw_name.lower())

    if smid_norm and smid_norm.startswith(raw_norm):
        return raw_name, raw_name, False

    corresponds = False
    if smid_norm and (rem_norm in smid_norm or smid_norm in rem_norm or rem_norm[:6] == smid_norm[:6]):
        corresponds = True
    elif url_norm and (rem_norm in url_norm or url_norm in rem_norm):
        corresponds = True

    rank_matches = False
    if known_rank is not None:
        try:
            if int(num_prefix_str) == int(float(str(known_rank))):
                rank_matches = True
        except (ValueError, TypeError):
            pass

    if corresponds or rank_matches:
        return remaining_text, raw_name, True

    return raw_name, raw_name, False


def coerce_val_to_json_type(key: str, val: Any) -> Any:
    """Coerce string values in CSV rows into native JSON types (int, float, bool, None)."""
    if val is None or val == "":
        return None

    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val != val or val == float("inf") or val == float("-inf"):
            return None
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return val

    val_str = str(val).strip()
    if not val_str or val_str in {"N/A", "--", "null", "None"}:
        return None

    if key in BOOL_FIELDS:
        val_lower = val_str.lower()
        if val_lower in {"true", "1", "yes"}:
            return True
        if val_lower in {"false", "0", "no"}:
            return False
        return None

    if key in INT_FIELDS or key.endswith("_rank") or key.endswith("_count"):
        try:
            return int(float(val_str))
        except (ValueError, TypeError):
            pass

    if (
        key.startswith(("benchmark_", "aa_", "llmdex_"))
        or key.endswith(("_score", "_index", "_usd", "_seconds", "_per_1m", "_tokens", "_time", "_elo", "_rate", "_accuracy", "_cost", "_throughput", "_latency"))
        or key in {"gdpval", "lcr", "omniscience", "hle", "gpqa", "scicode", "critpt", "mmmu_pro", "coverage_score", "confidence_factor", "agreement"}
    ):
        try:
            f = float(val_str)
            if f.is_integer():
                return int(f)
            return f
        except (ValueError, TypeError):
            pass

    return val_str


def coerce_row_dict_to_json(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce a dictionary to native JSON types."""
    res = {}
    for k, v in row.items():
        if isinstance(v, dict):
            res[k] = coerce_row_dict_to_json(v)
        elif isinstance(v, list):
            res[k] = [coerce_row_dict_to_json(item) if isinstance(item, dict) else coerce_val_to_json_type(k, item) for item in v]
        else:
            res[k] = coerce_val_to_json_type(k, v)
    return res


def load_json_file(path: Path) -> Any:
    """Defensively load JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required contract file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed JSON contract at {path}: {exc}") from exc


def load_csv_file(path: Path) -> List[Dict[str, str]]:
    """Defensively load CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required contract file: {path}")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as exc:
        raise ValueError(f"Malformed CSV contract at {path}: {exc}") from exc


def get_git_commit_sha() -> Optional[str]:
    """Get current Git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def generate_aa_model_key(r: Dict[str, Any], vid_counts: Counter) -> str:
    """Generate a stable model_key for Artificial Analysis records."""
    vid = (r.get("variant_id") or "").strip()
    slug = (r.get("model_slug") or "").strip()
    sname = (r.get("source_name") or r.get("canonical_name") or "").strip()

    if vid and vid_counts[vid] == 1:
        return vid
    if vid and slug:
        return f"{vid}:{slug}"
    if vid and sname:
        clean_s = re.sub(r"[^a-zA-Z0-9]+", "-", sname.lower()).strip("-")
        return f"{vid}:{clean_s}"
    if slug:
        return slug
    if sname:
        return re.sub(r"[^a-zA-Z0-9]+", "-", sname.lower()).strip("-")
    return r.get("model_id") or "unknown_aa_model"


def derive_aa_source_model_id(r: Dict[str, Any]) -> str:
    """Approved fallback order for source_model_id in AA."""
    smid = r.get("source_model_id")
    if smid and str(smid).strip():
        return str(smid).strip()

    mslug = r.get("model_slug")
    if mslug and str(mslug).strip():
        return str(mslug).strip()

    murl = r.get("source_model_url") or r.get("model_url")
    if murl and isinstance(murl, str) and "/models/" in murl:
        slug = murl.rsplit("/models/", 1)[-1].split("/")[0].strip()
        if slug:
            return slug

    return ""


# -------------------------------------------------------------------------
# Export 1: Artificial Analysis Wide CSV
# -------------------------------------------------------------------------

AA_WIDE_COLUMNS = [
    # Schema & Date
    "schema_version",
    "snapshot_date",
    "source",
    # Identity
    "model_key",
    "family_id",
    "variant_id",
    "canonical_name",
    "source_name",
    "provider",
    "creator",
    "availability_class",
    "license_type",
    "is_open_weights",
    "is_family_representative",
    "source_model_id",
    "source_model_url",
    "source_rank",
    "methodology_version",
    # Source-Native Metrics (aa_ prefix)
    "aa_intelligence_score",
    "aa_official_coding_index",
    "aa_omniscience_index",
    "aa_context_window_tokens",
    # Benchmarks
    "aa_gdpval",
    "aa_terminalbench_hard",
    "aa_terminalbench_v21",
    "aa_tau2",
    "aa_tau3_banking",
    "aa_lcr",
    "aa_omniscience_accuracy",
    "aa_non_hallucination_rate",
    "aa_hle",
    "aa_gpqa",
    "aa_scicode",
    "aa_ifbench",
    "aa_critpt",
    "aa_mmmu_pro",
    "aa_apex_agents",
    "aa_itbench",
    "aa_aime25",
    "aa_livecodebench",
    "aa_arena_elo",
    "aa_reasoning_score",
    "aa_multimodal_score",
    # Pricing
    "aa_cost_per_task_usd",
    "aa_input_cost_usd_per_1m",
    "aa_output_cost_usd_per_1m",
    "aa_blended_cost_usd_per_1m",
    "aa_cache_read_cost_usd_per_1m",
    "aa_cache_write_cost_usd_per_1m",
    # Speed
    "aa_tokens_per_second",
    "aa_speed_p5_tokens_per_second",
    "aa_speed_p25_tokens_per_second",
    "aa_speed_p75_tokens_per_second",
    "aa_speed_p95_tokens_per_second",
    # Latency
    "aa_latency_seconds",
    "aa_latency_first_token_seconds",
    "aa_latency_p5_seconds",
    "aa_latency_p25_seconds",
    "aa_latency_p75_seconds",
    "aa_latency_p95_seconds",
    "aa_total_response_time_seconds",
    "aa_reasoning_time_seconds",
    # LLMDEX-Derived Metrics (llmdex_ prefix)
    "llmdex_adjusted_performance",
    "llmdex_cost_index",
    "llmdex_speed_index",
    "llmdex_coverage_score",
    "llmdex_confidence_factor",
    "llmdex_composite_index",
    "llmdex_efficiency_score",
    "llmdex_performance_rank",
    "llmdex_value_rank",
    "llmdex_efficiency_rank",
    "llmdex_performance_component_count",
]

AA_FIELD_MAPPING = {
    "aa_intelligence_score": ["intelligence_score"],
    "aa_official_coding_index": ["aa_official_coding_index", "coding_score"],
    "aa_omniscience_index": ["omniscience_index"],
    "aa_context_window_tokens": ["context_window"],
    "aa_gdpval": ["gdpval"],
    "aa_terminalbench_hard": ["terminalbench_hard"],
    "aa_terminalbench_v21": ["terminalbench_v21"],
    "aa_tau2": ["tau2"],
    "aa_tau3_banking": ["tau3_banking"],
    "aa_lcr": ["lcr"],
    "aa_omniscience_accuracy": ["omniscience"],
    "aa_non_hallucination_rate": ["omniscience_hallucination"],
    "aa_hle": ["hle"],
    "aa_gpqa": ["gpqa"],
    "aa_scicode": ["scicode"],
    "aa_ifbench": ["ifbench"],
    "aa_critpt": ["critpt"],
    "aa_mmmu_pro": ["mmmu_pro"],
    "aa_apex_agents": ["apex_agents"],
    "aa_itbench": ["itbench"],
    "aa_aime25": ["aime25"],
    "aa_livecodebench": ["livecodebench"],
    "aa_arena_elo": ["arena_elo"],
    "aa_reasoning_score": ["reasoning_score"],
    "aa_multimodal_score": ["multimodal_score"],
    "aa_cost_per_task_usd": ["cost_per_task"],
    "aa_input_cost_usd_per_1m": ["input_cost_per_1m"],
    "aa_output_cost_usd_per_1m": ["output_cost_per_1m"],
    "aa_blended_cost_usd_per_1m": ["blended_cost_per_1m"],
    "aa_cache_read_cost_usd_per_1m": ["cache_read_cost_per_1m"],
    "aa_cache_write_cost_usd_per_1m": ["cache_write_cost_per_1m"],
    "aa_tokens_per_second": ["tokens_per_second"],
    "aa_speed_p5_tokens_per_second": ["speed_p5"],
    "aa_speed_p25_tokens_per_second": ["speed_p25"],
    "aa_speed_p75_tokens_per_second": ["speed_p75"],
    "aa_speed_p95_tokens_per_second": ["speed_p95"],
    "aa_latency_seconds": ["latency_seconds"],
    "aa_latency_first_token_seconds": ["latency_first_token"],
    "aa_latency_p5_seconds": ["latency_p5"],
    "aa_latency_p25_seconds": ["latency_p25"],
    "aa_latency_p75_seconds": ["latency_p75"],
    "aa_latency_p95_seconds": ["latency_p95"],
    "aa_total_response_time_seconds": ["total_response_time"],
    "aa_reasoning_time_seconds": ["reasoning_time"],
    "llmdex_adjusted_performance": ["adjusted_performance"],
    "llmdex_cost_index": ["cost_index"],
    "llmdex_speed_index": ["speed_index"],
    "llmdex_coverage_score": ["coverage_score"],
    "llmdex_confidence_factor": ["confidence_factor"],
    "llmdex_composite_index": ["composite_index"],
    "llmdex_efficiency_score": ["efficiency_score"],
    "llmdex_performance_rank": ["performance_rank"],
    "llmdex_value_rank": ["value_rank"],
    "llmdex_efficiency_rank": ["efficiency_rank"],
    "llmdex_performance_component_count": ["perf_source_count"],
}


def build_aa_wide_csv(aa_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], int]:
    vid_counts = Counter(r.get("variant_id") for r in aa_records if r.get("variant_id"))
    rows: List[Dict[str, str]] = []
    seen_keys: Set[Tuple[str, str]] = set()
    corrected_open_weights_count = 0

    for r in aa_records:
        snap_date = (r.get("snapshot_date") or r.get("last_updated") or "2026-07-24").strip()
        model_key = generate_aa_model_key(r, vid_counts)
        nat_key = (snap_date, model_key)
        if nat_key in seen_keys:
            continue
        seen_keys.add(nat_key)

        avail_class = (r.get("availability_class") or "").strip()
        is_ow = derive_is_open_weights(avail_class)

        legacy_open_source = bool_str(r.get("open_source"))
        if is_ow != legacy_open_source and avail_class in {"open_weights", "proprietary"}:
            corrected_open_weights_count += 1

        bm_breakdown = r.get("benchmark_breakdown", {})
        aa_bm_dict = bm_breakdown.get("Artificial Analysis", {}) if isinstance(bm_breakdown, dict) else {}

        row: Dict[str, str] = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_date": snap_date,
            "source": "Artificial Analysis",
            "model_key": model_key,
            "family_id": (r.get("family_id") or "").strip(),
            "variant_id": (r.get("variant_id") or "").strip(),
            "canonical_name": (r.get("canonical_name") or r.get("model_name") or "").strip(),
            "source_name": (r.get("source_name") or r.get("model_name") or "").strip(),
            "provider": (r.get("provider") or "").strip(),
            "creator": (r.get("creator") or "").strip(),
            "availability_class": avail_class,
            "license_type": (r.get("license_type") or "").strip(),
            "is_open_weights": is_ow,
            "is_family_representative": bool_str(r.get("is_family_representative")),
            "source_model_id": derive_aa_source_model_id(r),
            "source_model_url": (r.get("source_model_url") or r.get("model_url") or "").strip(),
            "source_rank": clean_num(r.get("source_rank") or r.get("performance_rank")) or "",
            "methodology_version": (r.get("methodology_version") or "Artificial Analysis v4.1").strip(),
        }

        for col, field_keys in AA_FIELD_MAPPING.items():
            val_num = None
            for fk in field_keys:
                if r.get(fk) is not None:
                    val_num = r.get(fk)
                    break
                elif isinstance(aa_bm_dict, dict) and aa_bm_dict.get(fk) is not None:
                    val_num = aa_bm_dict.get(fk)
                    break
            row[col] = clean_num(val_num) or ""

        rows.append(row)

    rows.sort(key=lambda x: (x["snapshot_date"], x["model_key"]))
    return rows, corrected_open_weights_count


# -------------------------------------------------------------------------
# Export 2: LLMStats Wide CSV
# -------------------------------------------------------------------------

CAPABILITY_NAMES = [
    "general",
    "coding",
    "math",
    "reasoning",
    "writing",
    "research",
    "long_context",
    "tool_calling",
]


def normalize_benchmark_column_name(raw_name: str) -> str:
    """Normalize raw benchmark name to lowercase ascii-safe column name with benchmark_ prefix."""
    s = raw_name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s.startswith("benchmark_"):
        s = f"benchmark_{s}"
    return s


def build_llmstats_wide_csv(
    llmstats_general: Dict[str, Any],
    capability_files: Dict[str, Dict[str, Any]],
    families_contract: Dict[str, Any],
    identity_registry: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str], Dict[str, int]]:
    # Build lookup maps for provider & availability enrichment
    family_map = {r["family_id"]: r for r in families_contract.get("rows", []) if r.get("family_id")}
    registry_index: Dict[str, Dict[str, Any]] = {}

    if isinstance(identity_registry, dict):
        for key, entry in identity_registry.items():
            if isinstance(entry, dict):
                registry_index[key.lower()] = entry
                if "/" in key:
                    registry_index[key.rsplit("/", 1)[-1].lower()] = entry
                for vkey in entry.get("variants", {}).keys():
                    registry_index[vkey.lower()] = entry
                    if ":" in vkey:
                        registry_index[vkey.rsplit(":", 1)[-1].lower()] = entry
                for salias in entry.get("source_aliases", {}).get("llmstats", []):
                    registry_index[salias.lower()] = entry
                for salias in entry.get("source_aliases", {}).get("artificial_analysis", []):
                    registry_index[salias.lower()] = entry

    def resolve_provider_from_approved_sources(smid: str, cname: str, fid: str, surl: str) -> str:
        # 1. Family contract lookup
        if fid and fid in family_map and family_map[fid].get("provider"):
            return family_map[fid]["provider"].strip()

        # 2. Registry index lookup
        clean_smid = re.sub(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$", "", smid.lower())
        for key in [smid.lower(), clean_smid, cname.lower(), fid.lower()]:
            if key in registry_index and registry_index[key].get("provider"):
                return registry_index[key]["provider"].strip()

        # 3. Known series prefix matching
        norm_text = f"{smid} {cname}".lower()
        for prefix, prov in KNOWN_MODEL_PROVIDERS.items():
            if norm_text.startswith(prefix) or f" {prefix}" in norm_text or f"/{prefix}" in norm_text:
                return prov

        return ""

    # Discover all benchmark columns
    benchmark_cols_set: Set[str] = set()
    bm_original_names: Dict[str, str] = {}

    def register_bm(bm_id: str, orig_name: str):
        col_name = normalize_benchmark_column_name(bm_id or orig_name)
        benchmark_cols_set.add(col_name)
        if col_name not in bm_original_names:
            bm_original_names[col_name] = orig_name or bm_id

    for cap in CAPABILITY_NAMES:
        cap_doc = llmstats_general if cap == "general" else capability_files.get(cap, {})
        for r in cap_doc.get("rows", []):
            bm_obs = r.get("benchmark_observations", {})
            if isinstance(bm_obs, dict):
                for bm_id, bm_data in bm_obs.items():
                    orig_name = bm_id
                    if isinstance(bm_data, dict):
                        orig_name = bm_data.get("canonical_name") or bm_data.get("source_name") or bm_id
                    register_bm(bm_id, orig_name)

    sorted_bm_cols = sorted(list(benchmark_cols_set))

    rows: List[Dict[str, str]] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    gen_row_count = 0
    gen_missing_score_count = 0
    gen_missing_rank_count = 0
    gen_missing_provider_count = 0

    provider_populated_count = 0
    provider_missing_count = 0
    unknown_avail_count = 0

    name_detected_count = 0
    name_corrected_count = 0
    name_unresolved_count = 0

    for cap in CAPABILITY_NAMES:
        cap_doc = llmstats_general if cap == "general" else capability_files.get(cap, {})
        gen_snap = (cap_doc.get("generated_at") or "2026-07-24")[:10]

        for r in cap_doc.get("rows", []):
            raw_source_name = (r.get("source_name") or r.get("model_name") or "").strip()
            source_model_id = (r.get("source_model_id") or "").strip()
            source_model_url = (r.get("source_model_url") or "").strip()

            known_rank = r.get("category_rank") or r.get("general_rank") or r.get("source_rank")
            cleaned_sname, orig_sname, is_corrected = clean_llmstats_display_name(
                raw_source_name, source_model_id, source_model_url, known_rank
            )

            if raw_source_name != cleaned_sname:
                name_detected_count += 1
                if is_corrected:
                    name_corrected_count += 1
                else:
                    name_unresolved_count += 1

            model_ident = source_model_id or cleaned_sname
            if not model_ident:
                continue

            snap_date = (r.get("source_updated_at") or r.get("scraped_at") or gen_snap)[:10].strip()
            nat_key = (snap_date, cap, model_ident)
            if nat_key in seen_keys:
                continue
            seen_keys.add(nat_key)

            # Match family ID via registry if missing or unknown
            fid = (r.get("family_id") or r.get("matched_aa_family_id") or "").strip()
            clean_smid = re.sub(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$", "", source_model_id.lower())
            reg_match = registry_index.get(source_model_id.lower()) or registry_index.get(clean_smid) or registry_index.get(cleaned_sname.lower())

            if (not fid or fid.startswith("unknown/")) and reg_match and reg_match.get("family_id"):
                fid = reg_match.get("family_id").strip()

            is_matched = bool(fid and not fid.startswith("unknown/"))
            if not fid or fid.startswith("unknown/"):
                clean_s = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned_sname.lower()).strip("-")
                fid = f"unknown/{clean_s}"

            # Provider Resolution
            provider = (r.get("provider") or "").strip()
            if not provider:
                provider = resolve_provider_from_approved_sources(source_model_id, cleaned_sname, fid, source_model_url)

            if provider:
                provider_populated_count += 1
            else:
                provider_missing_count += 1

            # Availability Resolution
            fam_match = family_map.get(fid, {})
            avail_class = (r.get("availability_class") or fam_match.get("availability_class") or (reg_match.get("availability_class") if reg_match else "") or "").strip()
            if not avail_class and fam_match.get("weights_available") is not None:
                avail_class = "open_weights" if fam_match.get("weights_available") else "proprietary"

            is_ow = derive_is_open_weights(avail_class)
            if not avail_class or avail_class == "unknown":
                unknown_avail_count += 1

            # Category score & rank extraction
            cat_score = clean_num(r.get("category_score") or r.get("general_score") or r.get("score")) or ""
            cat_rank = clean_num(r.get("category_rank") or r.get("general_rank") or r.get("source_rank") or r.get("rank")) or ""

            if cap == "general":
                gen_row_count += 1
                if not cat_score:
                    gen_missing_score_count += 1
                if not cat_rank:
                    gen_missing_rank_count += 1
                if not provider:
                    gen_missing_provider_count += 1

            row: Dict[str, str] = {
                "schema_version": SCHEMA_VERSION,
                "snapshot_date": snap_date,
                "source": "LLMStats",
                "capability": cap,
                "source_model_id": source_model_id,
                "source_name": cleaned_sname,
                "original_source_name": orig_sname if orig_sname != cleaned_sname else "",
                "canonical_name": (r.get("canonical_name") or cleaned_sname).strip(),
                "family_id": fid,
                "provider": provider,
                "availability_class": avail_class or "unknown",
                "is_open_weights": is_ow,
                "category_rank": cat_rank,
                "category_score": cat_score,
                "match_status": (r.get("match_status") or ("matched" if is_matched else "unmatched")).strip(),
                "source_model_url": source_model_url,
            }

            for bm_col in sorted_bm_cols:
                row[bm_col] = ""

            bm_obs = r.get("benchmark_observations", {})
            if isinstance(bm_obs, dict):
                for bm_id, bm_data in bm_obs.items():
                    orig_name = bm_id
                    val_num = None
                    if isinstance(bm_data, dict):
                        orig_name = bm_data.get("canonical_name") or bm_data.get("source_name") or bm_id
                        val_num = bm_data.get("value")
                    else:
                        val_num = bm_data

                    col_name = normalize_benchmark_column_name(bm_id or orig_name)
                    if col_name in row:
                        row[col_name] = clean_num(val_num) or ""

            rows.append(row)

    rows.sort(key=lambda x: (x["snapshot_date"], x["capability"], x["source_model_id"] or x["source_name"]))

    stats = {
        "llmstats_general_row_count": gen_row_count,
        "llmstats_general_missing_score_count": gen_missing_score_count,
        "llmstats_general_missing_rank_count": gen_missing_rank_count,
        "llmstats_general_missing_provider_count": gen_missing_provider_count,
        "llmstats_provider_populated_count": provider_populated_count,
        "llmstats_provider_missing_count": provider_missing_count,
        "llmstats_availability_unknown_count": unknown_avail_count,
        "malformed_name_count_detected": name_detected_count,
        "malformed_name_count_corrected": name_corrected_count,
        "unresolved_malformed_name_count": name_unresolved_count,
    }

    return rows, sorted_bm_cols, bm_original_names, stats


# -------------------------------------------------------------------------
# Export 3: Model Family History Wide CSV
# -------------------------------------------------------------------------

HISTORICAL_PROPRIETARY_FAMILIES = {
    "openai/gpt-3-5-turbo",
    "openai/gpt-4",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-2",
    "anthropic/claude-2-1",
    "anthropic/claude-instant",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-opus",
    "google/palm-2",
}


def load_git_historical_snapshots() -> List[Dict[str, str]]:
    """Defensively fetch committed historical family JSON revisions via git show without mutating working tree."""
    snapshots: List[Dict[str, str]] = []
    try:
        res = subprocess.run(
            ["git", "log", "--format=%H %cd", "--date=short", "--", "data/index/latest.csv", "data/families/latest.json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return snapshots

        seen_dates: Set[str] = set()
        for line in res.stdout.strip().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            commit_sha, commit_date = parts[0], parts[1].strip()
            if commit_date in seen_dates:
                continue
            seen_dates.add(commit_date)

            show_res = subprocess.run(
                ["git", "show", f"{commit_sha}:data/families/latest.json"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if show_res.returncode == 0 and show_res.stdout.strip():
                try:
                    data = json.loads(show_res.stdout)
                    rows = data.get("rows", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    for r in rows:
                        if isinstance(r, dict) and r.get("family_id"):
                            snapshots.append({
                                "snapshot_date": commit_date,
                                "family_id": r.get("family_id"),
                                "canonical_family_name": r.get("canonical_family_name") or r.get("family_id"),
                                "provider": r.get("provider") or "",
                                "aa_representative_variant_id": r.get("aa_representative_variant_id") or "",
                                "aa_representative_name": r.get("aa_representative_name") or "",
                                "availability_class": r.get("availability_class") or "",
                                "aa_intelligence": clean_num(r.get("aa_intelligence")),
                                "aa_rank": clean_num(r.get("aa_rank")),
                                "llmstats_general_score": clean_num(r.get("llmstats_general_score")),
                                "llmstats_general_rank": clean_num(r.get("llmstats_general_rank")),
                                "llmdex_score": clean_num(r.get("llmdex_score")),
                                "llmdex_rank": clean_num(r.get("llmdex_rank")),
                                "agreement": clean_num(r.get("agreement")),
                                "input_cost": clean_num(r.get("input_cost")),
                                "output_cost": clean_num(r.get("output_cost")),
                                "blended_cost": clean_num(r.get("blended_cost")),
                                "tokens_per_second": clean_num(r.get("tokens_per_second")),
                                "latency": clean_num(r.get("latency")),
                                "context_window": clean_num(r.get("context_window")),
                                "score_status": r.get("score_status") or "consensus_scored",
                                "score_version": r.get("score_version") or "LLMDEX General Consensus v1",
                            })
                except Exception:
                    pass
    except Exception:
        pass
    return snapshots


def build_model_family_history_wide_csv(
    current_families: List[Dict[str, Any]],
    family_snapshots_csv: List[Dict[str, str]],
    score_history_csv: List[Dict[str, str]],
    historical_models_json: List[Dict[str, Any]],
    include_git_history: bool = False,
) -> Tuple[List[Dict[str, str]], str, str, str, int, int]:
    current_family_ids = {r["family_id"] for r in current_families if r.get("family_id")}
    current_family_map = {r["family_id"]: r for r in current_families if r.get("family_id")}

    family_dates: Dict[str, List[str]] = defaultdict(list)
    family_records: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    def record_date(fid: str, dt: str):
        if fid and dt:
            family_dates[fid].append(dt[:10])

    cur_snap = (current_families[0].get("generated_at") or "2026-07-24")[:10] if current_families else "2026-07-24"
    for r in current_families:
        fid = r.get("family_id")
        if not fid:
            continue
        record_date(fid, cur_snap)

        avail = (r.get("availability_class") or "").strip()
        is_ow = derive_is_open_weights(avail)

        input_cost = clean_num(r.get("input_cost"))
        output_cost = clean_num(r.get("output_cost"))
        blended = clean_num(r.get("blended_cost"))
        if not blended and input_cost and output_cost:
            blended = clean_num(0.60 * float(input_cost) + 0.40 * float(output_cost))

        key = (cur_snap, fid, "observed_snapshot")
        family_records[key] = {
            "schema_version": SCHEMA_VERSION,
            "observation_date": cur_snap,
            "snapshot_date": cur_snap,
            "release_date": (r.get("release_date") or "").strip(),
            "family_id": fid,
            "record_type": "observed_snapshot",
            "canonical_family_name": (r.get("canonical_family_name") or fid).strip(),
            "provider": (r.get("provider") or "").strip(),
            "representative_variant_id": (r.get("aa_representative_variant_id") or "").strip(),
            "representative_model_name": (r.get("aa_representative_name") or "").strip(),
            "availability_class": avail,
            "is_open_weights": is_ow,
            "is_current": "True",
            "source_presence": "both" if (r.get("aa_intelligence") and r.get("llmstats_general_score")) else ("artificial_analysis" if r.get("aa_intelligence") else "llmstats"),
            "aa_intelligence": clean_num(r.get("aa_intelligence")) or "",
            "aa_rank": clean_num(r.get("aa_rank")) or "",
            "llmstats_general_score": clean_num(r.get("llmstats_general_score")) or "",
            "llmstats_general_rank": clean_num(r.get("llmstats_general_rank")) or "",
            "llmdex_score": clean_num(r.get("llmdex_score")) or "",
            "llmdex_rank": clean_num(r.get("llmdex_rank")) or "",
            "agreement": clean_num(r.get("agreement")) or "",
            "input_cost_usd_per_1m": input_cost or "",
            "output_cost_usd_per_1m": output_cost or "",
            "blended_cost_usd_per_1m": blended or "",
            "tokens_per_second": clean_num(r.get("tokens_per_second")) or "",
            "latency_seconds": clean_num(r.get("latency")) or "",
            "context_window_tokens": clean_num(r.get("context_window")) or "",
            "score_status": (r.get("score_status") or "").strip(),
            "score_version": (r.get("score_version") or "LLMDEX General Consensus v1").strip(),
            "historical_metric_source": "llmdex_consensus",
            "metric_methodology_version": "LLMDEX General Consensus v1",
            "is_comparable_to_current": "True",
            "evidence_source": "current_contract",
        }

    for csv_rows, source_label in [(family_snapshots_csv, "family_snapshot"), (score_history_csv, "score_history")]:
        for r in csv_rows:
            sdate = (r.get("snapshot_date") or "").strip()
            fid = (r.get("family_id") or "").strip()
            if not sdate or not fid:
                continue
            record_date(fid, sdate)

            key = (sdate, fid, "observed_snapshot")
            if key not in family_records:
                ref_cur = current_family_map.get(fid, {})
                avail = (ref_cur.get("availability_class") or "").strip()
                is_ow = derive_is_open_weights(avail)

                input_cost = clean_num(ref_cur.get("input_cost"))
                output_cost = clean_num(ref_cur.get("output_cost"))
                blended = clean_num(ref_cur.get("blended_cost"))
                if not blended and input_cost and output_cost:
                    blended = clean_num(0.60 * float(input_cost) + 0.40 * float(output_cost))

                family_records[key] = {
                    "schema_version": SCHEMA_VERSION,
                    "observation_date": sdate,
                    "snapshot_date": sdate,
                    "release_date": "",
                    "family_id": fid,
                    "record_type": "observed_snapshot",
                    "canonical_family_name": (r.get("canonical_family_name") or ref_cur.get("canonical_family_name") or fid).strip(),
                    "provider": (ref_cur.get("provider") or "").strip(),
                    "representative_variant_id": (r.get("aa_representative_variant_id") or ref_cur.get("aa_representative_variant_id") or "").strip(),
                    "representative_model_name": (ref_cur.get("aa_representative_name") or "").strip(),
                    "availability_class": avail,
                    "is_open_weights": is_ow,
                    "is_current": "True" if fid in current_family_ids else "False",
                    "source_presence": "both" if (r.get("aa_intelligence") and r.get("llmstats_general_score")) else ("artificial_analysis" if r.get("aa_intelligence") else "llmstats"),
                    "aa_intelligence": clean_num(r.get("aa_intelligence")) or "",
                    "aa_rank": clean_num(r.get("aa_rank")) or "",
                    "llmstats_general_score": clean_num(r.get("llmstats_general_score")) or "",
                    "llmstats_general_rank": clean_num(r.get("llmstats_general_rank")) or "",
                    "llmdex_score": clean_num(r.get("llmdex_score")) or "",
                    "llmdex_rank": clean_num(r.get("llmdex_rank")) or "",
                    "agreement": clean_num(r.get("agreement")) or "",
                    "input_cost_usd_per_1m": input_cost or "",
                    "output_cost_usd_per_1m": output_cost or "",
                    "blended_cost_usd_per_1m": blended or "",
                    "tokens_per_second": clean_num(ref_cur.get("tokens_per_second")) or "",
                    "latency_seconds": clean_num(ref_cur.get("latency")) or "",
                    "context_window_tokens": clean_num(ref_cur.get("context_window")) or "",
                    "score_status": (ref_cur.get("score_status") or "consensus_scored").strip(),
                    "score_version": (ref_cur.get("score_version") or "LLMDEX General Consensus v1").strip(),
                    "historical_metric_source": "llmdex_consensus",
                    "metric_methodology_version": "LLMDEX General Consensus v1",
                    "is_comparable_to_current": "True",
                    "evidence_source": source_label,
                }

    for hm in historical_models_json:
        rdate = (hm.get("release_date") or "")[:10].strip()
        mname = (hm.get("canonical_name") or hm.get("model_name") or "").strip()
        provider = (hm.get("provider") or "").strip()
        if not rdate or not mname:
            continue

        slug_name = re.sub(r"[^a-zA-Z0-9]+", "-", mname.lower()).strip("-")
        prov_slug = re.sub(r"[^a-zA-Z0-9]+", "-", provider.lower()).strip("-") or "historical"
        fid = f"{prov_slug}/{slug_name}"
        record_date(fid, rdate)

        if fid in HISTORICAL_PROPRIETARY_FAMILIES or "openai" in prov_slug or "anthropic" in prov_slug or "google" in prov_slug:
            avail = "proprietary"
            is_ow = "False"
        elif hm.get("historical") is True:
            avail = "open_weights"
            is_ow = "True"
        elif fid in current_family_map:
            avail = current_family_map[fid].get("availability_class") or "unknown"
            is_ow = derive_is_open_weights(avail)
        else:
            avail = "unknown"
            is_ow = ""

        rec_type = "catalog_event"
        key = (rdate, fid, rec_type)

        if key not in family_records:
            family_records[key] = {
                "schema_version": SCHEMA_VERSION,
                "observation_date": rdate,
                "snapshot_date": rdate,
                "release_date": rdate,
                "family_id": fid,
                "record_type": rec_type,
                "canonical_family_name": mname,
                "provider": provider,
                "representative_variant_id": f"{fid}:default",
                "representative_model_name": mname,
                "availability_class": avail,
                "is_open_weights": is_ow,
                "is_current": "True" if fid in current_family_ids else "False",
                "source_presence": "historical_only",
                "aa_intelligence": clean_num(hm.get("adjusted_performance")) or "",
                "aa_rank": "",
                "llmstats_general_score": "",
                "llmstats_general_rank": "",
                "llmdex_score": "",
                "llmdex_rank": "",
                "agreement": "",
                "input_cost_usd_per_1m": "",
                "output_cost_usd_per_1m": "",
                "blended_cost_usd_per_1m": "",
                "tokens_per_second": "",
                "latency_seconds": "",
                "context_window_tokens": "",
                "score_status": "historical_archived",
                "score_version": "Historical Catalog v1",
                "historical_metric_source": "historical_catalog",
                "metric_methodology_version": "Historical Catalog v1",
                "is_comparable_to_current": "False",
                "evidence_source": "historical_catalog",
            }

    if include_git_history:
        git_snaps = load_git_historical_snapshots()
        for r in git_snaps:
            sdate = r.get("snapshot_date")
            fid = r.get("family_id")
            if not sdate or not fid:
                continue
            record_date(fid, sdate)
            key = (sdate, fid, "observed_snapshot")
            if key not in family_records:
                avail = (r.get("availability_class") or "").strip()
                family_records[key] = {
                    "schema_version": SCHEMA_VERSION,
                    "observation_date": sdate,
                    "snapshot_date": sdate,
                    "release_date": "",
                    "family_id": fid,
                    "record_type": "observed_snapshot",
                    "canonical_family_name": (r.get("canonical_family_name") or fid).strip(),
                    "provider": (r.get("provider") or "").strip(),
                    "representative_variant_id": (r.get("aa_representative_variant_id") or "").strip(),
                    "representative_model_name": (r.get("aa_representative_name") or "").strip(),
                    "availability_class": avail,
                    "is_open_weights": derive_is_open_weights(avail),
                    "is_current": "True" if fid in current_family_ids else "False",
                    "source_presence": "both" if (r.get("aa_intelligence") and r.get("llmstats_general_score")) else ("artificial_analysis" if r.get("aa_intelligence") else "llmstats"),
                    "aa_intelligence": clean_num(r.get("aa_intelligence")) or "",
                    "aa_rank": clean_num(r.get("aa_rank")) or "",
                    "llmstats_general_score": clean_num(r.get("llmstats_general_score")) or "",
                    "llmstats_general_rank": clean_num(r.get("llmstats_general_rank")) or "",
                    "llmdex_score": clean_num(r.get("llmdex_score")) or "",
                    "llmdex_rank": clean_num(r.get("llmdex_rank")) or "",
                    "agreement": clean_num(r.get("agreement")) or "",
                    "input_cost_usd_per_1m": clean_num(r.get("input_cost")) or "",
                    "output_cost_usd_per_1m": clean_num(r.get("output_cost")) or "",
                    "blended_cost_usd_per_1m": clean_num(r.get("blended_cost")) or "",
                    "tokens_per_second": clean_num(r.get("tokens_per_second")) or "",
                    "latency_seconds": clean_num(r.get("latency")) or "",
                    "context_window_tokens": clean_num(r.get("context_window")) or "",
                    "score_status": (r.get("score_status") or "").strip(),
                    "score_version": (r.get("score_version") or "LLMDEX General Consensus v1").strip(),
                    "historical_metric_source": "llmdex_consensus",
                    "metric_methodology_version": "LLMDEX General Consensus v1",
                    "is_comparable_to_current": "True",
                    "evidence_source": "git_index_snapshot",
                }

    family_first_seen = {fid: min(dates) for fid, dates in family_dates.items()}
    family_last_seen = {fid: max(dates) for fid, dates in family_dates.items()}

    final_rows: List[Dict[str, str]] = []
    catalog_count = 0
    observed_count = 0

    for (sdate, fid, rtype), r in family_records.items():
        r_copy = dict(r)
        r_copy["first_seen_date"] = family_first_seen.get(fid, sdate)
        r_copy["last_seen_date"] = family_last_seen.get(fid, sdate)
        final_rows.append(r_copy)

        if rtype in {"catalog_event", "release_event"}:
            catalog_count += 1
        else:
            observed_count += 1

    final_rows.sort(key=lambda x: (x["observation_date"], x["family_id"], x["record_type"]))

    catalog_dates = [r["observation_date"] for r in final_rows if r["record_type"] in {"catalog_event", "release_event"}]
    observed_dates = [r["observation_date"] for r in final_rows if r["record_type"] == "observed_snapshot"]

    earliest_cat = min(catalog_dates) if catalog_dates else "2022-11-30"
    earliest_obs = min(observed_dates) if observed_dates else "2026-02-24"
    latest_obs = max(observed_dates) if observed_dates else "2026-07-24"

    return final_rows, earliest_cat, earliest_obs, latest_obs, catalog_count, observed_count


# -------------------------------------------------------------------------
# Export 4: Combined LLMDEX Dataset JSON
# -------------------------------------------------------------------------

def build_combined_latest_json(
    aa_wide_rows: List[Dict[str, str]],
    llmstats_wide_rows: List[Dict[str, str]],
    families_contract: Dict[str, Any],
    quality_contract: Dict[str, Any],
    methodology_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    aa_snap = (aa_wide_rows[0].get("snapshot_date") if aa_wide_rows else "2026-07-24") or "2026-07-24"
    llmstats_snap = (llmstats_wide_rows[0].get("snapshot_date") if llmstats_wide_rows else "2026-07-24") or "2026-07-24"

    # Coerce rows to native JSON types
    aa_typed = [coerce_row_dict_to_json(r) for r in aa_wide_rows]
    llmstats_typed = [coerce_row_dict_to_json(r) for r in llmstats_wide_rows]

    llmstats_caps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    llmstats_gen: List[Dict[str, Any]] = []
    for r in llmstats_typed:
        if r.get("capability") == "general":
            llmstats_gen.append(r)
        else:
            llmstats_caps[str(r.get("capability"))].append(r)

    # Build dedicated llmdex.consensus array
    consensus_rows: List[Dict[str, Any]] = []
    all_families = families_contract.get("rows", [])
    for f in all_families:
        if isinstance(f, dict) and (f.get("llmdex_score") is not None or f.get("score_status") == "consensus"):
            consensus_rows.append(coerce_row_dict_to_json({
                "family_id": f.get("family_id"),
                "canonical_family_name": f.get("canonical_family_name"),
                "provider": f.get("provider"),
                "representative_variant_id": f.get("aa_representative_variant_id"),
                "representative_model_name": f.get("aa_representative_name"),
                "availability_class": f.get("availability_class"),
                "is_open_weights": derive_is_open_weights(f.get("availability_class")),
                "aa_intelligence": f.get("aa_intelligence"),
                "aa_rank": f.get("aa_rank"),
                "llmstats_general_score": f.get("llmstats_general_score"),
                "llmstats_general_rank": f.get("llmstats_general_rank"),
                "llmdex_score": f.get("llmdex_score"),
                "llmdex_rank": f.get("llmdex_rank"),
                "agreement": f.get("agreement"),
                "source_coverage": f.get("perf_source_count") or f.get("source_count"),
                "score_status": f.get("score_status") or "consensus",
                "score_version": f.get("score_version") or "LLMDEX General Consensus v1",
                "is_sota": f.get("is_sota"),
                "is_open_sota": f.get("is_open_sota"),
            }))

    combined = {
        "schema_version": SCHEMA_VERSION,
        "generated_from_snapshot": aa_snap,
        "source_snapshots": {
            "artificial_analysis": aa_snap,
            "llmstats": llmstats_snap,
        },
        "artificial_analysis": {
            "models": aa_typed,
        },
        "llmstats": {
            "general": llmstats_gen,
            "capabilities": dict(llmstats_caps),
        },
        "llmdex": {
            "families": [coerce_row_dict_to_json(f) for f in all_families],
            "consensus": consensus_rows,
            "quality": coerce_row_dict_to_json(quality_contract),
            "methodology": coerce_row_dict_to_json(methodology_manifest),
        },
    }

    return combined


# -------------------------------------------------------------------------
# Export 5: Data Dictionary CSV
# -------------------------------------------------------------------------

def build_data_dictionary_csv(
    aa_wide_rows: List[Dict[str, str]],
    llmstats_wide_rows: List[Dict[str, str]],
    history_wide_rows: List[Dict[str, str]],
    bm_cols: List[str],
    bm_orig_names: Dict[str, str],
) -> List[Dict[str, str]]:
    dict_rows: List[Dict[str, str]] = []

    aa_pop: Dict[str, int] = defaultdict(int)
    for r in aa_wide_rows:
        for k, v in r.items():
            if v != "":
                aa_pop[k] += 1

    llm_pop: Dict[str, int] = defaultdict(int)
    for r in llmstats_wide_rows:
        for k, v in r.items():
            if v != "":
                llm_pop[k] += 1

    hist_pop: Dict[str, int] = defaultdict(int)
    for r in history_wide_rows:
        for k, v in r.items():
            if v != "":
                hist_pop[k] += 1

    def add_meta(tbl, col, orig, dtype, desc, unit, h_better, pop, pop_def, agg, src, origin, nullable):
        is_populated = "True" if pop > 0 else "False"
        dict_rows.append({
            "table_name": tbl,
            "column_name": col,
            "original_metric_name": orig,
            "data_type": dtype,
            "description": desc,
            "unit": unit,
            "higher_is_better": bool_str(h_better),
            "source_population": str(pop),
            "population_definition": pop_def,
            "recommended_aggregation": agg,
            "source": src,
            "metric_origin": origin,
            "nullable": bool_str(nullable),
            "currently_populated": is_populated,
        })

    for col in AA_WIDE_COLUMNS:
        pop_count = aa_pop[col]
        pop_def = f"Non-null model variant observations in export snapshot ({pop_count} records)"
        if col.startswith("aa_"):
            origin = "Artificial Analysis"
            src = "Artificial Analysis"
        elif col.startswith("llmdex_"):
            origin = "LLMDEX derived"
            src = "LLMDEX"
        else:
            origin = "LLMDEX identity"
            src = "Artificial Analysis / LLMDEX"

        dtype = "Decimal number" if col.startswith(("aa_", "llmdex_")) and "rank" not in col else ("Whole number" if "rank" in col or "tokens" in col or "count" in col else "Text")
        agg = "Average" if "Decimal" in dtype else ("Minimum" if "rank" in col else "Do not summarize")
        h_better = "False" if "rank" in col or "cost" in col or "latency" in col or "response_time" in col else "True"

        desc = f"Artificial Analysis export field {col}"
        if col in EMPTY_RESERVED_AA_COLS:
            desc = f"Reserved column for Artificial Analysis metric '{col}' (currently unpopulated in snapshot)."

        add_meta(
            tbl="artificial_analysis_benchmarks",
            col=col,
            orig=col.removeprefix("aa_").removeprefix("llmdex_").replace("_", " ").title(),
            dtype=dtype,
            desc=desc,
            unit="index" if "score" in col or "index" in col else ("USD / 1M tokens" if "cost" in col else ("tokens/sec" if "speed" in col or "tokens_per_second" in col else ("seconds" if "latency" in col or "time" in col else ""))),
            h_better=h_better,
            pop=pop_count,
            pop_def=pop_def,
            agg=agg,
            src=src,
            origin=origin,
            nullable=(col not in {"schema_version", "snapshot_date", "source", "model_key"}),
        )

    llm_base_cols = [
        "schema_version", "snapshot_date", "source", "capability", "source_model_id",
        "source_name", "original_source_name", "canonical_name", "family_id", "provider",
        "availability_class", "is_open_weights", "category_rank", "category_score",
        "match_status", "source_model_url"
    ]

    for col in llm_base_cols:
        pop_count = llm_pop[col]
        pop_def = f"Non-null model capability observations in export snapshot ({pop_count} records)"
        dtype = "Decimal number" if col == "category_score" else ("Whole number" if col == "category_rank" else "Text")
        agg = "Average" if col == "category_score" else ("Minimum" if col == "category_rank" else "Do not summarize")

        add_meta(
            tbl="llmstats_benchmarks",
            col=col,
            orig=col.replace("_", " ").title(),
            dtype=dtype,
            desc=f"LLMStats base field {col}",
            unit="index" if col == "category_score" else ("rank" if col == "category_rank" else ""),
            h_better="False" if col == "category_rank" else "True",
            pop=pop_count,
            pop_def=pop_def,
            agg=agg,
            src="LLMStats",
            origin="LLMStats",
            nullable=(col not in {"schema_version", "snapshot_date", "source", "capability"}),
        )

    for bm_col in bm_cols:
        pop_count = llm_pop[bm_col]
        pop_def = f"Non-null model observations published for this benchmark in current capability contracts ({pop_count} records)"
        orig_name = bm_orig_names.get(bm_col, bm_col.removeprefix("benchmark_").replace("_", " ").title())

        add_meta(
            tbl="llmstats_benchmarks",
            col=bm_col,
            orig=orig_name,
            dtype="Decimal number",
            desc=f"Source-native LLMStats benchmark observation: {orig_name}",
            unit="score",
            h_better="True",
            pop=pop_count,
            pop_def=pop_def,
            agg="Average",
            src="LLMStats",
            origin="LLMStats",
            nullable=True,
        )

    hist_cols = [
        "schema_version", "observation_date", "snapshot_date", "release_date",
        "first_seen_date", "last_seen_date", "record_type", "family_id",
        "canonical_family_name", "provider", "representative_variant_id",
        "representative_model_name", "availability_class", "is_open_weights",
        "is_current", "source_presence", "aa_intelligence", "aa_rank",
        "llmstats_general_score", "llmstats_general_rank", "llmdex_score",
        "llmdex_rank", "agreement", "input_cost_usd_per_1m", "output_cost_usd_per_1m",
        "blended_cost_usd_per_1m", "tokens_per_second", "latency_seconds",
        "context_window_tokens", "score_status", "score_version",
        "historical_metric_source", "metric_methodology_version",
        "is_comparable_to_current", "evidence_source"
    ]

    for col in hist_cols:
        pop_count = hist_pop[col]
        pop_def = f"Non-null family observation rows with {col} present ({pop_count} records)"
        origin = "historical catalogue" if col in {"historical_metric_source", "metric_methodology_version", "is_comparable_to_current"} else ("LLMDEX derived" if col.startswith("llmdex_") else "Artificial Analysis / LLMStats")

        dtype = "Decimal number" if ("score" in col or "cost" in col or "intelligence" in col or "speed" in col or "latency" in col) and "rank" not in col else ("Whole number" if "rank" in col or "tokens" in col else ("True/False" if col in BOOL_FIELDS else "Text"))
        agg = "Average" if "Decimal" in dtype else ("Minimum" if "rank" in col else "Do not summarize")

        add_meta(
            tbl="model_family_history",
            col=col,
            orig=col.replace("_", " ").title(),
            dtype=dtype,
            desc=f"Model family history progression field {col}",
            unit="index" if "score" in col or "intelligence" in col else ("USD / 1M tokens" if "cost" in col else ""),
            h_better="False" if "rank" in col or "cost" in col or "latency" in col else "True",
            pop=pop_count,
            pop_def=pop_def,
            agg=agg,
            src="LLMDEX History",
            origin=origin,
            nullable=(col not in {"schema_version", "observation_date", "family_id", "record_type"}),
        )

    dict_rows.sort(key=lambda x: (x["table_name"], x["column_name"]))
    return dict_rows


# -------------------------------------------------------------------------
# Main Exporter Runner
# -------------------------------------------------------------------------

def run_exports(include_git_history: bool = False) -> Dict[str, Any]:
    print("=== Launching LLMDEX Power BI Exporter (v2 Wide Schema) ===")

    aa_records = load_json_file(ROOT / "data" / "cleaned" / "artificial_analysis" / "latest.json")
    llmstats_general = load_json_file(ROOT / "data" / "cleaned" / "llmstats" / "latest.json")

    capability_files: Dict[str, Dict[str, Any]] = {}
    for cap in CAPABILITY_NAMES:
        if cap != "general":
            cap_path = ROOT / "data" / "capabilities" / f"{cap}.json"
            capability_files[cap] = load_json_file(cap_path)

    families_contract = load_json_file(ROOT / "data" / "families" / "latest.json")
    quality_contract = load_json_file(ROOT / "data" / "quality" / "latest.json")
    methodology_manifest = load_json_file(ROOT / "data" / "methodology" / "publication_manifest.json")
    identity_registry = load_json_file(ROOT / "data" / "identity" / "model_registry.json")

    family_snapshots_csv = load_csv_file(ROOT / "data" / "history" / "family_snapshots.csv")
    score_history_csv = load_csv_file(ROOT / "data" / "history" / "score_history.csv")
    historical_models_json = load_json_file(ROOT / "data" / "history" / "historical_models.json")

    print("Generating Wide Artificial Analysis Benchmarks CSV...")
    aa_wide_rows, corrected_ow_count = build_aa_wide_csv(aa_records)

    print("Generating Wide LLMStats Benchmarks CSV...")
    llmstats_wide_rows, bm_cols, bm_orig_names, llm_stats = build_llmstats_wide_csv(
        llmstats_general,
        capability_files,
        families_contract,
        identity_registry,
    )

    print("Generating Wide Model Family History CSV...")
    history_wide_rows, earliest_cat, earliest_obs, latest_obs, cat_count, obs_count = build_model_family_history_wide_csv(
        families_contract.get("rows", []),
        family_snapshots_csv,
        score_history_csv,
        historical_models_json,
        include_git_history=include_git_history,
    )

    print("Generating Combined Latest JSON...")
    combined_json_doc = build_combined_latest_json(
        aa_wide_rows,
        llmstats_wide_rows,
        families_contract,
        quality_contract,
        methodology_manifest,
    )

    print("Generating Data Dictionary CSV...")
    data_dict_rows = build_data_dictionary_csv(
        aa_wide_rows,
        llmstats_wide_rows,
        history_wide_rows,
        bm_cols,
        bm_orig_names,
    )

    consensus_fam_count = len(combined_json_doc.get("llmdex", {}).get("consensus", []))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        aa_csv_path = tmp_path / "artificial_analysis_benchmarks.csv"
        with open(aa_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            if aa_wide_rows:
                writer = csv.DictWriter(
                    f, fieldnames=AA_WIDE_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(aa_wide_rows)

        llmstats_csv_path = tmp_path / "llmstats_benchmarks.csv"
        llm_fieldnames = [
            "schema_version", "snapshot_date", "source", "capability", "source_model_id",
            "source_name", "original_source_name", "canonical_name", "family_id", "provider",
            "availability_class", "is_open_weights", "category_rank", "category_score",
            "match_status", "source_model_url"
        ] + bm_cols

        with open(llmstats_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            if llmstats_wide_rows:
                writer = csv.DictWriter(
                    f, fieldnames=llm_fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(llmstats_wide_rows)

        combined_json_path = tmp_path / "combined_latest.json"
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_json_doc, f, indent=2, allow_nan=False, ensure_ascii=False)

        history_csv_path = tmp_path / "model_family_history.csv"
        hist_fieldnames = [
            "schema_version", "observation_date", "snapshot_date", "release_date",
            "first_seen_date", "last_seen_date", "record_type", "family_id",
            "canonical_family_name", "provider", "representative_variant_id",
            "representative_model_name", "availability_class", "is_open_weights",
            "is_current", "source_presence", "aa_intelligence", "aa_rank",
            "llmstats_general_score", "llmstats_general_rank", "llmdex_score",
            "llmdex_rank", "agreement", "input_cost_usd_per_1m", "output_cost_usd_per_1m",
            "blended_cost_usd_per_1m", "tokens_per_second", "latency_seconds",
            "context_window_tokens", "score_status", "score_version",
            "historical_metric_source", "metric_methodology_version",
            "is_comparable_to_current", "evidence_source"
        ]

        with open(history_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            if history_wide_rows:
                writer = csv.DictWriter(
                    f, fieldnames=hist_fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(history_wide_rows)

        dict_csv_path = tmp_path / "data_dictionary.csv"
        dict_fieldnames = [
            "table_name", "column_name", "original_metric_name", "data_type",
            "description", "unit", "higher_is_better", "source_population",
            "population_definition", "recommended_aggregation", "source",
            "metric_origin", "nullable", "currently_populated"
        ]

        with open(dict_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            if data_dict_rows:
                writer = csv.DictWriter(
                    f, fieldnames=dict_fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(data_dict_rows)

        hashes: Dict[str, str] = {}
        for fn in [
            "artificial_analysis_benchmarks.csv",
            "llmstats_benchmarks.csv",
            "combined_latest.json",
            "model_family_history.csv",
            "data_dictionary.csv",
        ]:
            h = hashlib.sha256()
            with open(tmp_path / fn, "rb") as f:
                h.update(f.read())
            hashes[fn] = h.hexdigest()

        aa_snap = (aa_records[0].get("snapshot_date") if aa_records else "2026-07-24") or "2026-07-24"
        llmstats_snap = (llmstats_general.get("generated_at") or "2026-07-24")[:10]

        manifest_doc = {
            "schema_version": SCHEMA_VERSION,
            "source_snapshot_dates": {
                "artificial_analysis": aa_snap,
                "llmstats": llmstats_snap,
            },
            "earliest_catalog_event_date": earliest_cat,
            "earliest_observed_snapshot_date": earliest_obs,
            "latest_observed_snapshot_date": latest_obs,
            "catalog_event_row_count": cat_count,
            "observed_snapshot_row_count": obs_count,
            "historical_coverage_limitations": "Pre-LLMDEX historical rows represent repository-supported catalogue or release events and must not be interpreted as daily historical benchmark measurements.",
            "row_counts": {
                "artificial_analysis_benchmarks": len(aa_wide_rows),
                "llmstats_benchmarks": len(llmstats_wide_rows),
                "combined_latest": 1,
                "model_family_history": len(history_wide_rows),
                "data_dictionary": len(data_dict_rows),
            },
            "column_counts": {
                "artificial_analysis_benchmarks": len(AA_WIDE_COLUMNS),
                "llmstats_benchmarks": len(llm_fieldnames),
                "model_family_history": len(hist_fieldnames),
            },
            "number_of_benchmark_columns": len(bm_cols),
            "llmstats_general_row_count": llm_stats["llmstats_general_row_count"],
            "llmstats_general_missing_score_count": llm_stats["llmstats_general_missing_score_count"],
            "llmstats_general_missing_rank_count": llm_stats["llmstats_general_missing_rank_count"],
            "llmstats_general_missing_provider_count": llm_stats["llmstats_general_missing_provider_count"],
            "llmstats_provider_populated_count": llm_stats["llmstats_provider_populated_count"],
            "llmstats_provider_missing_count": llm_stats["llmstats_provider_missing_count"],
            "llmstats_availability_unknown_count": llm_stats["llmstats_availability_unknown_count"],
            "malformed_name_count_detected": llm_stats["malformed_name_count_detected"],
            "malformed_name_count_corrected": llm_stats["malformed_name_count_corrected"],
            "unresolved_malformed_name_count": llm_stats["unresolved_malformed_name_count"],
            "consensus_family_count": consensus_fam_count,
            "empty_reserved_columns": EMPTY_RESERVED_AA_COLS,
            "corrected_open_weight_inconsistency_count": corrected_ow_count,
            "sha256_hashes": hashes,
            "validation_status": "valid",
            "warnings": [],
            "generated_from_commit": get_git_commit_sha(),
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_doc, f, indent=2, allow_nan=False, ensure_ascii=False)

        POWERBI_DIR.mkdir(parents=True, exist_ok=True)
        for fn in [
            "artificial_analysis_benchmarks.csv",
            "llmstats_benchmarks.csv",
            "combined_latest.json",
            "model_family_history.csv",
            "manifest.json",
            "data_dictionary.csv",
        ]:
            shutil.copy2(tmp_path / fn, POWERBI_DIR / fn)

    print(f"Successfully published wide exports to {POWERBI_DIR}:")
    print(f"  - artificial_analysis_benchmarks.csv ({len(aa_wide_rows)} rows, {len(AA_WIDE_COLUMNS)} cols)")
    print(f"  - llmstats_benchmarks.csv ({len(llmstats_wide_rows)} rows, {len(llm_fieldnames)} cols, {len(bm_cols)} benchmark cols)")
    print(f"  - combined_latest.json ({consensus_fam_count} consensus families)")
    print(f"  - model_family_history.csv ({len(history_wide_rows)} rows, {len(hist_fieldnames)} cols)")
    print(f"  - manifest.json (schema: {SCHEMA_VERSION})")
    print(f"  - data_dictionary.csv ({len(data_dict_rows)} rows)")

    return manifest_doc


def main():
    parser = argparse.ArgumentParser(description="Export LLMDEX Power BI v2 Wide Datasets")
    parser.add_argument("--include-git-history", action="store_true", help="Inspect earlier committed contracts via git show")
    args = parser.parse_args()

    try:
        run_exports(include_git_history=args.include_git_history)
        sys.exit(0)
    except Exception as exc:
        print(f"EXPORT ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
