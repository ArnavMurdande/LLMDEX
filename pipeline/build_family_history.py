"""Build a readable model-family progression dataset from the current index."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_families import (
    infer_family_brand,
    infer_model_family,
    normalize_release_name,
    release_sort_key,
)


def build_family_history(data: List[dict]) -> Dict[str, List[dict]]:
    """Collapse effort variants and compute release-to-release improvements."""
    grouped: Dict[str, Dict[str, List[dict]]] = {}

    for row in data:
        name = row.get("canonical_name") or row.get("model_name")
        if not name:
            continue
        performance = row.get("adjusted_performance")
        if performance is None:
            performance = row.get("performance_index")
        if performance is None:
            continue

        family = infer_model_family(name, row.get("provider"))
        release = normalize_release_name(name)
        grouped.setdefault(family, {}).setdefault(release, []).append(
            {
                "name": name,
                "performance": float(performance),
                "rank": row.get("performance_rank"),
                "provider": row.get("provider"),
            }
        )

    output: Dict[str, List[dict]] = {}
    for family, releases in grouped.items():
        members: List[dict] = []
        for release, variants in releases.items():
            best = max(variants, key=lambda item: item["performance"]).copy()
            best["name"] = release
            best["variants"] = [item["name"] for item in variants]
            best["variant_count"] = len(variants)
            members.append(best)

        members.sort(key=lambda item: release_sort_key(item["name"]))
        for index, member in enumerate(members):
            previous = members[index - 1] if index else None
            difference = (
                member["performance"] - previous["performance"] if previous else 0.0
            )
            member["improvement_abs"] = round(difference, 2)
            member["improvement_pct"] = round(
                (difference / previous["performance"] * 100)
                if previous and previous["performance"]
                else 0.0,
                2,
            )
            member["predecessor"] = previous["name"] if previous else None
        output[family] = members

    # Hide one-off brands, while preserving singleton product lines such as
    # Claude Fable whenever their brand has other releases in the dataset.
    brand_release_counts: Dict[str, int] = {}
    for family, members in output.items():
        brand = infer_family_brand(family)
        brand_release_counts[brand] = brand_release_counts.get(brand, 0) + len(members)

    return {
        family: members
        for family, members in sorted(output.items())
        if brand_release_counts.get(infer_family_brand(family), 0) >= 2
    }


def build_history() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    latest_path = os.path.join(base_dir, "index", "latest.json")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"Index not found: {latest_path}")

    with open(latest_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    history = build_family_history(data)
    history_dir = os.path.join(base_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    output_path = os.path.join(history_dir, "family_growth.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Built {len(history)} model families: {output_path}")


if __name__ == "__main__":
    build_history()
