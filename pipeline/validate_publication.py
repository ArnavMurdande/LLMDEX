"""Fail CI before publishing structurally stale or incomplete LLMDEX data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.sentiment_pipeline import get_models_for_sentiment


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_publication() -> None:
    index = _read("data/index/latest.json")
    history = _read("data/history/family_growth.json")
    sentiment = _read("data/sentiment/latest.json")

    if len(index) < 100:
        raise ValueError(f"Index unexpectedly small: {len(index)} rows")
    ranked = [row for row in index if row.get("performance_rank") is not None]
    if not ranked or min(row["performance_rank"] for row in ranked) != 1:
        raise ValueError("Performance ranking has no rank 1")
    sources = {
        source
        for row in index
        for source in (row.get("sources") or [])
        if source
    }
    if sources != {"Artificial Analysis"}:
        raise ValueError(f"Unexpected benchmark sources: {sorted(sources)}")

    if any(
        "fable" in (row.get("canonical_name") or "").lower() for row in index
    ) and "Claude Fable" not in history:
        raise ValueError("Claude Fable exists in the index but not family history")

    expected_models = get_models_for_sentiment(limit=10)
    actual_models = [row.get("model_name") for row in sentiment]
    if len(actual_models) != 10:
        raise ValueError(f"Sentiment must contain exactly 10 models, got {len(actual_models)}")
    if actual_models != expected_models:
        raise ValueError(
            "Sentiment model list does not match the current top ten: "
            f"expected={expected_models}, actual={actual_models}"
        )
    for row in sentiment:
        if row.get("_experimental") is not True:
            raise ValueError(f"Sentiment row is not labeled experimental: {row}")
        if not isinstance(row.get("source_counts"), dict):
            raise ValueError(f"Sentiment source breakdown missing: {row.get('model_name')}")

    print(
        f"Publication contract passed: {len(index)} models, "
        f"{len(history)} families, {len(sentiment)} sentiment rows"
    )


if __name__ == "__main__":
    validate_publication()
