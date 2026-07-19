"""Recompute rankings from the latest published index without scraping.

This is useful after changing scoring logic: it preserves the current model
set, source data, snapshot date, and row order while refreshing all derived
scores and ranks.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.merge_data import save_dataset_layer
from pipeline.scoring import score_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index" / "latest.json"

RANK_FIELDS = (
    "performance_rank",
    "value_rank",
    "efficiency_rank",
    "global_rank",
)


def _snapshot_date(rows: list[dict]) -> str:
    for row in rows:
        value = row.get("last_updated") or row.get("snapshot_date")
        if value:
            return str(value)[:10]
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_legacy_outputs(index_df: pd.DataFrame) -> None:
    flat = index_df.copy()
    for column in flat.columns:
        if flat[column].apply(lambda value: isinstance(value, (list, dict))).any():
            flat[column] = flat[column].apply(
                lambda value: json.dumps(value, default=str)
                if isinstance(value, (list, dict))
                else value
            )
    flat.to_csv(DATA_DIR / "models.csv", index=False, lineterminator="\n")
    index_df.to_json(
        DATA_DIR / "models.json",
        orient="records",
        indent=2,
        default_handler=str,
    )


def main() -> int:
    if not INDEX_PATH.exists():
        print("ERROR: data/index/latest.json does not exist. Run the pipeline first.")
        return 1

    rows = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print("ERROR: the latest index is empty or invalid.")
        return 1

    snapshot_date = _snapshot_date(rows)
    frame = pd.DataFrame(rows).drop(columns=list(RANK_FIELDS), errors="ignore")
    scored = score_dataset(frame)
    scored["last_updated"] = snapshot_date

    save_dataset_layer(scored, "index", str(DATA_DIR), snapshot_date)
    _write_legacy_outputs(scored)

    ranked = int(scored["performance_rank"].notna().sum())
    print(
        f"Re-scoring complete: {len(scored)} models, {ranked} performance-ranked, "
        f"snapshot {snapshot_date}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
