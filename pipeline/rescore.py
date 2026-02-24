"""
rescore.py — Re-process existing cleaned data through the updated merge + scoring pipeline.

This script reads the existing raw_snapshot/latest.json,
runs it through the updated merge_data.py and scoring.py,
and writes the new index/latest.json.

Usage:
    python rescore.py
"""

import json
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.merge_data import process_and_save

def main():
    # Read existing raw snapshot (pre-merge data)
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_snapshot", "latest.json")
    raw_path = os.path.abspath(raw_path)

    if not os.path.exists(raw_path):
        # Fallback: read the cleaned data and treat each row as a raw row
        cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "latest.json")
        cleaned_path = os.path.abspath(cleaned_path)
        
        if not os.path.exists(cleaned_path):
            print("ERROR: No raw or cleaned data found. Run the pipeline first.")
            sys.exit(1)
        
        with open(cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} rows from cleaned/latest.json")
        
        # Flatten provenance back to raw rows for re-processing
        raw_rows = []
        for row in data:
            provenance = row.get("provenance", [])
            if provenance:
                for prov in provenance:
                    raw_row = {
                        "model_name": prov.get("raw_name", row.get("model_name", "")),
                        "source": prov.get("source", "unknown"),
                        "scraped_at": prov.get("scraped_at"),
                        "confidence": prov.get("confidence", 1.0),
                    }
                    # Add raw scores
                    for k, v in prov.get("raw_scores", {}).items():
                        raw_row[k] = v
                    
                    # Add provider if available
                    if row.get("provider"):
                        raw_row["provider"] = row["provider"]
                    
                    raw_rows.append(raw_row)
            else:
                # No provenance, use the row directly
                raw_rows.append(row)
        
        data = raw_rows
        print(f"Expanded to {len(data)} raw rows from provenance")
    else:
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} rows from raw_snapshot/latest.json")

    # Re-process through updated pipeline
    result = process_and_save(data, snapshot_date="2026-02-18")
    
    if result is not None:
        print(f"\n✓ Re-scoring complete. {len(result)} models in index.")
        print(f"  Ranked: {int(result['composite_index'].notna().sum())}")
        print(f"  Unranked: {int(result['composite_index'].isna().sum())}")
    else:
        print("✗ Re-scoring FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
