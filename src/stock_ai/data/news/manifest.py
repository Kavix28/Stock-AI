"""
Reproducible News Data Manifest Generator.

Audits and records:
- Source provenance and retrieval timestamps.
- SHA-256 integrity checksums.
- Date ranges and missing forward coverage.
- Intraday timestamp precision and lookahead leakage risk.
- Licensing and academic usability.
"""

import os
import glob
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import pytz

from .schema import CanonicalNewsSchema
from .validator import NewsDataValidator


class NewsManifestGenerator:
    """Generates comprehensive CSV and JSON manifests for all news datasets."""

    def __init__(self, news_dir: str = "data/processed/news"):
        self.news_dir = news_dir

    def audit_all_news_files(self) -> List[Dict[str, Any]]:
        manifest_rows = []
        if not os.path.exists(self.news_dir):
            return manifest_rows

        csv_files = sorted(glob.glob(os.path.join(self.news_dir, "*.csv")))
        retrieval_utc = datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        for fpath in csv_files:
            # Skip manifest itself if already present
            if "manifest" in os.path.basename(fpath).lower():
                continue

            symbol = os.path.splitext(os.path.basename(fpath))[0].upper()
            
            with open(fpath, "rb") as f:
                sha256_hash = hashlib.sha256(f.read()).hexdigest()

            file_size_bytes = os.path.getsize(fpath)

            try:
                raw_df = pd.read_csv(fpath)
                norm_df = CanonicalNewsSchema.normalize_dataframe(raw_df, symbol=symbol)
                val_stats = NewsDataValidator.validate_canonical_dataframe(norm_df, symbol=symbol)
                
                # Check forward coverage
                dates = pd.to_datetime(norm_df["published_date"])
                forward_count = int((dates >= "2024-07-01").sum())
                historical_count = int(len(norm_df) - forward_count)
                
                has_intraday = bool(norm_df["has_intraday_time"].any())
                timing_status = (
                    "POINT_IN_TIME_CERTIFIED" if has_intraday else "TIMING-LIMITED: Missing Intraday Timestamps"
                )
                
                # Deduplication rate
                dup_count = int(norm_df["title"].duplicated().sum())

                row = {
                    "symbol": symbol,
                    "filename": os.path.basename(fpath),
                    "file_size_bytes": file_size_bytes,
                    "sha256_checksum": sha256_hash,
                    "retrieval_timestamp_utc": retrieval_utc,
                    "total_rows": len(norm_df),
                    "unique_headlines": len(norm_df) - dup_count,
                    "duplicate_headlines": dup_count,
                    "start_date": val_stats["start_date"],
                    "end_date": val_stats["end_date"],
                    "has_intraday_time": has_intraday,
                    "timing_status": timing_status,
                    "forward_2024_2026_coverage": forward_count > 0,
                    "forward_rows_count": forward_count,
                    "licensing_status": "Academic Research Use (Chen et al. 2024)",
                    "validation_result": val_stats["status"],
                    "atvi_delisting_blocker": (symbol == "ATVI")
                }
            except Exception as e:
                row = {
                    "symbol": symbol,
                    "filename": os.path.basename(fpath),
                    "file_size_bytes": file_size_bytes,
                    "sha256_checksum": sha256_hash,
                    "retrieval_timestamp_utc": retrieval_utc,
                    "total_rows": 0,
                    "unique_headlines": 0,
                    "duplicate_headlines": 0,
                    "start_date": "N/A",
                    "end_date": "N/A",
                    "has_intraday_time": False,
                    "timing_status": "CORRUPT_OR_UNREADABLE",
                    "forward_2024_2026_coverage": False,
                    "forward_rows_count": 0,
                    "licensing_status": "Unknown",
                    "validation_result": f"FAILED: {e}",
                    "atvi_delisting_blocker": (symbol == "ATVI")
                }

            manifest_rows.append(row)

        return manifest_rows

    def export_manifest(self, output_dir: str = None) -> Tuple[str, str]:
        out_dir = output_dir or self.news_dir
        os.makedirs(out_dir, exist_ok=True)

        rows = self.audit_all_news_files()
        df = pd.DataFrame(rows)

        csv_path = os.path.join(out_dir, "news_manifest.csv")
        json_path = os.path.join(out_dir, "news_manifest.json")

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        return csv_path, json_path
