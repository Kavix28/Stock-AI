import pytest
import os
import json
import pandas as pd

from src.stock_ai.data.news.manifest import NewsManifestGenerator


def test_manifest_generator_exports_valid_files(tmp_path):
    # Use existing news directory
    gen = NewsManifestGenerator(news_dir="data/processed/news")
    csv_path, json_path = gen.export_manifest(output_dir=str(tmp_path))

    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "symbol" in df.columns
    assert "sha256_checksum" in df.columns
    assert "timing_status" in df.columns
    assert "licensing_status" in df.columns
    assert "atvi_delisting_blocker" in df.columns

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == len(df)
