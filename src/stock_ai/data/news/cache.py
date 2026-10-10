"""
Embedding Cache for FinBERT Financial News Representations.

Provides deterministic on-disk caching of 768-dimensional FinBERT embeddings
indexed by ticker and trading date, with SHA-256 provenance manifests.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from stock_ai.models.news.finbert import FinBertNewsEncoder
except ImportError:
    from src.stock_ai.models.news.finbert import FinBertNewsEncoder

logger = logging.getLogger(__name__)


class NewsEmbeddingCache:
    """
    Manages generation, storage, retrieval, and invalidation of FinBERT news embeddings.
    """

    def __init__(
        self,
        cache_dir: str = "data/processed/embeddings",
        encoder: Optional[FinBertNewsEncoder] = None,
        model_name: str = "ProsusAI/finbert",
        pooling_method: str = "cls",
        version: str = "v1.0"
    ):
        self.cache_dir = cache_dir
        self.encoder = encoder or FinBertNewsEncoder(
            model_name=model_name,
            pooling_method=pooling_method,
            lazy_load=True
        )
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.version = version
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Returns deterministic 16-char hex hash of text content."""
        if not text:
            return "NO_NEWS"
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def get_ticker_cache_path(self, ticker: str) -> str:
        """Returns path to ticker embedding file."""
        return os.path.join(self.cache_dir, f"{ticker.upper()}_embeddings.npz")

    def get_ticker_manifest_path(self, ticker: str) -> str:
        """Returns path to ticker embedding metadata manifest."""
        return os.path.join(self.cache_dir, f"{ticker.upper()}_manifest.json")

    def build_ticker_cache(
        self,
        ticker: str,
        aligned_news_df: pd.DataFrame,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Builds and saves the embedding cache for a given ticker from an aligned news DataFrame.
        
        aligned_news_df must have columns:
        - target_date (YYYY-MM-DD)
        - combined_text (str)
        - has_news (bool)
        - article_count (int)
        """
        ticker = ticker.upper()
        cache_path = self.get_ticker_cache_path(ticker)
        manifest_path = self.get_ticker_manifest_path(ticker)

        if not force_rebuild and os.path.exists(cache_path) and os.path.exists(manifest_path):
            logger.info(f"Using existing embedding cache for {ticker}: {cache_path}")
            return self.load_ticker_manifest(ticker)

        logger.info(f"Generating FinBERT embedding cache for {ticker} ({len(aligned_news_df)} dates)...")

        dates = aligned_news_df["target_date"].astype(str).tolist()
        texts = aligned_news_df["combined_text"].fillna("").astype(str).tolist()
        has_news_flags = aligned_news_df["has_news"].astype(bool).tolist()
        article_counts = aligned_news_df["article_count"].astype(int).tolist()

        # Batch encode all texts using FinBERT
        embeddings = self.encoder.encode_batch(texts, batch_size=32)

        # Compute text hashes
        text_hashes = [self.compute_text_hash(t) for t in texts]

        # Save compressed numpy archive
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            dates=np.array(dates, dtype=object),
            has_news=np.array(has_news_flags, dtype=bool),
            article_counts=np.array(article_counts, dtype=np.int32)
        )

        manifest = {
            "ticker": ticker,
            "version": self.version,
            "model_name": self.model_name,
            "pooling_method": self.pooling_method,
            "embedding_dim": int(embeddings.shape[1]),
            "total_dates": len(dates),
            "dates_with_news": int(sum(has_news_flags)),
            "news_coverage_pct": round((sum(has_news_flags) / len(dates) * 100.0) if dates else 0.0, 2),
            "start_date": dates[0] if dates else None,
            "end_date": dates[-1] if dates else None,
            "cache_file": os.path.basename(cache_path),
            "cache_file_size_bytes": os.path.getsize(cache_path),
            "text_hashes_sample": text_hashes[:5]
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Embedding cache saved for {ticker}: {cache_path} ({manifest['dates_with_news']}/{manifest['total_dates']} news days)")
        return manifest

    def load_ticker_embeddings(self, ticker: str) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
        """
        Loads embeddings and date-to-index mapping for a ticker.
        
        Returns:
            embeddings: np.ndarray of shape (N, 768)
            date_to_idx: dict mapping 'YYYY-MM-DD' -> row index
            has_news: boolean array of shape (N,)
        """
        ticker = ticker.upper()
        cache_path = self.get_ticker_cache_path(ticker)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Embedding cache for ticker '{ticker}' not found at {cache_path}.")

        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        dates = data["dates"]
        has_news = data["has_news"]

        date_to_idx = {str(d): idx for idx, d in enumerate(dates)}
        return embeddings, date_to_idx, has_news

    def load_ticker_manifest(self, ticker: str) -> Dict[str, Any]:
        """Loads metadata manifest for a ticker."""
        manifest_path = self.get_ticker_manifest_path(ticker.upper())
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Embedding manifest for ticker '{ticker}' not found at {manifest_path}.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_embedding_for_date(self, ticker: str, date_str: str) -> Tuple[np.ndarray, bool]:
        """
        Retrieves embedding for a specific ticker and date.
        If date is not cached or has no news, returns (zero_vector, False).
        """
        try:
            embeddings, date_to_idx, has_news = self.load_ticker_embeddings(ticker)
            if date_str in date_to_idx:
                idx = date_to_idx[date_str]
                return embeddings[idx], bool(has_news[idx])
        except Exception as e:
            logger.warning(f"Could not load embedding for {ticker} on {date_str}: {e}")

        return self.encoder.get_zero_news_vector(), False
