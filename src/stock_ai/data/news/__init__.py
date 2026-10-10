"""
Financial news data ingestion, canonicalization, and causal alignment package.
"""

from src.stock_ai.data.news.schema import (
    CanonicalNewsSchema,
    CanonicalNewsArticle,
    DailyNewsFeature,
    CANONICAL_NEWS_COLUMNS
)
from src.stock_ai.data.news.base import (
    NewsDataProvider,
    NewsProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderDataError
)
from src.stock_ai.data.news.local_provider import LocalCanonicalNewsProvider
from src.stock_ai.data.news.polygon_provider import PolygonNewsProvider
from src.stock_ai.data.news.tiingo_provider import TiingoNewsProvider
from src.stock_ai.data.news.sec_edgar_provider import SecEdgarNewsProvider
from src.stock_ai.data.news.alignment import NewsMarketAligner, CausalAlignmentError
from src.stock_ai.data.news.validator import NewsDataValidator, NewsValidationError
from src.stock_ai.data.news.manifest import NewsManifestGenerator

__all__ = [
    "CanonicalNewsSchema",
    "CanonicalNewsArticle",
    "DailyNewsFeature",
    "CANONICAL_NEWS_COLUMNS",
    "NewsDataProvider",
    "NewsProviderError",
    "ProviderConfigurationError",
    "ProviderRateLimitError",
    "ProviderDataError",
    "LocalCanonicalNewsProvider",
    "PolygonNewsProvider",
    "TiingoNewsProvider",
    "SecEdgarNewsProvider",
    "NewsMarketAligner",
    "CausalAlignmentError",
    "NewsDataValidator",
    "NewsValidationError",
    "NewsManifestGenerator"
]
