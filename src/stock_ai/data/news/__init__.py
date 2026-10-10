"""
Financial news data ingestion, canonicalization, and causal alignment package.
"""

from .schema import (
    CanonicalNewsSchema,
    CanonicalNewsArticle,
    DailyNewsFeature,
    CANONICAL_NEWS_COLUMNS
)
from .base import (
    NewsDataProvider,
    NewsProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderDataError
)
from .local_provider import LocalCanonicalNewsProvider
from .polygon_provider import PolygonNewsProvider
from .tiingo_provider import TiingoNewsProvider
from .sec_edgar_provider import SecEdgarNewsProvider
from .alignment import NewsMarketAligner, CausalAlignmentError
from .validator import NewsDataValidator, NewsValidationError
from .manifest import NewsManifestGenerator
from .cache import NewsEmbeddingCache

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
    "NewsManifestGenerator",
    "NewsEmbeddingCache"
]
