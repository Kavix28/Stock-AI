"""
U.S. SEC EDGAR corporate disclosure provider.

Fetches official regulatory filings (8-K material events, 10-K, 10-Q) from
data.sec.gov with certified second-level UTC timestamps (acceptanceDateTime).
Public domain regulatory data compliant with SEC Fair Access Policy.
"""

import os
from typing import Dict, List, Any, Optional
import pandas as pd
import requests

from src.stock_ai.data.news.base import (
    NewsDataProvider,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderDataError
)
from src.stock_ai.data.news.schema import CanonicalNewsSchema


TICKER_CIK_MAP = {
    "AAPL": "0000320193",
    "AMT": "0001053507",
    "ATVI": "0000718877",
    "BIO": "0000012208",
    "JNJ": "0000200406",
    "NDAQ": "0001120193",
    "NFLX": "0001065280",
    "NVDA": "0001045810",
    "PLD": "0001045609",
    "SBUX": "0000829224",
    "SCHW": "0000316709",
    "TSLA": "0001318605"
}


class SecEdgarNewsProvider(NewsDataProvider):
    """
    Fetches official corporate filings from SEC EDGAR API.
    
    Provides second-level UTC timestamps (acceptanceDateTime).
    Complies with SEC Fair Access policy by requiring User-Agent header.
    """

    BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT",
            "StockAI-ResearchProject/1.0 (academic-audit@stock-ai.org)"
        )

    def get_provider_name(self) -> str:
        return "sec_edgar_disclosures"

    def is_configured(self) -> bool:
        return bool(self.user_agent and "@" in self.user_agent)

    def get_required_credentials(self) -> List[str]:
        return ["SEC_EDGAR_USER_AGENT"]

    def get_provider_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": self.get_provider_name(),
            "data_source": "U.S. Securities and Exchange Commission (SEC EDGAR)",
            "licensing": "Public Domain (U.S. Government Work)",
            "timestamp_quality": "POINT-IN-TIME CERTIFIED (Second-level UTC acceptanceDateTime)",
            "intraday_precision": True,
            "historical_range": "Full SEC electronic history (1994-present)",
            "legal_usability": "Free public access; requires compliant User-Agent email header",
            "content_type": "Official Form 8-K (Material Events), 10-Q, and 10-K Corporate Disclosures"
        }

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol_upper = symbol.upper().strip()
        cik = TICKER_CIK_MAP.get(symbol_upper)
        if not cik:
            raise ProviderDataError(f"CIK not mapped for ticker '{symbol_upper}'. Available: {list(TICKER_CIK_MAP.keys())}")

        if not self.is_configured():
            raise ProviderConfigurationError(
                "SecEdgarNewsProvider requires a valid User-Agent with contact email per SEC guidelines. "
                "Set 'SEC_EDGAR_USER_AGENT=YourName contact@domain.com'"
            )

        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        url = self.BASE_URL.format(cik=cik)

        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as e:
            raise ProviderDataError(f"SEC EDGAR request failed for {symbol_upper}: {e}")

        if resp.status_code == 429:
            raise ProviderRateLimitError("SEC EDGAR rate limit exceeded (>10 requests/sec).")
        if resp.status_code != 200:
            raise ProviderDataError(f"SEC EDGAR returned HTTP {resp.status_code} for CIK {cik}")

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent or "form" not in recent:
            return pd.DataFrame()

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        acceptance_datetimes = recent.get("acceptanceDateTime", [])
        primary_docs = recent.get("primaryDocDescription", [])
        accession_nums = recent.get("accessionNumber", [])

        rows = []
        for i in range(len(forms)):
            form_type = forms[i]
            # Focus on material corporate news filings (8-K, 10-Q, 10-K)
            if form_type not in ["8-K", "8-K/A", "10-Q", "10-K", "6-K"]:
                continue

            accept_dt = acceptance_datetimes[i] if i < len(acceptance_datetimes) else filing_dates[i]
            doc_desc = primary_docs[i] if i < len(primary_docs) and primary_docs[i] else f"Form {form_type} Disclosure"
            acc_num = accession_nums[i] if i < len(accession_nums) else ""

            headline = f"SEC {form_type} Filing: {doc_desc}"
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_num.replace('-', '')}/{acc_num}-index.htm"

            rows.append({
                "title": headline,
                "date": accept_dt,
                "source": "U.S. SEC EDGAR",
                "url": doc_url
            })

        if not rows:
            return pd.DataFrame()

        raw_df = pd.DataFrame(rows)
        normalized = CanonicalNewsSchema.normalize_dataframe(
            raw_df,
            symbol=symbol_upper,
            provider=self.get_provider_name(),
            default_source="U.S. SEC EDGAR"
        )

        mask = (normalized["published_date"] >= start_date) & (normalized["published_date"] <= end_date)
        return normalized[mask].copy().reset_index(drop=True)
