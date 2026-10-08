import time
import logging
from typing import Optional
import pandas as pd
from yahooquery import Ticker

from stock_ai.data.market.base import MarketDataProvider
from stock_ai.logger import get_logger

logger = get_logger(__name__)


class YahooQueryMarketDataProvider(MarketDataProvider):
    """
    Market data provider implementation using YahooQuery.
    Handles network retries, timeouts, and throttling.
    """

    def __init__(
        self,
        retry_attempts: int = 3,
        retry_backoff_factor: float = 1.5,
        timeout: int = 30,
        rate_limit_delay: float = 0.5
    ):
        self.retry_attempts = retry_attempts
        self.retry_backoff_factor = retry_backoff_factor
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay

    def get_provider_name(self) -> str:
        return "YahooQuery"

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol via YahooQuery with bounded retry logic.
        """
        last_exception = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.info(f"Fetching data for '{symbol}' from {start_date} to {end_date} (Attempt {attempt}/{self.retry_attempts})...")
                
                ticker = Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                if df is None or df.empty:
                    raise ValueError(f"No market data returned by YahooQuery for ticker '{symbol}'.")

                # If YahooQuery returns an error string dict instead of DataFrame
                if isinstance(df, dict) or "error" in df:
                    raise ValueError(f"YahooQuery returned error for ticker '{symbol}': {df}")

                logger.info(f"Successfully fetched {len(df)} rows for '{symbol}'.")
                return df

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed for '{symbol}': {e}")
                if attempt < self.retry_attempts:
                    sleep_time = self.rate_limit_delay * (self.retry_backoff_factor ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"Failed to fetch market data for '{symbol}' after {self.retry_attempts} attempts. Error: {last_exception}")
