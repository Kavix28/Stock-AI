"""
Causal alignment pipeline for pairing financial news representations with market sessions.

Handles:
- US Eastern / UTC timezone conversions with Daylight Saving Time handling.
- Strict causal cutoff enforcement (zero future-timestamp leakage).
- Headline deduplication and text normalization.
- Explicit missing news handling without synthetic fabrication.
- Dual alignment modes:
  1. PRE_MARKET_WINDOW (Intraday certified: Close(t-1) 16:00 to Open(t) 09:30 EST)
  2. DAILY_DATE_CONVENTIONAL (Chen et al. 2024 historical replication, tagged TIMING-LIMITED)
"""

from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import pytz

from src.stock_ai.data.news.schema import (
    CanonicalNewsArticle,
    DailyNewsFeature,
    CanonicalNewsSchema
)


# Timezone constants
EASTERN_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC

# Market hours in Eastern Time
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)


class CausalAlignmentError(Exception):
    """Raised when causal alignment or anti-leakage invariants are violated."""
    pass


class NewsMarketAligner:
    """
    Aligns canonical news articles with target trading days.
    
    Ensures that for target trading day T, no information published after
    the specified prediction cutoff is admitted into features.
    """

    def __init__(
        self,
        alignment_mode: str = "PRE_MARKET_WINDOW",
        max_headlines_per_day: int = 10,
        no_news_placeholder: str = ""
    ):
        """
        Args:
            alignment_mode:
                'PRE_MARKET_WINDOW': News between Close(t-1) 16:00 EST and Open(t) 09:30 EST.
                'DAILY_DATE_CONVENTIONAL': Matches published_date == target_date (tagged TIMING-LIMITED).
                'DAILY_DATE_LAG1': Matches published_date == trading_dates[t-1].
            max_headlines_per_day: Maximum unique headlines to concatenate.
            no_news_placeholder: String representation for days with zero news.
        """
        if alignment_mode not in ["PRE_MARKET_WINDOW", "DAILY_DATE_CONVENTIONAL", "DAILY_DATE_LAG1"]:
            raise ValueError(f"Unknown alignment mode: {alignment_mode}")

        self.alignment_mode = alignment_mode
        self.max_headlines_per_day = max_headlines_per_day
        self.no_news_placeholder = no_news_placeholder

    @staticmethod
    def get_market_session_bounds_utc(
        prev_trading_date_str: str,
        target_trading_date_str: str
    ) -> Tuple[datetime, datetime]:
        """
        Computes UTC start and cutoff datetimes for a pre-market observation window.
        
        Window: Market Close of prev_trading_date (16:00 EST) to Market Open of target date (09:30 EST).
        """
        prev_dt = datetime.strptime(prev_trading_date_str, "%Y-%m-%d")
        target_dt = datetime.strptime(target_trading_date_str, "%Y-%m-%d")

        # Localize in America/New_York to correctly handle EST vs EDT
        close_prev_local = EASTERN_TZ.localize(
            datetime.combine(prev_dt.date(), MARKET_CLOSE_TIME)
        )
        open_target_local = EASTERN_TZ.localize(
            datetime.combine(target_dt.date(), MARKET_OPEN_TIME)
        )

        return close_prev_local.astimezone(UTC_TZ), open_target_local.astimezone(UTC_TZ)

    def align_ticker_news(
        self,
        news_df: pd.DataFrame,
        trading_dates: List[str],
        symbol: str
    ) -> pd.DataFrame:
        """
        Aligns news articles for a single ticker to a chronological list of eligible trading dates.

        Args:
            news_df: DataFrame matching CanonicalNewsSchema.
            trading_dates: Sorted list of trading date strings (YYYY-MM-DD).
            symbol: Ticker symbol.

        Returns:
            pd.DataFrame with columns:
                ['symbol', 'target_date', 'combined_text', 'article_count',
                 'has_news', 'timing_quality', 'article_ids', 'duplicate_count']
        """
        if len(trading_dates) < 2:
            raise CausalAlignmentError("At least 2 trading dates are required to form causal windows.")

        # Ensure trading dates are strictly monotonically increasing
        t_dts = [datetime.strptime(d, "%Y-%m-%d") for d in trading_dates]
        for i in range(1, len(t_dts)):
            if t_dts[i] <= t_dts[i - 1]:
                raise CausalAlignmentError(f"Trading dates must be strictly increasing. Violation at {trading_dates[i]}")

        # Check timestamp quality of the input news
        has_intraday = False
        if not news_df.empty and "has_intraday_time" in news_df.columns:
            has_intraday = bool(news_df["has_intraday_time"].any())

        if self.alignment_mode == "PRE_MARKET_WINDOW" and not has_intraday:
            # Fall back to daily conventional with warning if intraday timestamps are absent
            active_mode = "DAILY_DATE_CONVENTIONAL"
            timing_quality = "TIMING_LIMITED_DAILY_ONLY"
        elif self.alignment_mode == "PRE_MARKET_WINDOW" and has_intraday:
            active_mode = "PRE_MARKET_WINDOW"
            timing_quality = "POINT_IN_TIME_CERTIFIED"
        else:
            active_mode = self.alignment_mode
            timing_quality = "TIMING_LIMITED_DAILY_ONLY"

        # Prepare parsed timestamps if available
        news_work = news_df.copy()
        if not news_work.empty and "published_at_utc" in news_work.columns and has_intraday:
            news_work["dt_utc"] = pd.to_datetime(news_work["published_at_utc"], utc=True)
        else:
            news_work["dt_utc"] = None

        aligned_records = []

        # Iterate over trading dates starting from index 1 (since index 0 has no t-1)
        for idx in range(1, len(trading_dates)):
            t_curr = trading_dates[idx]
            t_prev = trading_dates[idx - 1]

            articles_for_day: List[Dict] = []

            if news_work.empty:
                # No news at all
                pass
            elif active_mode == "PRE_MARKET_WINDOW":
                start_utc, end_utc = self.get_market_session_bounds_utc(t_prev, t_curr)
                mask = (news_work["dt_utc"] >= start_utc) & (news_work["dt_utc"] < end_utc)
                matched = news_work[mask]

                # Anti-leakage invariant: No article >= end_utc can enter!
                for _, row in matched.iterrows():
                    if row["dt_utc"] >= end_utc:
                        raise CausalAlignmentError(
                            f"LEAKAGE DETECTED! Article {row['article_id']} published at {row['dt_utc']} "
                            f"enters feature for target date {t_curr} with cutoff {end_utc}!"
                        )
                    articles_for_day.append(row.to_dict())

            elif active_mode == "DAILY_DATE_CONVENTIONAL":
                matched = news_work[news_work["published_date"] == t_curr]
                articles_for_day = matched.to_dict("records")

            elif active_mode == "DAILY_DATE_LAG1":
                matched = news_work[news_work["published_date"] == t_prev]
                articles_for_day = matched.to_dict("records")

            # Deduplication
            seen_titles: Set[str] = set()
            unique_articles: List[Dict] = []
            duplicate_count = 0

            for art in articles_for_day:
                clean_t = CanonicalNewsSchema.clean_headline_text(art.get("title", ""))
                t_lower = clean_t.lower()
                if not t_lower:
                    continue
                if t_lower in seen_titles:
                    duplicate_count += 1
                    continue
                seen_titles.add(t_lower)
                unique_articles.append(art)

            # Cap at max headlines
            selected_articles = unique_articles[: self.max_headlines_per_day]

            if selected_articles:
                # Combine unique headlines with period separator
                combined = ". ".join([a["title"].rstrip(".") for a in selected_articles]) + "."
                art_ids = [a.get("article_id", "") for a in selected_articles]
                has_news = True
            else:
                combined = self.no_news_placeholder
                art_ids = []
                has_news = False

            aligned_records.append({
                "symbol": symbol.upper(),
                "target_date": t_curr,
                "combined_text": combined,
                "article_count": len(selected_articles),
                "has_news": has_news,
                "timing_quality": timing_quality,
                "article_ids": art_ids,
                "duplicate_count": duplicate_count
            })

        return pd.DataFrame(aligned_records)
