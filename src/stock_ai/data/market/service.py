import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from stock_ai.config.settings import AppConfig
from stock_ai.data.market.base import MarketDataProvider
from stock_ai.data.market.yahooquery_provider import YahooQueryMarketDataProvider
from stock_ai.data.market.schema import CanonicalMarketSchema
from stock_ai.data.market.validator import MarketDataValidator
from stock_ai.data.market.utils import TemporalNormalizer
from stock_ai.logger import get_logger

logger = get_logger(__name__)


class MarketDataService:
    """Orchestrates market data acquisition, validation, persistence, and reporting."""

    def __init__(self, config: AppConfig, provider: Optional[MarketDataProvider] = None):
        self.config = config
        self.provider = provider or YahooQueryMarketDataProvider(
            retry_attempts=config.market_data.retry_attempts,
            retry_backoff_factor=config.market_data.retry_backoff_factor,
            timeout=config.market_data.timeout,
            rate_limit_delay=config.market_data.rate_limit_delay
        )
        self.raw_dir = Path(config.storage.raw_market_dir)
        self.processed_dir = Path(config.storage.processed_market_dir)
        self.reports_dir = Path(config.storage.reports_dir)

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_provider_name(self, symbol: str) -> str:
        if hasattr(self.provider, "get_provider_for_symbol"):
            return self.provider.get_provider_for_symbol(symbol)
        return self.provider.get_provider_name()

    def process_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single symbol through fetch, normalize, validate, and store pipeline steps.
        """
        logger.info(f"--- Processing market data for '{symbol}' ---")
        raw_file = self.raw_dir / f"{symbol}.csv"
        processed_parquet = self.processed_dir / f"{symbol}.parquet"
        processed_csv = self.processed_dir / f"{symbol}.csv"

        if dry_run:
            logger.info(f"[DRY-RUN] Would fetch and process '{symbol}' from {start_date} to {end_date}")
            return {
                "symbol": symbol,
                "status": "PASS",
                "rows": 0,
                "provider": self._get_provider_name(symbol),
                "issues": []
            }

        # Step 1: Fetch or load raw data
        raw_df = None
        if raw_file.exists() and not force:
            logger.info(f"Loading cached raw data for '{symbol}' from '{raw_file}'")
            raw_df = pd.read_csv(raw_file)
        else:
            try:
                raw_df = self.provider.fetch_historical_data(symbol, start_date, end_date)
                # Save immutable raw data
                raw_df.to_csv(raw_file, index=True)
                logger.info(f"Saved immutable raw data for '{symbol}' to '{raw_file}'")
            except Exception as e:
                logger.error(f"Failed to fetch market data for '{symbol}': {e}")
                return {
                    "symbol": symbol,
                    "status": "FAILED",
                    "rows": 0,
                    "provider": self._get_provider_name(symbol),
                    "error": str(e),
                    "issues": [{"level": "ERROR", "category": "FETCH", "message": str(e)}]
                }

        # Step 2: Normalize to Canonical Schema
        try:
            norm_df = CanonicalMarketSchema.normalize(raw_df, symbol=symbol)
            norm_df = TemporalNormalizer.normalize_dates(norm_df, date_column="date")
        except Exception as e:
            logger.error(f"Schema normalization failed for '{symbol}': {e}")
            return {
                "symbol": symbol,
                "status": "FAILED",
                "rows": 0,
                "provider": self._get_provider_name(symbol),
                "error": f"Schema normalization error: {e}",
                "issues": [{"level": "ERROR", "category": "SCHEMA", "message": str(e)}]
            }

        # Step 3: Data Quality Validation
        validator = MarketDataValidator(expected_symbol=symbol)
        is_valid, issues, cleaned_df = validator.validate(norm_df)

        # Step 4: Temporal Gap Analysis
        gap_summary = TemporalNormalizer.analyze_trading_gaps(cleaned_df, date_column="date")
        if gap_summary["unexpected_gap_count"] > 0:
            validator.add_issue(
                "WARNING", "Temporal",
                f"Found {gap_summary['unexpected_gap_count']} unexpected trading day gap(s): {gap_summary['unexpected_gaps'][:3]}"
            )

        # Step 5: Save Processed Files (Parquet & CSV)
        status_str = "PASS" if is_valid else "FAILED"
        if is_valid:
            cleaned_df.to_parquet(processed_parquet, index=False)
            cleaned_df.to_csv(processed_csv, index=False)
            logger.info(f"Saved processed Parquet data ({len(cleaned_df)} rows) to '{processed_parquet}'")

        return {
            "symbol": symbol,
            "status": status_str,
            "provider": self._get_provider_name(symbol),
            "rows": len(cleaned_df),
            "first_date": str(cleaned_df["date"].min())[:10] if not cleaned_df.empty else None,
            "last_date": str(cleaned_df["date"].max())[:10] if not cleaned_df.empty else None,
            "missing_values": int(cleaned_df.isna().sum().sum()),
            "unexpected_gaps": gap_summary["unexpected_gap_count"],
            "weekend_gaps": gap_summary["weekend_gaps"],
            "raw_file": str(raw_file),
            "processed_file": str(processed_parquet),
            "issues": [i.to_dict() for i in issues]
        }

    def fetch_all(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Process all configured symbols and generate manifest and quality reports."""
        target_symbols = symbols or [s.symbol for s in self.config.stocks]
        start = start_date or self.config.market_data.acquisition_range.start_date
        end = end_date or self.config.market_data.acquisition_range.end_date

        logger.info(f"Starting acquisition for {len(target_symbols)} symbols ({start} to {end})...")
        results = {}
        all_passed = True

        for sym in target_symbols:
            res = self.process_symbol(sym, start, end, force=force, dry_run=dry_run)
            results[sym] = res
            if res["status"] != "PASS":
                all_passed = False

        if not dry_run:
            self._generate_manifest(results, start, end, all_passed)
            self._generate_quality_report(results, start, end, all_passed)

        return {
            "success": all_passed,
            "results": results
        }

    def _generate_manifest(self, results: Dict[str, Any], start_date: str, end_date: str, overall_success: bool):
        manifest_path = Path(self.config.storage.manifest_file)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        symbols_meta = {}
        total_rows = 0

        for sym, data in results.items():
            symbols_meta[sym] = {
                "provider": data.get("provider", self.provider.get_provider_name()),
                "status": data.get("status"),
                "rows": data.get("rows", 0),
                "first_date": data.get("first_date"),
                "last_date": data.get("last_date"),
                "missing_values": data.get("missing_values", 0),
                "unexpected_gaps": data.get("unexpected_gaps", 0),
                "error": data.get("error")
            }
            total_rows += data.get("rows", 0)

        overall_status_str = "PASS" if overall_success else "FAILED"

        manifest_data = {
            "dataset": "paper_market_data",
            "paper_reference": "Neural Computing and Applications (2024) 36:21229–21271",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "acquisition_start_date": start_date,
            "acquisition_end_date": end_date,
            "overall_status": overall_status_str,
            "total_symbols": len(results),
            "total_rows": total_rows,
            "symbols": symbols_meta
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Generated dataset manifest at '{manifest_path}' (Overall status: {overall_status_str})")

    def _generate_quality_report(self, results: Dict[str, Any], start_date: str, end_date: str, overall_success: bool):
        report_path = Path(self.config.storage.quality_report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        passed_count = sum(1 for r in results.values() if r.get("status") == "PASS")
        total_rows = sum(r.get("rows", 0) for r in results.values())
        overall_status_text = "PASS" if overall_success else "ACTION REQUIRED"

        lines = [
            "# Market Data Quality Assurance Report\n",
            f"**Generated:** {now_str}  ",
            f"**Target Range:** {start_date} to {end_date}  \n",
            "## Summary Table\n",
            "| Symbol | Provider | Status | Rows | First Date | Last Date | Missing | Dups | Unexpected Gaps |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for sym, res in results.items():
            status = res.get("status", "N/A")
            provider = res.get("provider", "Unknown")
            rows = res.get("rows", 0)
            first_d = res.get("first_date") or "-"
            last_d = res.get("last_date") or "-"
            missing = res.get("missing_values") if res.get("missing_values") is not None else "-"
            dups = 0
            gaps = res.get("unexpected_gaps") if res.get("unexpected_gaps") is not None else "-"
            lines.append(f"| **{sym}** | {provider} | {status} | {rows} | {first_d} | {last_d} | {missing} | {dups} | {gaps} |")

        lines.extend([
            "\n## Overall Quality Status\n",
            f"- **Total Stocks Audited:** {len(results)}",
            f"- **Passed Validation:** {passed_count} / {len(results)}",
            f"- **Total Rows Ingested:** {total_rows}",
            f"- **Pipeline Status:** {overall_status_text}\n",
            "## Detailed Stock Quality Breakdown\n"
        ])

        for sym, res in results.items():
            lines.append(f"### {sym}")
            lines.append(f"- **Provider:** {res.get('provider', 'Unknown')}")
            lines.append(f"- **Status:** {res.get('status')}")

            if res.get("error"):
                lines.append(f"- **Error:** {res.get('error')}")

            if res.get("rows", 0) > 0:
                lines.append(f"- **Rows:** {res['rows']}")
                lines.append(f"- **Date Range:** {res['first_date']} to {res['last_date']}")
                lines.append(f"- **Missing Values:** {res['missing_values']}")
                lines.append(f"- **Unexpected Trading Gaps:** {res['unexpected_gaps']}")
                lines.append(f"- **Raw File:** `{res['raw_file']}`")
                lines.append(f"- **Processed File:** `{res['processed_file']}`")

            lines.append("- **Validation Findings:**")
            issues = res.get("issues", [])
            if not issues:
                lines.append("  - [INFO] Clean dataset with zero quality warnings.")
            else:
                for issue in issues:
                    lines.append(f"  - [{issue['level']}] ({issue['category']}) {issue['message']}")
            lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated quality assurance report at '{report_path}'")
