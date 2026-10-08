import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from stock_ai.config.settings import AppConfig, load_config
from stock_ai.data.market.base import MarketDataProvider
from stock_ai.data.market.yahooquery_provider import YahooQueryMarketDataProvider
from stock_ai.data.market.schema import CanonicalMarketSchema
from stock_ai.data.market.validator import MarketDataValidator, ValidationIssue
from stock_ai.data.market.utils import TemporalNormalizer
from stock_ai.logger import get_logger

logger = get_logger(__name__)


class MarketDataService:
    """
    Market Data Download & Quality Assurance Service.
    Orchestrates download, raw storage, validation, processed Parquet export,
    dataset manifest creation, and quality report generation.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        provider: Optional[MarketDataProvider] = None
    ):
        self.config = config or load_config()
        self.provider = provider or YahooQueryMarketDataProvider(
            retry_attempts=self.config.market_data.retry_attempts,
            retry_backoff_factor=self.config.market_data.retry_backoff_factor,
            timeout=self.config.market_data.timeout,
            rate_limit_delay=self.config.market_data.rate_limit_delay,
        )

        self.raw_dir = Path(self.config.storage.raw_market_dir)
        self.processed_dir = Path(self.config.storage.processed_market_dir)
        self.reports_dir = Path(self.config.storage.reports_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch market data for multiple symbols, process, validate, and write outputs.
        """
        target_symbols = symbols or [s.symbol for s in self.config.stocks]
        start = start_date or self.config.market_data.acquisition_range.start_date
        end = end_date or self.config.market_data.acquisition_range.end_date

        if dry_run:
            logger.info("=== DRY RUN MODE ===")
            logger.info(f"Target symbols: {target_symbols}")
            logger.info(f"Date range: {start} to {end}")
            logger.info(f"Raw output path: {self.raw_dir}")
            logger.info(f"Processed output path: {self.processed_dir}")
            return {
                "dry_run": True,
                "symbols": target_symbols,
                "start_date": start,
                "end_date": end,
                "status": "dry_run_complete"
            }

        results: Dict[str, Dict[str, Any]] = {}
        all_passed = True

        for symbol in target_symbols:
            logger.info(f"Processing market data for '{symbol}'...")
            try:
                res = self.process_symbol(symbol, start, end, force=force)
                results[symbol] = res
                if res["status"] == "FAILED":
                    all_passed = False
            except Exception as e:
                logger.error(f"Error processing symbol '{symbol}': {e}", exc_info=True)
                results[symbol] = {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": str(e),
                    "rows": 0
                }
                all_passed = False

        # Generate Manifest and Quality Report
        manifest_data = self.generate_manifest(results, start, end)
        quality_report = self.generate_quality_report(results, start, end)

        return {
            "success": all_passed,
            "results": results,
            "manifest_path": str(Path(self.config.storage.manifest_file).resolve()),
            "quality_report_path": str(Path(self.config.storage.quality_report_file).resolve())
        }

    def process_symbol(
        self, symbol: str, start_date: str, end_date: str, force: bool = False
    ) -> Dict[str, Any]:
        """Fetch, raw preserve, normalize, validate, and write processed parquet for a symbol."""
        raw_csv_path = self.raw_dir / f"{symbol}.csv"
        processed_parquet_path = self.processed_dir / f"{symbol}.parquet"
        processed_csv_path = self.processed_dir / f"{symbol}.csv"

        df_raw: Optional[pd.DataFrame] = None
        source_type = "API"

        # Check Cache / Raw Data Policy
        if raw_csv_path.exists() and not force:
            logger.info(f"Raw file for '{symbol}' exists at {raw_csv_path}. Reading from cache...")
            df_raw = pd.read_csv(raw_csv_path)
            source_type = "CACHE"
        else:
            df_raw = self.provider.fetch_historical_data(symbol, start_date, end_date)
            # Save Raw CSV immutably
            df_raw.to_csv(raw_csv_path, index=True)
            logger.info(f"Raw data saved to {raw_csv_path}")

        # Schema Normalization
        df_norm = CanonicalMarketSchema.normalize(
            df_raw, symbol=symbol, column_mapping=self.config.market_data.provider_column_mapping
        )

        # Date Filtering to exact requested range if needed
        df_norm = df_norm[(df_norm["date"] >= start_date) & (df_norm["date"] <= end_date)].reset_index(drop=True)

        # Data Validation
        validator = MarketDataValidator(expected_symbol=symbol)
        is_valid, issues, df_clean = validator.validate(df_norm)

        # Save Processed Data
        df_clean.to_parquet(processed_parquet_path, index=False)
        df_clean.to_csv(processed_csv_path, index=False)
        logger.info(f"Processed dataset saved to {processed_parquet_path}")

        gap_analysis = TemporalNormalizer.analyze_trading_gaps(df_clean)

        return {
            "symbol": symbol,
            "status": "PASS" if is_valid else "FAILED",
            "source": source_type,
            "rows": len(df_clean),
            "start_date": df_clean["date"].min() if not df_clean.empty else start_date,
            "end_date": df_clean["date"].max() if not df_clean.empty else end_date,
            "issues": [issue.to_dict() for issue in issues],
            "missing_values": int(df_clean.isnull().sum().sum()),
            "duplicates": len(df_norm) - len(df_clean),
            "unexpected_gaps": gap_analysis["unexpected_gap_count"],
            "raw_file": str(raw_csv_path),
            "processed_file": str(processed_parquet_path)
        }

    def generate_manifest(
        self, results: Dict[str, Dict[str, Any]], start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """Generate data/processed/market/manifest.json."""
        manifest_path = Path(self.config.storage.manifest_file)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        symbols_meta = {}
        total_rows = 0

        for symbol, res in results.items():
            if res.get("status") != "FAILED":
                symbols_meta[symbol] = {
                    "rows": res.get("rows", 0),
                    "first_date": res.get("start_date"),
                    "last_date": res.get("end_date"),
                    "missing_values": res.get("missing_values", 0),
                    "duplicates": res.get("duplicates", 0),
                    "unexpected_gaps": res.get("unexpected_gaps", 0),
                    "status": res.get("status", "UNKNOWN")
                }
                total_rows += res.get("rows", 0)
            else:
                symbols_meta[symbol] = {
                    "status": "FAILED",
                    "error": res.get("error", "Unknown error")
                }

        manifest = {
            "dataset": "paper_market_data",
            "version": "1.0.0",
            "provider": self.provider.get_provider_name(),
            "download_timestamp": datetime.utcnow().isoformat() + "Z",
            "start_date": start_date,
            "end_date": end_date,
            "total_symbols": len(results),
            "total_rows": total_rows,
            "canonical_columns": self.config.market_data.required_columns,
            "symbols": symbols_meta,
            "overall_status": "PASS" if all(r.get("status") == "PASS" for r in results.values()) else "FAILED"
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Dataset manifest written to {manifest_path}")
        return manifest

    def generate_quality_report(
        self, results: Dict[str, Dict[str, Any]], start_date: str, end_date: str
    ) -> str:
        """Generate reports/market_data_quality_report.md."""
        report_path = Path(self.config.storage.quality_report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Market Data Quality Assurance Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Provider:** {self.provider.get_provider_name()}",
            f"**Target Range:** {start_date} to {end_date}",
            "",
            "## Summary Table",
            "",
            "| Symbol | Status | Rows | First Date | Last Date | Missing | Dups | Unexpected Gaps |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        total_rows = 0
        pass_count = 0

        for symbol, res in results.items():
            if res.get("status") == "PASS":
                pass_count += 1
                total_rows += res.get("rows", 0)
                lines.append(
                    f"| **{symbol}** | PASS | {res.get('rows')} | {res.get('start_date')} | {res.get('end_date')} | {res.get('missing_values')} | {res.get('duplicates')} | {res.get('unexpected_gaps')} |"
                )
            else:
                lines.append(
                    f"| **{symbol}** | FAILED | 0 | - | - | - | - | - |"
                )

        lines.extend([
            "",
            "## Overall Quality Status",
            "",
            f"- **Total Stocks Audited:** {len(results)}",
            f"- **Passed Validation:** {pass_count} / {len(results)}",
            f"- **Total Rows Ingested:** {total_rows}",
            f"- **Pipeline Status:** {'SUCCESS' if pass_count == len(results) else 'ACTION REQUIRED'}",
            "",
            "## Detailed Stock Quality Breakdown",
            ""
        ])

        for symbol, res in results.items():
            lines.append(f"### {symbol}")
            if res.get("status") == "FAILED":
                lines.append(f"- **Status:** FAILED")
                lines.append(f"- **Error:** {res.get('error')}")
            else:
                lines.append(f"- **Status:** {res.get('status')}")
                lines.append(f"- **Rows:** {res.get('rows')}")
                lines.append(f"- **Date Range:** {res.get('start_date')} to {res.get('end_date')}")
                lines.append(f"- **Missing Values:** {res.get('missing_values')}")
                lines.append(f"- **Duplicates Removed:** {res.get('duplicates')}")
                lines.append(f"- **Unexpected Trading Gaps:** {res.get('unexpected_gaps')}")
                lines.append(f"- **Raw File:** `{res.get('raw_file')}`")
                lines.append(f"- **Processed File:** `{res.get('processed_file')}`")
                
                issues = res.get("issues", [])
                if issues:
                    lines.append("- **Validation Findings:**")
                    for iss in issues:
                        lines.append(f"  - [{iss['level']}] ({iss['category']}) {iss['message']}")
            lines.append("")

        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Quality report written to {report_path}")
        return report_content
