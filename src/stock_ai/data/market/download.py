import argparse
import sys
from pathlib import Path

from stock_ai.config.settings import load_config
from stock_ai.data.market.service import MarketDataService
from stock_ai.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Market Data Acquisition & Quality Assurance Pipeline")
    parser.add_argument("--start", type=str, help="Acquisition start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Acquisition end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, help="Comma-separated ticker symbols (e.g. AAPL,NVDA,TSLA)")
    parser.add_argument("--force", action="store_true", help="Force fresh download from provider ignoring cache")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without downloading or writing files")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config file")

    args = parser.parse_args()

    try:
        config = load_config(base_config_path=Path(args.config))
        service = MarketDataService(config=config)

        symbols_list = None
        if args.symbols:
            symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

        result = service.fetch_all(
            symbols=symbols_list,
            start_date=args.start,
            end_date=args.end,
            force=args.force,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print("\nDry run completed successfully.")
            sys.exit(0)

        if result.get("success"):
            logger.info("Market data pipeline executed successfully for all requested symbols!")
            sys.exit(0)
        else:
            logger.error("Market data pipeline completed with failures for one or more symbols.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
