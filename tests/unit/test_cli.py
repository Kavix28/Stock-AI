import pytest
from unittest.mock import patch, MagicMock
from stock_ai.data.market.download import main


def test_cli_dry_run():
    test_args = ["download.py", "--dry-run", "--symbols", "AAPL,NVDA"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
