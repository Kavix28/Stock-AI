try:
    from stock_ai.models.news.finbert import FinBertNewsEncoder
except ImportError:
    from src.stock_ai.models.news.finbert import FinBertNewsEncoder

__all__ = ["FinBertNewsEncoder"]
