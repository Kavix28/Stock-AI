"""
Stock-AI: Multimodal AI for Stock Trend Prediction.
"""

__version__ = "0.2.0"
__author__ = "Stock-AI Research Contributors"

from stock_ai.logger import get_logger
from stock_ai.data.window_builder import build_chronological_windows
from stock_ai.features.indicators import compute_indicators
from stock_ai.inference.engine import InferenceEngine

__all__ = [
    "get_logger",
    "build_chronological_windows",
    "compute_indicators",
    "InferenceEngine"
]
