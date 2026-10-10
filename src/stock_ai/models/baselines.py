"""
Benchmark Baselines for Stock Trend Prediction.

1. MajorityBaseline: Predicts the empirical majority class from training data.
2. DirectionPersistenceBaseline: Assumes direction t matches direction t-1.
3. LogisticRegressionEndpoint: Scikit-learn Logistic Regression using t-1 features.
4. LogisticRegressionFlat: Scikit-learn Logistic Regression using flattened 20-day window.
"""

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression


class MajorityBaseline:
    def __init__(self):
        self.majority_class = 1

    def fit(self, y_train: np.ndarray) -> "MajorityBaseline":
        self.majority_class = int(np.bincount(y_train).argmax())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class, dtype=int)


class DirectionPersistenceBaseline:
    """Predicts next-day direction from previous-day direction."""
    def predict(self, persist_preds: np.ndarray) -> np.ndarray:
        return persist_preds.astype(int)


class LogisticRegressionBaseline:
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
