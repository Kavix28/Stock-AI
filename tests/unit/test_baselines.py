"""Unit tests for baseline models."""
import unittest
import numpy as np
from stock_ai.models.baselines import (
    MajorityBaseline,
    DirectionPersistenceBaseline,
    LogisticRegressionBaseline
)


class TestBaselines(unittest.TestCase):
    def test_majority_baseline(self):
        y_train = np.array([1, 1, 0, 1, 0])
        model = MajorityBaseline().fit(y_train)
        preds = model.predict(np.zeros((3, 5)))
        np.testing.assert_array_equal(preds, np.array([1, 1, 1]))

    def test_direction_persistence(self):
        persist = DirectionPersistenceBaseline()
        raw_lag1 = np.array([0, 1, 1, 0])
        preds = persist.predict(raw_lag1)
        np.testing.assert_array_equal(preds, raw_lag1)

    def test_logistic_regression(self):
        X = np.array([[1.0, 2.0], [1.1, 2.1], [-1.0, -2.0], [-1.2, -1.9]])
        y = np.array([1, 1, 0, 0])
        lr = LogisticRegressionBaseline().fit(X, y)
        preds = lr.predict(np.array([[2.0, 2.0], [-2.0, -2.0]]))
        np.testing.assert_array_equal(preds, np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
