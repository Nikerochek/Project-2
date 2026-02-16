"""
Метрики: MAPE, SMAPE.
"""

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)"""
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)"""
    eps = 1e-8
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(numerator / denominator) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"MAPE": mape(y_true, y_pred), "SMAPE": smape(y_true, y_pred)}
