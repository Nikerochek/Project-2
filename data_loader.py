"""
Загрузка данных: FAOSTAT, USDA Market News.
Для демо — синтетические данные с сезонностью и трендом.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def fetch_faostat_synthetic(product: str = "wheat", years: int = 10) -> pd.DataFrame:
    """
    Синтетические данные объёмов продаж в стиле FAOSTAT.
    Источник: https://www.fao.org/faostat/
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=years * 12, freq="ME")
    np.random.seed(42)
    t = np.arange(len(dates))
    # Тренд + сезонность (годовая)
    trend = 100 + 2 * t + 0.01 * t ** 2
    season = 15 * np.sin(2 * np.pi * t / 12) + 5 * np.sin(2 * np.pi * t / 6)
    noise = np.random.randn(len(dates)) * 3
    sales = np.clip(trend + season + noise, 50, 200)
    return pd.DataFrame({"ds": dates, "y": sales})


def fetch_usda_synthetic(product: str = "milk", years: int = 10) -> pd.DataFrame:
    """
    Синтетические данные в стиле USDA Market News.
    Источник: https://www.ams.usda.gov/market-news
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=years * 12, freq="ME")
    np.random.seed(43)
    t = np.arange(len(dates))
    trend = 80 + 1.5 * t
    season = 12 * np.sin(2 * np.pi * t / 12 - np.pi / 4)  # сдвиг фазы для молока
    noise = np.random.randn(len(dates)) * 2
    sales = np.clip(trend + season + noise, 40, 150)
    return pd.DataFrame({"ds": dates, "y": sales})


def load_dataset(product: str = "wheat") -> pd.DataFrame:
    """
    Загружает датасет для продукта.
    wheat, milk — синтетические; при наличии API — можно подключить FAOSTAT/USDA.
    """
    if product.lower() in ("wheat", "пшеница"):
        return fetch_faostat_synthetic("wheat", years=10)
    elif product.lower() in ("milk", "молоко"):
        return fetch_usda_synthetic("milk", years=10)
    else:
        return fetch_faostat_synthetic(product, years=10)
