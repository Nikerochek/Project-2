"""
Модели: ARIMA (baseline), Prophet.
Учёт сезонности и трендов.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Prophet — опционально (может не ставиться на Windows)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def train_arima(series: pd.Series, order: tuple = (2, 1, 2), seasonal_order: tuple = (1, 1, 1, 12)) -> tuple:
    """
    ARIMA/SARIMA — baseline модель.
    Возвращает прогноз и тестовые значения.
    """
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    steps = len(test)

    # Пробуем SARIMA (сезонность)
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=steps)
        return np.array(forecast), test.values
    except Exception:
        pass

    # Пробуем простой ARIMA
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        return np.array(forecast), test.values
    except Exception:
        pass

    # Fallback: наивный прогноз
    return np.full(steps, train.iloc[-1]), test.values


def train_prophet(df: pd.DataFrame, test_size: int) -> tuple:
    """
    Prophet — учитывает тренд и сезонность.
    df должен иметь колонки ds (datetime), y (значение).
    """
    if not PROPHET_AVAILABLE:
        # Fallback: сезонная декомпозиция (среднее по месяцам) + линейный тренд
        y = df["y"].values
        train, test = y[:-test_size], y[-test_size:]
        n = len(test)
        if len(train) >= 24:
            # Сезонная компонента: среднее отклонение по месяцу
            seasonal = np.array([np.mean(train[i::12]) for i in range(12)]) - np.mean(train)
            # Тренд: линейная регрессия по всему train
            x_tr = np.arange(len(train))
            coef = np.polyfit(x_tr, train, 1)
            pred = []
            for j in range(n):
                trend_val = np.polyval(coef, len(train) + j)
                pred.append(trend_val + seasonal[j % 12])
            pred = np.array(pred)
        else:
            pred = np.full(n, np.mean(train))
        return pred[:n], test[:n]

    train_df = df.iloc[:-test_size]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)

    future = pd.DataFrame({"ds": df["ds"].iloc[-test_size:]})
    forecast = model.predict(future)
    return forecast["yhat"].values, df["y"].iloc[-test_size:].values


def forecast_both(df: pd.DataFrame, test_months: int = 24) -> dict:
    """
    Обучает ARIMA и Prophet, возвращает прогнозы и тестовые значения.
    """
    series = df.set_index("ds")["y"]
    test_size = min(test_months, len(df) // 5)

    arima_pred, y_test = train_arima(series, order=(2, 1, 2))
    # Обрезаем до одинаковой длины
    n = min(len(arima_pred), len(y_test), test_size)
    arima_pred, y_test = arima_pred[:n], y_test[:n]

    prophet_pred, y_test_p = train_prophet(df, test_size=n)
    prophet_pred = prophet_pred[:n]
    y_test_p = y_test_p[:n]

    # Выравниваем
    n = min(len(arima_pred), len(prophet_pred), len(y_test))
    return {
        "y_true": y_test[:n],
        "arima": arima_pred[:n],
        "prophet": prophet_pred[:n],
    }
