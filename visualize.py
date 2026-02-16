"""
Визуализация: сравнение ARIMA vs Prophet.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_forecast_comparison(y_true, arima_pred, prophet_pred, product: str = "wheat"):
    """График: фактический ряд vs прогнозы ARIMA и Prophet."""
    n = len(y_true)
    x = np.arange(n)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true, "o-", label="Факт", color="black", markersize=4)
    plt.plot(x, arima_pred, "s--", label="ARIMA (baseline)", color="steelblue", markersize=4)
    plt.plot(x, prophet_pred, "^--", label="Prophet", color="coral", markersize=4)
    plt.xlabel("Месяц (тестовая выборка)")
    plt.ylabel("Объём продаж")
    plt.title(f"Прогноз спроса: {product} — сравнение с baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("forecast_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Сохранено: forecast_comparison.png")


def plot_metrics_bar(metrics: dict):
    """Столбчатая диаграмма MAPE и SMAPE по моделям."""
    models = list(metrics.keys())
    mape_vals = [metrics[m]["MAPE"] for m in models]
    smape_vals = [metrics[m]["SMAPE"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, mape_vals, width, label="MAPE (%)", color="steelblue")
    bars2 = ax.bar(x + width / 2, smape_vals, width, label="SMAPE (%)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("%")
    ax.set_title("Сравнение метрик: ARIMA vs Prophet")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Сохранено: metrics_comparison.png")
