"""
Прогноз спроса на сельхозпродукцию на основе открытых рыночных данных.
Студенческий проект: ARIMA baseline, Prophet, MAPE/SMAPE.
"""

from data_loader import load_dataset
from models import forecast_both, PROPHET_AVAILABLE
from evaluate import evaluate
from visualize import plot_forecast_comparison, plot_metrics_bar


def main():
    product = "wheat"
    print(f"Загрузка данных: {product} (FAOSTAT/USDA-style)...")
    df = load_dataset(product)
    print(f"Данных: {len(df)} месяцев")

    print("\nОбучение ARIMA (baseline) и Prophet...")
    if not PROPHET_AVAILABLE:
        print("  Prophet не установлен — используется fallback-модель")
    results = forecast_both(df, test_months=24)

    y_true = results["y_true"]
    arima_pred = results["arima"]
    prophet_pred = results["prophet"]

    metrics_arima = evaluate(y_true, arima_pred)
    metrics_prophet = evaluate(y_true, prophet_pred)
    metrics = {"ARIMA (baseline)": metrics_arima, "Prophet": metrics_prophet}

    print("\nМетрики:")
    for name, m in metrics.items():
        print(f"  {name}: MAPE={m['MAPE']:.2f}%, SMAPE={m['SMAPE']:.2f}%")

    print("\nВизуализация...")
    plot_forecast_comparison(y_true, arima_pred, prophet_pred, product)
    plot_metrics_bar(metrics)

    print("\nГотово. Результаты: forecast_comparison.png, metrics_comparison.png")


if __name__ == "__main__":
    main()
