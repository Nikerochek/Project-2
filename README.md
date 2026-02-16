# Прогноз спроса на сельхозпродукцию

Студенческий проект: прогноз объёма продаж (пшеница, молоко) на основе открытых рыночных данных.

## Задача

- **Модели:** ARIMA (baseline), Prophet
- **Данные:** FAOSTAT, USDA Market News (для демо — синтетические с сезонностью и трендом)
- **Метрики:** MAPE, SMAPE
- **Учёт:** сезонность, тренды, сравнение с baseline

## Установка

```bash
pip install -r requirements.txt
```

> Prophet может требовать дополнительных зависимостей на Windows. При ошибке установки скрипт использует fallback-модель.

## Запуск

```bash
python main.py
```

Скрипт:
1. Загружает данные (wheat по умолчанию)
2. Обучает ARIMA и Prophet
3. Выводит MAPE и SMAPE для обеих моделей
4. Сохраняет графики: `forecast_comparison.png`, `metrics_comparison.png`

## Структура

```
demand-forecast/
├── main.py
├── data_loader.py   # FAOSTAT, USDA
├── models.py        # ARIMA, Prophet
├── evaluate.py      # MAPE, SMAPE
├── visualize.py
└── requirements.txt
```

## Источники данных

- **FAOSTAT:** https://www.fao.org/faostat/
- **USDA Market News:** https://www.ams.usda.gov/market-news
