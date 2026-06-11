# House Price Project

Воспроизводимый ML-проект для соревнования Kaggle
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Быстрый запуск

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m house_price.train
python -m house_price.predict
python -m pytest
```

## Установка зависимостей

Все библиотеки проекта перечислены в одном файле `requirements.txt`. Для установки
выполните:

```powershell
python -m pip install -r requirements.txt
```

Флаг `-r` говорит `pip` прочитать список библиотек из указанного файла. Папка
`src` настроена как источник Python-кода, поэтому пакет `house_price` доступен
в PyCharm и при запуске тестов без добавления локального пути в `requirements.txt`.

Исходные файлы должны находиться по путям:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

CSV-файлы исключены из Git, потому что это локальные исходные данные. Пути и параметры
проекта задаются централизованно в `src/house_price/config.py`.

## Структура

```text
House_price_project/
├── baseline/           # исследовательский notebook
├── data/               # локальные данные Kaggle
├── src/house_price/    # пакет обучения и предсказания
├── tests/              # автоматические тесты
├── artifacts/          # модель и отчет обучения
├── submissions/        # предсказания для Kaggle
├── .github/workflows/  # CI-проверки
├── pyproject.toml      # описание пакета и настройки инструментов
└── requirements.txt    # все библиотеки проекта
```

## Как устроен pipeline

1. `load_training_data()` читает данные и проверяет обязательные колонки.
2. `HouseFeatureEngineer` создает предметные признаки одинаково для train и test.
3. `build_pipeline()` объединяет feature engineering, заполнение пропусков, кодирование
   категорий и модель `HistGradientBoostingRegressor`.
4. Обучение оценивается по RMSLE с воспроизводимой кросс-валидацией.
5. Финальная модель и отчет сохраняются в `artifacts/`.
6. `python -m house_price.predict` создает `submissions/submission.csv`.

Служебные каталоги (`.venv`, кэши, настройки IDE, артефакты и submissions) не попадают
в Git. Пустые каталоги для результатов сохраняются с помощью `.gitkeep`.
