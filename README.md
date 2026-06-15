# House Price Project

Воспроизводимый ML-проект для соревнования Kaggle
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Проект сравнивает несколько регрессионных моделей на одинаковых данных, преобразованиях,
CV-разбиениях и метрике RMSLE. После выбора модели полный pipeline обучается на всех
доступных данных и сохраняется для создания submission.

## Быстрый запуск

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m house_price.compare
python -m house_price.train
python -m house_price.predict

python -m pytest
python -m ruff check .
```

Команды создают локальные файлы:

```text
artifacts/model_comparison.json
artifacts/model.joblib
artifacts/training_report.json
submissions/submission.csv
```

Они исключены из Git.

## Данные

Исходные данные Kaggle должны находиться здесь:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

CSV-файлы также исключены из Git.

## Сравниваемые модели

- `HistGradientBoostingRegressor`
- `RandomForestRegressor`
- `XGBRegressor`

`python -m house_price.compare` оценивает все модели на одних и тех же пяти
воспроизводимых KFold-разбиениях. Для каждой модели сохраняются оценки фолдов,
средний RMSLE, стандартное отклонение и время оценки.

Параметры моделей находятся в `src/house_price/config.py`. Текущие значения являются
разумными стартовыми конфигурациями, а не доказанно лучшими гиперпараметрами.

Результаты текущего воспроизводимого сравнения:

| Модель | CV RMSLE mean | CV RMSLE std |
|---|---:|---:|
| HistGradientBoosting | 0.13398 | 0.01796 |
| Random Forest | 0.14194 | 0.01846 |
| XGBoost | **0.12947** | **0.01702** |

XGBoost выбран текущей моделью по умолчанию. Следующий отдельный эксперимент может
быть посвящён настройке его гиперпараметров без изменения протокола оценки.

## Архитектура

```text
src/house_price/
├── config.py       # пути и параметры моделей
├── data.py         # загрузка и проверка данных
├── features.py     # предметные признаки
├── models.py       # фабрика моделей
├── pipeline.py     # общий preprocessing и модель
├── evaluation.py   # единая CV-оценка
├── compare.py      # сравнение моделей
├── train.py        # обучение и сохранение финального pipeline
└── predict.py      # создание submission
```

Pipeline объединяет feature engineering, удаление `Id`, заполнение пропусков,
масштабирование числовых признаков, one-hot кодирование категорий и выбранную модель.
При обучении и предсказании используется один и тот же сохранённый pipeline.

## Git

В Git не добавляются `.venv`, настройки IDE, CSV, кэши, обученные модели и submissions.
Код проверяется тестами и Ruff локально и через GitHub Actions.
