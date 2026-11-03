# Job Postings with Salaries

Учебный исследовательский проект: анализ вакансий с зарплатными вилками, проверка гипотез о том, что связано с зарплатой, и модель прогноза зарплаты по описанию вакансии.

Два основных ноутбука:

- [`notebooks/hypothesis_testing.ipynb`](notebooks/hypothesis_testing.ipynb) — подготовка данных, 21 гипотеза, размеры эффектов, множественные проверки и выводы о признаках;
- [`notebooks/salary_prediction.ipynb`](notebooks/salary_prediction.ipynb) — явный набор признаков, сравнение моделей на двух схемах валидации, отдельный текстовый пайплайн, сохранение модели и пример предсказания.

## Документация

Подробная методичка для начинающих — в PDF: **[docs/hypothesis_testing_guide.pdf](docs/hypothesis_testing_guide.pdf)**. Исходник — [`docs/hypothesis_testing_guide.html`](docs/hypothesis_testing_guide.html).

Она объясняет, как устроены проверки гипотез, зачем нужны логарифм таргета, середина вилки, Kruskal–Wallis, кросс-валидация и target encoding, что такое p-value, размер эффекта, множественные проверки и утечка данных.

## Быстрый старт

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
jupyter lab
```

Данные должны лежать по пути `data/raw/job_postings_with_salaries.csv`. Путь вычисляется в `src/utils.py` относительно корня проекта.

Ноутбуки рассчитаны на запуск из каталога `notebooks/`; корень проекта они добавляют в `sys.path` автоматически. Выполняйте **Kernel → Restart & Run All**.

## Структура проекта

```
.
├── data/raw/job_postings_with_salaries.csv   # исходные данные (~60 МБ, не в репозитории)
├── notebooks/
│   ├── eda_salary.ipynb                      # разведочный анализ
│   ├── hypothesis_testing.ipynb              # 21 гипотеза
│   └── salary_prediction.ipynb               # модель, валидация, сохранение артефакта
├── src/
│   ├── utils.py                              # загрузка данных
│   ├── preprocessing.py                      # валюта/период → USD/год, валидация, эвристики
│   ├── features.py                           # явный список признаков и запретов
│   ├── modeling.py                           # OOF target encoding, модели, метрики, SalaryModel
│   └── predict.py                            # подготовка одной записи и CLI-прогноз
├── tests/
│   ├── test_preprocessing.py                 # валидация зарплат, FX, эвристики
│   ├── test_features.py                      # отсутствие утечек, политика признаков
│   └── test_modeling.py                      # OOF TE, модели, сохранение/загрузка, инференс
├── models/salary_model.joblib                # создаётся ноутбуком salary_prediction.ipynb
├── docs/
│   ├── hypothesis_testing_guide.pdf
│   └── hypothesis_testing_guide.html
├── .github/workflows/tests.yml               # pytest в CI
├── requirements.txt
├── README.md
└── .gitignore
```

## Признаки модели

Список зафиксирован в [`src/features.py`](src/features.py) по результатам EDA и проверки гипотез. Модель **не** получает «все колонки минус таргет»: в X попадают только разрешённые признаки.

**Категориальные (10):**

`raw_period`, `ats`, `country_code`, `employment_type`, `workplace_type`, `seniority`, `seniority_title`, `family`, `department`, `us_state`.

**Числовые (9):**

`description_len` и индикаторы пропусков `is_missing_workplace_type`, `is_missing_seniority`, `is_missing_language`, `is_missing_country_code`, `is_missing_employment_type`, `is_missing_department`, `is_missing_region`, `is_missing_city`.

**Target encoding:**

`company_id` — только через out-of-fold кодирование внутри train-фолда.

**Текст:**

`description` не входит в базовую табличную модель. Для него собирается отдельный `TfidfVectorizer` + Ridge; в варианте «метаданные + текст» оба блока объединяются через `ColumnTransformer`.

**Исключены полностью:**

- идентификаторы `job_id`, `url`, `apply_url`;
- таргет и его производные: `salary_min`, `salary_max`, `salary_mid`, `log_salary`, `salary_min_annual`, `salary_max_annual`, `salary_text`, `spread`, `spread_rel`;
- флаги, посчитанные по зарплате: `outlier_*`, `suspicious_period`, `invalid_*`, winsorized-колонки;
- сырые высококардинальные `title`, `company`, `location`, `region`, `city`, `language`;
- `salary_source` оставлен только для аудита качества данных, а не как рыночный признак;
- даты `posted_at`, `first_seen_at` и производные: на недельном снимке они бесполезны.

## Таргет и обратное преобразование

Таргет — `log_salary = log(salary_mid)`, где `salary_mid` — середина вилки в USD за год.

Модель минимизирует MAE в лог-шкале. Поэтому:

- `MAE (log)` — основная метрика сравнения; `exp(MAE_log) − 1` примерно равна средней относительной ошибке;
- `exp(pred_log)` — оценка **условной медианы / типичного значения**, а не гарантированное условное среднее;
- Duan's smearing (`mean(exp(residual))` на train) сдвигает оценку к среднему, но и он не устраняет смещение выбора модели и данных;
- `MAE (USD)` считается после обратного преобразования и поэтому чувствительна к дорогим хвостам.

В отчёте приводятся обе шкалы: лог-шкала — для честного сравнения моделей, USD — для практической интерпретации.

## Схемы валидации

1. **Обычный `train_test_split`** (20 % тест, `random_state=42`) — оценка «в знакомых компаниях»: компания может встретиться и в train, и в test.
2. **`GroupShuffleSplit` по `company_id`** — оценка обобщения на **новые** компании; пересечения групп нет (проверяется в ноутбуке).
3. **5-fold `GroupKFold` по `company_id`** для линейной модели — проверка стабильности оценки на нескольких фолдах.

`company_id` кодируется только через `OOFTargetEncoder`: для каждой train-строки статистика строится по другим фолдам, а для validation/test — по полному train-фолду. Это устраняет утечку таргета, которая возникает при обычном target encoding «на всём train».

## Результаты моделей

Случайный split, 7 324 train / 1 831 test:

| Модель | MAE (log) | MAE (USD) | MAPE |
|---|---:|---:|---:|
| DummyRegressor (median) | 0,5023 | 54 424 | 57,9 % |
| LinearRegression | 0,2788 | 33 690 | 32,8 % |
| RandomForestRegressor | 0,2826 | 33 305 | 32,3 % |
| **HistGradientBoostingRegressor** | **0,2737** | **32 279** | **31,0 %** |

Group split по `company_id`, 7 367 train / 1 788 test:

| Модель | MAE (log) | MAE (USD) |
|---|---:|---:|
| DummyRegressor (median) | 0,523 | 55 408 |
| LinearRegression | 0,318 | 35 299 |
| RandomForestRegressor | 0,309 | 34 483 |
| **HistGradientBoostingRegressor** | **0,306** | **34 143** |

`GroupKFold`, LinearRegression: MAE (log) 0,312, MAE (USD) 35 624.

Вывод: на новых компаниях качество ожидаемо хуже. Разрыв между random и group split — цена частичного запоминания работодателей через `company_id`; именно group-оценка ближе к сценарию «новая вакансия новой компании».

### Текст и обратное преобразование

| Вариант | MAE (log) | MAE (USD) | Δ MAE (log) к метаданным |
|---|---:|---:|---:|
| Метаданные (LinearRegression) | 0,279 | 33 690 | — |
| Только TF-IDF | 0,252 | 29 407 | −9,5 % |
| Метаданные + TF-IDF | **0,222** | **26 478** | **−20,2 %** |

Текст — самый сильный источник сигнала, но он требует `description`; поэтому сохраняется именно табличная модель, а текстовый вариант остаётся отдельным экспериментом.

Для лучшей табличной модели (HistGradientBoosting):

| Обратное преобразование | Среднее, USD | Медиана, USD | MAE, USD |
|---|---:|---:|---:|
| `exp(pred)` | 111 577 | 105 119 | 31 813 |
| `exp(pred) * smearing` | 116 253 | 109 525 | 32 279 |
| Факт | 116 743 | 99 665 | — |

`exp(pred)` занижает среднее, потому что это медиана; smearing приближает среднее, но немного увеличивает MAE в USD. Это не ошибка, а разные целевые функционалы.

## Интеграция модели

После выполнения `notebooks/salary_prediction.ipynb` создаётся `models/salary_model.joblib` — пайплайн вместе со списком признаков, smearing-коэффициентом и метаданными эксперимента.

### CLI

```bash
python -m src.predict \
  --title "Senior Data Scientist" \
  --company-id acme \
  --country-code US \
  --raw-period year \
  --ats greenhouse \
  --employment-type full-time \
  --workplace-type remote \
  --location "New York, NY"
```

Ответ — JSON с полями `salary_pred_log`, `salary_pred_usd` (медиана/типичное значение), `salary_pred_usd_conditional_mean` (со smearing при флаге `--conditional-mean`) и пояснением.

Для пакетного прогноза можно передать JSON-файл с объектом или списком объектов:

```bash
python -m src.predict --input records.json
```

### Python API

```python
from src.modeling import load_model
from src.predict import predict_salary

bundle = load_model("models/salary_model.joblib")

record = {
    "title": "Senior Data Scientist",
    "company_id": "new-company",
    "ats": "greenhouse",
    "country_code": "US",
    "employment_type": "full-time",
    "workplace_type": "remote",
    "seniority": None,
    "department": "Data",
    "location": "New York, NY",
    "raw_period": "year",
    "description": "Build ML models and dashboards.",
}

result = predict_salary(record, model=bundle)
print(result["salary_pred_usd"])
```

`company_id` новой компании получает prior train через target encoding; `title` обязателен, потому что из него извлекаются `seniority_title` и `family`. Зарплатные поля записи игнорируются: модель не должна видеть таргет.

## Гипотезы и множественные проверки

В `hypothesis_testing.ipynb` проверяются 21 гипотеза. Чтобы не смешивать статистическую значимость с размером эффекта:

- для парных post-hoc сравнений внутри одной гипотезы используется поправка **Holm** (контроль FWER, равномерно мощнее Бонферрони);
- 21 гипотеза разбита на шесть содержательных семейств: таргет и структура вилки, роль и уровень, география и формат, компания и платформа, занятость и период, текст/пропуски/даты. Внутри каждого семейства также применяется **Holm**;
- дополнительно по всем 21 гипотезе приводится **Benjamini–Hochberg (FDR)**: исследование разведочное, и жёсткий FWER по всем проверкам слишком консервативен.

Почему не «Bonferroni на каждый вызов»: поправка внутри отдельного `pairwise_tests` не контролирует ошибку по всему семейству гипотез, а Bonferroni по всем 21 проверке сильно теряет мощность. Итоговая таблица с `p_Holm_в_семействе` и `p_BH_по_21` строится в конце ноутбука.

При n ≈ 9 000 почти любое различие значимо. Поэтому в выводах главное — медианы, ε², Cramér's V и прирост MAE, а p-value используется как ориентир.

### Что показали данные

- Логарифм убирает скошенность: skew 3,56 → −0,08, RMSLE 0,55 → 0,45.
- Самые полезные табличные признаки: `raw_period` (MAE −16,4 %), target encoding компании `company` + `company_id` (−12,1 %; в финальной модели используется один `company_id`), `family` (−7,5 %), `seniority_title` (−10,5 %), страна (−4,7 %).
- `description` — сильнейший источник сигнала (MAE −20,2 % в комбинации с метаданными), но 88 % описаний содержат суммы; после маскирования сумм качество почти не падает, значит модель использует контекст, а не готовые цифры.
- `salary_min` почти идеально предсказывает `salary_max` (R² 0,92), но это утечка: в момент прогноза для новой вакансии min неизвестен.
- Пропуски — не «среднее»: пропуск `country_code`, `seniority`, `employment_type` образует отдельный сегмент рынка. Поэтому NaN кодируется отдельной категорией и индикаторами.
- Формат работы после контроля на состав выборки даёт лишь +6,9 % (remote) и +8,4 % (hybrid) к onsite, а не сырые +45 %.
- Даты на недельном снимке бесполезны: сезонность проверить нельзя.

## Курсы валют

Зарплаты приводятся к USD по фиксированным курсам **на 2026-10-15**. Источник — официальные курсы Национального банка Республики Беларусь (НБРБ), https://www.nbrb.by/, кросс-курсы через BYN. Направление: **единиц валюты за 1 USD**; пересчёт в коде — `salary_usd = salary * period_coef / CURRENCY_PER_USD`.

Курсы не обновляются автоматически. Конвертация в USD **не** убирает различия в налогах, стоимости жизни, паритете покупательной способности и структуре вакансий между странами. Межстрановые сравнения зарплат — приблизительные.

## Ограничения

- Только вакансии с указанной зарплатой — смещение выборки.
- Снимок покрывает около недели и один месяц: сезонность, тренды и динамика не проверяются.
- Курсы валют зафиксированы на одну дату; налоги и стоимость жизни не учитываются.
- `seniority` заполнен примерно у 7 %, `department` — у 59 %; уровень, семейство профессии и штат извлечены эвристиками из `title`/`location`, метки шумные. Например, `Senior Director` может попасть в `executive`, а `Lead Data Scientist` — в `senior`; это приближение, а не ручная разметка.
- Baseline в ноутбуке гипотез — Ridge на one-hot и TF-IDF; бустинг в модели прогноза даёт другие абсолютные значения MAE, но направления сравнений сохраняются.
- Все выводы — ассоциативные. Причинность по наблюдательным данным установить нельзя, особенно для формата работы и `ats`.

## Тесты

```bash
pytest -q
```

Покрыты: нулевые/отрицательные зарплаты, `salary_min > salary_max`, неизвестная валюта/период, пропуски и нечисловые значения, неизменность входного датафрейма, конечность выходов, отсутствие утечек в признаках, OOF target encoding, сохранение/загрузка модели и инференс одной записи.

CI: [`.github/workflows/tests.yml`](.github/workflows/tests.yml) запускает `pytest -q` на Python 3.13.

## Требования

Python 3.13. Зависимости зафиксированы в [`requirements.txt`](requirements.txt): `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `joblib`, `jupyterlab`, `nbconvert`, `ipykernel`, `pytest`. GPU и доступ к сети не нужны.
