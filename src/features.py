"""Явный набор признаков для моделей прогноза зарплаты.

Список зафиксирован по результатам ``notebooks/eda_salary.ipynb`` и
``notebooks/hypothesis_testing.ipynb``. В отличие от старого подхода
(«взять все колонки и удалить таргет»), здесь перечислено только то, что
разрешено подавать в модель. Всё остальное по умолчанию не попадает в X.

Правила:

* идентификаторы (``job_id``, ``url``, ``apply_url``) не являются признаками;
* поля, из которых напрямую считается таргет (``salary_min``, ``salary_max``,
  ``salary_mid``, ``log_salary``, ``salary_*_annual``, ``salary_text``,
  ``spread``, ``spread_rel``), запрещены;
* производные от зарплаты флаги (``outlier_*``, ``suspicious_period``)
  тоже запрещены: они содержат информацию о целевой переменной;
* ``description`` не входит в базовый набор и кодируется отдельным
  TF-IDF-пайплайном;
* ``company_id`` передаётся в модель только через out-of-fold target encoding;
* высококардинальные ``title``, ``company``, ``location``, ``city``,
  ``region`` заменены инженерными признаками: уровнем и семейством из
  ``title``, штатом из ``location``.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.preprocessing import (
    SALARY_MENTION,
    family_from_title,
    seniority_from_title,
    us_state_from_location,
)

RANDOM_STATE = 42

#: Единая метка пропуска для категориальных признаков и target encoding.
MISSING_CATEGORY = "__missing__"

TARGET_LOG = "log_salary"
TARGET_USD = "salary_mid"

#: Идентификаторы и служебные ссылки — не содержательные признаки зарплаты.
ID_COLUMNS = ["job_id", "url", "apply_url"]

#: Текст описания обрабатывается отдельным TF-IDF-пайплайном.
TEXT_COLUMN = "description"
TEXT_MASKED_COLUMN = "description_masked"

#: Признаки, которые прямо раскрывают таргет или получены из него.
LEAKAGE_COLUMNS = [
    "salary_min",
    "salary_max",
    "salary_mid",
    "log_salary",
    "log_salary_min",
    "log_salary_max",
    "salary_min_annual",
    "salary_max_annual",
    "salary_text",
    "spread",
    "spread_rel",
    "outlier_any",
    "outlier_iqr_salary_min",
    "outlier_iqr_salary_max",
    "outlier_z_salary_min",
    "outlier_z_salary_max",
    "salary_min_winsor",
    "salary_max_winsor",
    "suspicious_period",
    "invalid_salary_min",
    "invalid_salary_max",
    "invalid_salary_range",
]

#: Колонки, которые сознательно не подаются в базовую модель.
#: ``salary_source`` оставлен для аудита качества данных, даты на недельном
#: снимке бесполезны, а сырые высококардинальные поля заменены эвристиками.
EXCLUDED_COLUMNS = sorted(
    set(
        ID_COLUMNS
        + LEAKAGE_COLUMNS
        + [
            TEXT_COLUMN,
            TEXT_MASKED_COLUMN,
            "title",
            "company",
            "location",
            "region",
            "city",
            "language",
            "salary_currency",
            "salary_period",
            "salary_source",
            "posted_at",
            "first_seen_at",
            "posted_month",
            "posted_date",
            "posted_weekday",
            "posting_lag_days",
        ]
    )
)

#: Категориальные признаки базовой модели.
CATEGORICAL_FEATURES = [
    "raw_period",
    "ats",
    "country_code",
    "employment_type",
    "workplace_type",
    "seniority",
    "seniority_title",
    "family",
    "department",
    "us_state",
]

#: Числовые признаки базовой модели. Это длина описания (метаданные текста,
#: а не сам текст) и индикаторы пропусков, которые оказались значимыми.
NUMERIC_FEATURES = [
    "description_len",
    "is_missing_workplace_type",
    "is_missing_seniority",
    "is_missing_language",
    "is_missing_country_code",
    "is_missing_employment_type",
    "is_missing_department",
    "is_missing_region",
    "is_missing_city",
]

BASE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#: Признаки, кодируемые target encoding внутри train-фолда.
TARGET_ENCODING_FEATURES = ["company_id"]

#: Колонки, по которым считаются индикаторы пропусков.
MISSING_FLAG_COLUMNS = [
    "workplace_type",
    "seniority",
    "language",
    "country_code",
    "employment_type",
    "department",
    "region",
    "city",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки к копии датафрейма.

    Функция не меняет входной объект. Все колонки, нужные базовой модели,
    появляются здесь: ``seniority_title``, ``family``, ``us_state``,
    ``description_len``, ``description_masked`` и ``is_missing_*``.
    """
    result = df.copy()

    if "title" in result:
        derived_seniority = result["title"].map(seniority_from_title)
        derived_family = result["title"].map(family_from_title)
        if "seniority_title" in result:
            result["seniority_title"] = result["seniority_title"].fillna(derived_seniority)
        else:
            result["seniority_title"] = derived_seniority
        if "family" in result:
            result["family"] = result["family"].fillna(derived_family)
        else:
            result["family"] = derived_family

    if "location" in result or "region" in result:
        derived_state = result.apply(us_state_from_location, axis=1)
        if "us_state" in result:
            result["us_state"] = result["us_state"].fillna(derived_state)
        else:
            result["us_state"] = derived_state

    if TEXT_COLUMN in result:
        result[TEXT_COLUMN] = result[TEXT_COLUMN].fillna("")
        derived_len = result[TEXT_COLUMN].str.len()
        if "description_len" in result:
            result["description_len"] = result["description_len"].fillna(derived_len)
        else:
            result["description_len"] = derived_len
        result[TEXT_MASKED_COLUMN] = result[TEXT_COLUMN].str.replace(
            SALARY_MENTION, " [salary] ", regex=True
        )

    for column in MISSING_FLAG_COLUMNS:
        if column in result:
            result[f"is_missing_{column}"] = result[column].isna()

    return result


def feature_frame(
    df: pd.DataFrame, *, include_categorical: bool = True, include_te: bool = False
) -> pd.DataFrame:
    """Возвращает X только из разрешённых признаков.

    Parameters
    ----------
    df:
        Подготовленный датафрейм (после ``prepare_salary_data`` и
        ``add_engineered_features``).
    include_categorical:
        Если ``False``, возвращается только числовая часть (полезно для
        быстрых проверок).
    include_te:
        Добавить ``company_id`` для последующего target encoding.
    """
    columns = (CATEGORICAL_FEATURES if include_categorical else []) + NUMERIC_FEATURES
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"В датафрейме нет признаков: {missing}")

    features = df[columns].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = as_category(features[column])
    if include_te:
        features[TARGET_ENCODING_FEATURES[0]] = as_category(
            df[TARGET_ENCODING_FEATURES[0]]
        )
    return features


def as_category(series: pd.Series) -> pd.Series:
    """Приводит категориальный признак к строкам с единой меткой пропуска.

    Так OneHotEncoder не зависит от того, был ли NaN в обучающей выборке, и
    одинаково обрабатывает пропуски в train, validation и inference.
    """
    return (
        series.astype("object")
        .where(series.notna(), MISSING_CATEGORY)
        .astype(str)
    )


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Проверяет, что в списке признаков нет id, текста и утечек таргета."""
    columns = list(columns)
    forbidden = sorted(set(columns) & set(EXCLUDED_COLUMNS))
    if forbidden:
        raise ValueError(
            "В признаки попали запрещённые колонки (id/текст/утечка таргета): "
            f"{forbidden}"
        )


def describe_feature_policy() -> pd.DataFrame:
    """Возвращает таблицу «признак → роль» для README и ноутбуков."""
    rows = []
    for column in CATEGORICAL_FEATURES:
        rows.append({"признак": column, "роль": "категориальный", "источник": "EDA / hypothesis_testing"})
    for column in NUMERIC_FEATURES:
        rows.append({"признак": column, "роль": "числовой", "источник": "EDA / hypothesis_testing"})
    rows.append({"признак": "company_id", "роль": "target encoding (OOF)", "источник": "гипотеза 13"})
    rows.append({"признак": "description", "роль": "отдельный TF-IDF", "источник": "гипотеза 18"})
    return pd.DataFrame(rows)
