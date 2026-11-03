"""Подготовка данных для анализа и моделей зарплат.

Единый модуль, который нормализует исходные данные о зарплатных вилках:
- переводит суммы из исходной валюты и за исходный период в USD/год;
- вычисляет целевую переменную (середину вилки) и её логарифм;
- хранит эвристики восстановления уровня и семейства профессии из `title`;
- содержит признаки для поиска выбросов.

Все ноутбуки проекта должны использовать этот модуль, а не дублировать
константы и расчёты локально — так правки в нормализации не рассинхронизируются.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: Стандартное число рабочих единиц в году (американский рынок труда).
PERIOD_COEF = {
    "day": 260,
    "week": 52,
    "month": 12,
    "hour": 2080,
    "year": 1,
}

#: Источник и дата фиксации курсов.
#:
#: Значения — это количество единиц валюты за 1 USD (USD per unit — обратная
#: величина). Поэтому сумма в валюте переводится в USD делением на
#: коэффициент: ``salary_usd = salary * PERIOD_COEF / CURRENCY_PER_USD``.
#: Курсы статические, зафиксированы на 2026-10-15 по данным Национального
#: банка Республики Беларусь (https://www.nbrb.by/) и пересчитаны в кросс-курсы
#: через BYN. При обновлении датасета курсы нужно перепроверить.
FX_RATES_AS_OF = "2026-10-15"
FX_RATES_SOURCE = "Национальный банк Республики Беларусь (НБРБ), https://www.nbrb.by/"
FX_RATES_DIRECTION = "единиц валюты за 1 USD"

CURRENCY_PER_USD = {
    "USD": 1.0,
    "EUR": 0.89,
    "CAD": 1.42,
    "GBP": 0.75,
    "AUD": 1.45,
    "PLN": 3.9,
    "NZD": 1.78,
    "CHF": 0.83,
}

SENIORITY_PATTERNS = [
    ("executive", r"\b(head|director|vp|vice president|chief|cfo|ceo|cto|coo|president|founder|partner)\b"),
    ("internship", r"\b(intern|internship|trainee|praktikant)\b"),
    ("senior", r"\b(senior|sr\.?|lead|principal|staff|expert)\b"),
    ("entry", r"\b(entry[- ]level|junior|jr\.?|graduate|apprentice|associate)\b"),
]

FAMILY_PATTERNS = {
    "engineering": r"\b(engineer\w*|developer|programmer|sre|devops|architect\w*|qa|software|frontend|backend|full[- ]stack|mobile|ios|android)\b",
    "data_ai": r"\b(data scien\w*|machine learning|ml |ai |artificial intelligence|analyst|analytics|statistic\w*|research\w*)\b",
    "sales": r"\b(sales|account executive|account manager|business development|bdr|sdr|revenue)\b",
    "marketing": r"\b(marketing|growth|seo|content|brand|social media|communications|copywriter)\b",
    "finance": r"\b(finance|financial|accountant|accounting|controller|auditor|tax|treasury|investment|banking)\b",
    "healthcare": r"\b(nurse|clinical|medical|physician|therapist|pharmac\w*|health|patient|radiolog\w*|surgeon|dental)\b",
    "operations": r"\b(operations|logistics|supply chain|warehouse|production|manufactur\w*|procurement|driver|facilit\w*)\b",
    "hr": r"\b(recruit\w*|talent|human resources|people|hr |compensation|benefits)\b",
    "legal": r"\b(legal|counsel|attorney|lawyer|compliance|paralegal)\b",
    "support": r"\b(customer support|customer success|support specialist|help desk|service desk|call center)\b",
    "product": r"\b(product manager|product owner|project manager|program manager|scrum|product lead)\b",
    "design": r"\b(design\w*|ux|ui|graphic|creative|art director)\b",
    "education": r"\b(teacher|tutor|instructor|professor|education|curriculum)\b",
}

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
US_STATE_ABBRS = set(US_STATES.values())

SALARY_MENTION = re.compile(
    r"(?:[$€£]\s?\d[\d,.\s]*(?:k)?|\b\d[\d,.]*\s?(?:usd|eur|gbp|cad|dollars|euros)\b)",
    re.IGNORECASE,
)


def seniority_from_title(title):
    """Восстанавливает уровень позиции (executive / internship / senior / entry / mid).

    Правила применяются по порядку: ``executive`` → ``internship`` → ``senior``
    → ``entry``. Поэтому «Senior Director» и «Head of Analytics» получают
    ``executive``, а «Lead Data Scientist» и «Principal Engineer» — ``senior``.

    Известные ограничения: эвристика не понимает контекст («Director of
    Engineering» — executive, но «Assistant to the Director» тоже), не
    различает senior/lead/principal по силе, а заголовки без явного маркера
    попадают в ``mid``. Это грубая группировка для модели, а не замена
    ручной разметке ``seniority``.
    """
    text = str(title).lower()
    for label, pattern in SENIORITY_PATTERNS:
        if re.search(pattern, text):
            return label
    return "mid"


def family_from_title(title):
    """Классифицирует семейство профессии по ключевым словам в названии.

    Порядок словаря задаёт приоритет: при пересечении слов выигрывает более
    раннее семейство (например, «Data Engineer» → ``engineering``, а
    «Data Scientist» → ``data_ai``). Заголовки без совпадений попадают в
    ``other``; около 43 % строк датасета — известное ограничение эвристики.
    """
    text = str(title).lower()
    for label, pattern in FAMILY_PATTERNS.items():
        if re.search(pattern, text):
            return label
    return "other"


def us_state_from_location(row):
    """Извлекает двухбуквенный код штата США из location / region."""
    location = str(row["location"]) if "location" in row and pd.notna(row["location"]) else ""
    region = str(row["region"]) if "region" in row and pd.notna(row["region"]) else ""
    for text in (location, region):
        lowered = text.lower()
        for name, abbr in US_STATES.items():
            if name in lowered:
                return abbr
    for text in (location, region):
        match = re.search(r"(?:^|,\s*)([A-Z]{2})(?:\s+\d{5})?\s*$", text)
        if match and match.group(1) in US_STATE_ABBRS:
            return match.group(1)
    return None


def iqr_outlier(values):
    """Флаг выбросов по правилу Тьюки (IQR)."""
    values = pd.Series(values).astype(float)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def _clean_salary_column(df, column, *, strict):
    """Приводит границу вилки к числу и помечает некорректные значения."""
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")

    non_numeric = original.notna() & numeric.isna()
    non_positive = numeric.notna() & (~np.isfinite(numeric) | (numeric <= 0))
    invalid = non_numeric | non_positive

    if strict and invalid.any():
        bad = original.loc[invalid].head(5).tolist()
        raise ValueError(
            f"Некорректные значения в {column} (не число, NaN/inf или <= 0): {bad}"
        )

    return numeric.where(~invalid), invalid


def prepare_salary_data(df, *, strict=False):
    """Приводит зарплатные вилки к USD в год.

    Таргет — середина вилки ``salary_mid = (salary_min + salary_max) / 2``,
    в которой ``salary_min`` / ``salary_max`` уже пересчитаны в USD за год.

    Результат добавляет в копию входных данных:
    ``raw_period``, ``raw_currency``, ``salary_min``, ``salary_max``,
    ``salary_mid``, ``log_salary``, ``log_salary_min``, ``log_salary_max``,
    ``spread``, ``spread_rel`` и флаги некорректных значений.

    Правила обработки:

    * отсутствующие период/валюта/граница вилки остаются NaN — вызывающий
      код сам решает, удалять такие строки или нет;
    * неизвестные (не NaN) период или валюта всегда приводят к ``ValueError``;
    * нечисловые, нулевые и отрицательные значения по умолчанию становятся
      NaN и помечаются в ``invalid_salary_min`` / ``invalid_salary_max``
      (при ``strict=True`` — исключение);
    * если ``salary_min > salary_max``, обе границы обнуляются, а строка
      помечается ``invalid_salary_range`` (при ``strict=True`` — исключение);
    * входной DataFrame не изменяется.
    """
    result = df.copy()

    period_coef = result["salary_period"].map(PERIOD_COEF)
    currency_coef = result["salary_currency"].map(CURRENCY_PER_USD)

    unknown_periods = result.loc[
        result["salary_period"].notna() & period_coef.isna(), "salary_period"
    ].unique()
    if len(unknown_periods):
        raise ValueError(f"Неизвестные периоды оплаты: {sorted(unknown_periods)}")

    unknown_currencies = result.loc[
        result["salary_currency"].notna() & currency_coef.isna(), "salary_currency"
    ].unique()
    if len(unknown_currencies):
        raise ValueError(f"Неизвестные валюты: {sorted(unknown_currencies)}")

    result["raw_period"] = result["salary_period"]
    result["raw_currency"] = result["salary_currency"]

    salary_min, invalid_min = _clean_salary_column(result, "salary_min", strict=strict)
    salary_max, invalid_max = _clean_salary_column(result, "salary_max", strict=strict)

    invalid_range = salary_min.notna() & salary_max.notna() & (salary_min > salary_max)
    if strict and invalid_range.any():
        bad = result.loc[invalid_range, ["salary_min", "salary_max"]].head(5)
        raise ValueError(f"salary_min больше salary_max: {bad.to_dict('records')}")

    salary_min = salary_min.where(~invalid_range)
    salary_max = salary_max.where(~invalid_range)

    result["invalid_salary_min"] = invalid_min
    result["invalid_salary_max"] = invalid_max
    result["invalid_salary_range"] = invalid_range

    result["salary_min"] = salary_min * period_coef / currency_coef
    result["salary_max"] = salary_max * period_coef / currency_coef

    result["salary_mid"] = (result["salary_min"] + result["salary_max"]) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        result["log_salary"] = np.log(result["salary_mid"])
        result["log_salary_min"] = np.log(result["salary_min"])
        result["log_salary_max"] = np.log(result["salary_max"])

    result["spread"] = result["salary_max"] - result["salary_min"]
    with np.errstate(divide="ignore", invalid="ignore"):
        result["spread_rel"] = result["spread"] / result["salary_mid"]

    # Нормализация не должна порождать бесконечности: log/деление на ноль и
    # переполнение возвращаем в NaN, чтобы ошибка была явной и безопасной.
    for column in ["salary_min", "salary_max", "salary_mid", "log_salary",
                   "log_salary_min", "log_salary_max", "spread", "spread_rel"]:
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)

    return result
