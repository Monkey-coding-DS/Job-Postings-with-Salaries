"""Подготовка данных для анализа и моделей зарплат.

Единый модуль, который нормализует исходные данные о зарплатных вилках:
- переводит суммы из исходной валюты и за исходный период в USD/год;
- вычисляет целевую переменную (середину вилки) и её логарифм;
- хранит эвристики восстановления уровня и семейства профессии из `title`;
- содержит признаки для поиска выбросов.

Все ноутбуки проекта должны использовать этот модуль, а не дублировать
константы и расчёты локально — так правки в нормализации не рассинхронизируются.
"""

import re

import numpy as np
import pandas as pd

PERIOD_COEF = {
    "day": 260,
    "week": 52,
    "month": 12,
    "hour": 2080,
    "year": 1,
}

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
    "engineering": r"\b(engineer|developer|programmer|sre|devops|architect|qa|software|frontend|backend|full[- ]stack|mobile|ios|android)\b",
    "data_ai": r"\b(data scien|machine learning|ml |ai |artificial intelligence|analyst|analytics|statistic|research)\b",
    "sales": r"\b(sales|account executive|account manager|business development|bdr|sdr|revenue)\b",
    "marketing": r"\b(marketing|growth|seo|content|brand|social media|communications|copywriter)\b",
    "finance": r"\b(finance|financial|accountant|accounting|controller|auditor|tax|treasury|investment|banking)\b",
    "healthcare": r"\b(nurse|clinical|medical|physician|therapist|pharmac|health|patient|radiolog|surgeon|dental)\b",
    "operations": r"\b(operations|logistics|supply chain|warehouse|production|manufactur|procurement|driver|facilit)\b",
    "hr": r"\b(recruit|talent|human resources|people|hr |compensation|benefits)\b",
    "legal": r"\b(legal|counsel|attorney|lawyer|compliance|paralegal)\b",
    "support": r"\b(customer support|customer success|support specialist|help desk|service desk|call center)\b",
    "product": r"\b(product manager|product owner|project manager|program manager|scrum|product lead)\b",
    "design": r"\b(designer|ux|ui|graphic|creative|art director)\b",
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
    """Восстанавливает уровень позиции (executive / internship / senior / entry / mid) из названия."""
    text = str(title).lower()
    for label, pattern in SENIORITY_PATTERNS:
        if re.search(pattern, text):
            return label
    return "mid"


def family_from_title(title):
    """Классифицирует семейство профессии по ключевым словам в названии."""
    text = str(title).lower()
    for label, pattern in FAMILY_PATTERNS.items():
        if re.search(pattern, text):
            return label
    return "other"


def us_state_from_location(row):
    """Извлекает двухбуквенный код штата США из location / region."""
    location = str(row["location"]) if pd.notna(row["location"]) else ""
    region = str(row["region"]) if pd.notna(row["region"]) else ""
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


def prepare_salary_data(df):
    """Приводит зарплатные вилки к USD в год.

    Таргет — середина вилки `salary_mid = (salary_min + salary_max) / 2`,
    в которой `salary_min` / `salary_max` уже пересчитаны в USD за год.

    Результат добавляет в копию входных данных:
    `raw_period`, `raw_currency`, `salary_min`, `salary_max`, `salary_mid`,
    `log_salary`, `log_salary_min`, `log_salary_max`, `spread`, `spread_rel`.

    Строки с неполной вилкой (одна граница NaN) оставляются в датафрейме —
    вызывающий код решает, их исключать.
    """
    df = df.copy()

    period_coef = df["salary_period"].map(PERIOD_COEF)
    currency_coef = df["salary_currency"].map(CURRENCY_PER_USD)

    unknown_periods = df.loc[
        df["salary_period"].notna() & period_coef.isna(), "salary_period"
    ].unique()
    if len(unknown_periods):
        raise ValueError(f"Неизвестные периоды оплаты: {sorted(unknown_periods)}")

    unknown_currencies = df.loc[
        df["salary_currency"].notna() & currency_coef.isna(), "salary_currency"
    ].unique()
    if len(unknown_currencies):
        raise ValueError(f"Неизвестные валюты: {sorted(unknown_currencies)}")

    df["raw_period"] = df["salary_period"]
    df["raw_currency"] = df["salary_currency"]

    df["salary_min"] = df["salary_min"] * period_coef / currency_coef
    df["salary_max"] = df["salary_max"] * period_coef / currency_coef

    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2
    df["log_salary"] = np.log(df["salary_mid"])
    df["log_salary_min"] = np.log(df["salary_min"])
    df["log_salary_max"] = np.log(df["salary_max"])
    df["spread"] = df["salary_max"] - df["salary_min"]
    df["spread_rel"] = df["spread"] / df["salary_mid"]

    return df
