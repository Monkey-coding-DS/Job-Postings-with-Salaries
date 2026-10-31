"""Загрузка исходных данных проекта."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "job_postings_with_salaries.csv"

REQUIRED_COLUMNS = [
    "job_id",
    "title",
    "description",
    "salary_min",
    "salary_max",
    "salary_period",
    "salary_currency",
    "salary_source",
    "ats",
    "company",
    "company_id",
    "country_code",
    "location",
    "region",
    "city",
    "seniority",
    "employment_type",
    "workplace_type",
    "department",
    "language",
    "posted_at",
    "first_seen_at",
]


def load_data(path=None):
    """Загружает датасет вакансий с зарплатами.

    Путь вычисляется относительно корня проекта, поэтому указывать его
    вручную не нужно ни из корня проекта, ни из папки `notebooks`.
    """
    if path is None:
        path = DATA_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing}")

    return df
