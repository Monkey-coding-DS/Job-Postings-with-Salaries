"""Интеграция модели: подготовка одного объекта и CLI-прогноз зарплаты.

Пример:

    python -m src.predict \
        --title "Senior Data Scientist" \
        --company-id acme \
        --country-code US \
        --raw-period year \
        --ats greenhouse \
        --employment-type full-time \
        --workplace-type remote \
        --location "New York, NY"

Модель должна быть предварительно обучена и сохранена ноутбуком
``salary_prediction.ipynb`` в ``models/salary_model.joblib``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    MISSING_CATEGORY,
    MISSING_FLAG_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_ENCODING_FEATURES,
    add_engineered_features,
    feature_frame,
)
from src.modeling import SalaryModel, load_model

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "salary_model.joblib"


def _as_frame(record: dict | pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(record, dict):
        return pd.DataFrame([record])
    if isinstance(record, pd.Series):
        return record.to_frame().T
    if isinstance(record, pd.DataFrame):
        return record.copy()
    raise TypeError("Ожидается dict, pandas.Series или pandas.DataFrame")


def prepare_inference_frame(record: dict | pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Готовит X для модели из «сырой» записи о вакансии.

    Обязателен только ``title`` (для эвристик уровня/семейства). Остальные
    поля опциональны: отсутствующие категории становятся отдельной
    категорией/пропуском, ``company_id`` — prior через target encoding,
    ``raw_period`` берётся из ``salary_period`` (по умолчанию ``year``).

    Зарплатные поля записи (``salary_min`` и т. п.) игнорируются: модель
    не должна видеть таргет.
    """
    frame = _as_frame(record)

    source_columns = [
        "title",
        "company_id",
        "location",
        "region",
        "city",
        "language",
        "ats",
        "country_code",
        "employment_type",
        "workplace_type",
        "seniority",
        "department",
        "description",
        "salary_period",
        "raw_period",
    ]
    for column in source_columns:
        if column not in frame:
            frame[column] = np.nan

    frame["description"] = frame["description"].fillna("")
    frame["company_id"] = frame["company_id"].where(
        frame["company_id"].notna(), MISSING_CATEGORY
    )

    if "raw_period" not in frame or frame["raw_period"].isna().all():
        frame["raw_period"] = frame["salary_period"].fillna("year")
    else:
        frame["raw_period"] = frame["raw_period"].fillna(
            frame["salary_period"].fillna("year")
        )

    frame = add_engineered_features(frame)

    for column in CATEGORICAL_FEATURES:
        if column not in frame:
            frame[column] = np.nan
    for column in NUMERIC_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    for column in MISSING_FLAG_COLUMNS:
        frame[f"is_missing_{column}"] = frame[column].isna()
    for column in TARGET_ENCODING_FEATURES:
        if column not in frame:
            frame[column] = MISSING_CATEGORY

    return feature_frame(frame, include_te=True)


def predict_salary(record: dict | pd.DataFrame | pd.Series,
                   model: SalaryModel | str | Path | None = None) -> dict:
    """Возвращает прогноз для одной записи.

    ``model`` может быть уже загруженным ``SalaryModel`` или путём к joblib.
    """
    if model is None:
        model = DEFAULT_MODEL_PATH
    if isinstance(model, (str, Path)):
        model = load_model(model)

    return model.predict_record(record)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогноз зарплаты по описанию вакансии")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="путь к сохранённой модели joblib")
    parser.add_argument("--input", type=Path,
                        help="JSON-файл с объектом или списком объектов вакансии")
    parser.add_argument("--title", default=None)
    parser.add_argument("--company-id", default=None)
    parser.add_argument("--ats", default=None)
    parser.add_argument("--country-code", default=None)
    parser.add_argument("--employment-type", default=None)
    parser.add_argument("--workplace-type", default=None)
    parser.add_argument("--seniority", default=None)
    parser.add_argument("--department", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--raw-period", default=None,
                        help="year / month / week / day / hour")
    parser.add_argument("--description", default=None)
    parser.add_argument("--conditional-mean", action="store_true",
                        help="применить Duan smearing (оценка среднего, не медианы)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = load_model(args.model)

    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        payload = {
            "title": args.title,
            "company_id": args.company_id,
            "ats": args.ats,
            "country_code": args.country_code,
            "employment_type": args.employment_type,
            "workplace_type": args.workplace_type,
            "seniority": args.seniority,
            "department": args.department,
            "location": args.location,
            "region": args.region,
            "raw_period": args.raw_period,
            "description": args.description,
        }

    records = payload if isinstance(payload, list) else [payload]
    output = []
    for record in records:
        frame = prepare_inference_frame(record)
        log_prediction = float(model.predict_log(frame)[0])
        output.append(
            {
                "salary_pred_log": log_prediction,
                "salary_pred_usd": float(model.predict_usd(frame)[0]),
                "salary_pred_usd_conditional_mean": float(
                    model.predict_usd(frame, conditional_mean=args.conditional_mean)[0]
                ),
                "note": (
                    "salary_pred_usd = exp(log_prediction): оценка типичной "
                    "зарплаты, не гарантированное условное среднее."
                ),
            }
        )

    print(json.dumps(output if len(output) > 1 else output[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
