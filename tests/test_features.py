import numpy as np
import pandas as pd
import pytest

from src.features import (
    BASE_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    add_engineered_features,
    assert_no_leakage,
    describe_feature_policy,
    feature_frame,
)
from src.preprocessing import prepare_salary_data


def raw_frame(n=40):
    return pd.DataFrame(
        {
            "job_id": [f"job-{i}" for i in range(n)],
            "title": ["Data Scientist" if i % 2 else "Nurse" for i in range(n)],
            "company": [f"Company {i % 5}" for i in range(n)],
            "company_id": [f"c{i % 5}" for i in range(n)],
            "ats": ["greenhouse", "ashby", "workday", None] * (n // 4),
            "department": ["Data", "Clinical", "Engineering", "Sales"] * (n // 4),
            "location": ["New York, NY", "London", "Berlin", "Austin, TX"] * (n // 4),
            "city": ["New York", None, "Berlin", "Austin"] * (n // 4),
            "region": ["NY", None, None, "TX"] * (n // 4),
            "country_code": ["US", "GB", "DE", None] * (n // 4),
            "workplace_type": ["remote", "hybrid", "onsite", None] * (n // 4),
            "employment_type": ["full-time", "contract", "part-time", None] * (n // 4),
            "seniority": ["senior", None, "entry", None] * (n // 4),
            "salary_min": [80_000.0] * n,
            "salary_max": [120_000.0] * n,
            "salary_currency": ["USD"] * n,
            "salary_period": ["year"] * n,
            "salary_text": ["$80k-$120k"] * n,
            "salary_source": ["ats"] * n,
            "description": ["Build models and dashboards."] * n,
            "posted_at": ["2026-10-01"] * n,
            "first_seen_at": ["2026-10-02"] * n,
            "language": ["en", None, "en", "de"] * (n // 4),
        }
    )


def model_frame(n=40):
    prepared = prepare_salary_data(raw_frame(n))
    return add_engineered_features(prepared)


def test_explicit_features_exclude_id_text_and_target_derivatives():
    frame = model_frame()
    features = feature_frame(frame, include_te=True)

    assert list(features.columns) == BASE_FEATURES + ["company_id"]
    for forbidden in [
        "job_id",
        "description",
        "salary_min",
        "salary_max",
        "salary_mid",
        "log_salary",
        "salary_text",
        "spread",
        "spread_rel",
        "outlier_any",
    ]:
        assert forbidden not in features.columns
        assert forbidden in EXCLUDED_COLUMNS

    assert_no_leakage(features.columns)


def test_assert_no_leakage_raises_for_any_forbidden_column():
    with pytest.raises(ValueError, match="запрещённые колонки"):
        assert_no_leakage(BASE_FEATURES + ["job_id"])
    with pytest.raises(ValueError, match="запрещённые колонки"):
        assert_no_leakage(BASE_FEATURES + ["log_salary"])
    with pytest.raises(ValueError, match="запрещённые колонки"):
        assert_no_leakage(BASE_FEATURES + ["description"])


def test_add_engineered_features_does_not_overwrite_provided_values():
    frame = model_frame(4)
    frame["seniority_title"] = "executive"
    frame["family"] = "legal"
    frame["us_state"] = "CA"

    enriched = add_engineered_features(frame)

    assert (enriched["seniority_title"] == "executive").all()
    assert (enriched["family"] == "legal").all()
    assert (enriched["us_state"] == "CA").all()


def test_feature_policy_table_covers_base_and_te():
    policy = describe_feature_policy()
    assert set(BASE_FEATURES).issubset(set(policy["признак"]))
    assert "company_id" in set(policy["признак"])
    assert "description" in set(policy["признак"])


def test_numeric_features_are_numeric_and_finite_or_nan():
    frame = model_frame()
    numeric = feature_frame(frame)[NUMERIC_FEATURES]

    # description_len и флаги не должны быть бесконечностями.
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
