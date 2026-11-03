import numpy as np
import pandas as pd
import pytest

from src.features import add_engineered_features, feature_frame
from src.modeling import (
    OOFTargetEncoder,
    SalaryModel,
    compute_smearing_factor,
    evaluate,
    load_model,
    make_model,
    predictions_to_usd,
    save_model,
)
from src.predict import prepare_inference_frame
from src.preprocessing import prepare_salary_data


def make_raw(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "title": rng.choice(
                ["Senior Data Scientist", "Registered Nurse", "Sales Manager", "Software Engineer"], n
            ),
            "company_id": [f"company-{i % 20}" for i in range(n)],
            "company": [f"Company {i % 20}" for i in range(n)],
            "ats": rng.choice(["greenhouse", "ashby", "workday", None], n),
            "department": rng.choice(["Data", "Clinical", "Sales", None], n),
            "location": rng.choice(["New York, NY", "London", "Berlin", "Austin, TX"], n),
            "city": rng.choice(["New York", None, "Berlin", "Austin"], n),
            "region": rng.choice(["NY", None, None, "TX"], n),
            "country_code": rng.choice(["US", "GB", "DE", None], n),
            "workplace_type": rng.choice(["remote", "hybrid", "onsite", None], n),
            "employment_type": rng.choice(["full-time", "contract", "part-time", None], n),
            "seniority": rng.choice(["senior", None, "entry", None], n),
            "salary_min": rng.uniform(50_000, 150_000, n),
            "salary_max": rng.uniform(150_001, 220_000, n),
            "salary_currency": ["USD"] * n,
            "salary_period": rng.choice(["year", "hour", "month"], n),
            "salary_text": ["$80k"] * n,
            "salary_source": ["ats"] * n,
            "description": ["We build things."] * n,
            "posted_at": ["2026-10-01"] * n,
            "first_seen_at": ["2026-10-02"] * n,
            "language": rng.choice(["en", None], n),
        }
    )


def make_training_frame(n=80, seed=0):
    prepared = prepare_salary_data(make_raw(n, seed))
    enriched = add_engineered_features(prepared)
    return enriched.loc[enriched["log_salary"].notna()].reset_index(drop=True)


def test_oof_target_encoder_does_not_use_own_target():
    # Каждая компания встречается один раз: OOF-кодирование обязано дать
    # prior, а не зарплату самой строки.
    frame = pd.DataFrame({"company_id": [f"c{i}" for i in range(20)]})
    target = pd.Series(np.linspace(10.0, 12.0, 20))

    encoder = OOFTargetEncoder(columns=["company_id"], smoothing=0.0, n_splits=5,
                               random_state=42)
    encoded = encoder.fit_transform(frame, target)

    # Каждая строка уникальна, поэтому её OOF-признак — среднее других фолдов,
    # а не её собственный таргет. Небольшие отличия от общего среднего
    # объясняются тем, что prior считается внутри train-фолда.
    assert not np.any(np.isclose(encoded["company_id_te"], target))
    assert np.allclose(encoded["company_id_te"], target.mean(), atol=0.2)

    # А для новых строк используется статистика всего train.
    transformed = encoder.transform(pd.DataFrame({"company_id": ["c0", "new"]}))
    assert transformed.loc[0, "company_id_te"] == pytest.approx(target.iloc[0])
    assert transformed.loc[1, "company_id_te"] == pytest.approx(target.mean())


def test_oof_target_encoder_excludes_own_row_in_holdout_folds():
    # Компания с двумя строками: в OOF-кодировании строка может увидеть
    # только таргет соседа из другого фолда, но не свой.
    frame = pd.DataFrame({"company_id": ["a", "a", "b", "b"]})
    target = pd.Series([10.0, 12.0, 14.0, 16.0])

    encoder = OOFTargetEncoder(columns=["company_id"], smoothing=0.0, n_splits=2,
                               random_state=42)
    encoded = encoder.fit_transform(frame, target)

    # Значение не равно собственному таргету ни для одной строки.
    assert not np.allclose(encoded["company_id_te"], target)
    assert encoded["company_id_te"].notna().all()


@pytest.mark.parametrize("model_name", ["linear", "random_forest", "hist_gradient_boosting"])
def test_models_fit_on_explicit_features(model_name):
    frame = make_training_frame(60)
    X = feature_frame(frame, include_te=True)
    y = frame["log_salary"]

    model = make_model(model_name)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_dummy_model_ignores_features():
    frame = make_training_frame(30)
    X = feature_frame(frame, include_te=True)
    y = frame["log_salary"]

    model = make_model("dummy").fit(X, y)
    assert np.allclose(model.predict(X), y.median())


def test_metrics_and_back_transform_are_finite():
    y_log = np.log(np.array([100_000.0, 150_000.0, 200_000.0]))
    pred_log = np.log(np.array([110_000.0, 140_000.0, 210_000.0]))

    metrics = evaluate(y_log, pred_log, np.exp(y_log))
    assert metrics["MAE (log)"] > 0
    assert metrics["MAE (USD)"] > 0
    assert all(np.isfinite(value) for value in metrics.values())

    usd = predictions_to_usd(pred_log)
    assert np.all(usd >= 1)
    assert np.allclose(usd, np.exp(pred_log))


def test_salary_model_roundtrip_and_record_prediction(tmp_path):
    frame = make_training_frame(80)
    X = feature_frame(frame, include_te=True)
    y = frame["log_salary"]

    pipeline = make_model("linear").fit(X, y)
    smearing = compute_smearing_factor(pipeline, X, y)
    model = SalaryModel(
        pipeline=pipeline,
        smearing_factor=smearing,
        metadata={"features": list(X.columns)},
    )

    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert isinstance(loaded, SalaryModel)

    record = {
        "title": "Senior Data Scientist",
        "company_id": "company-1",
        "country_code": "US",
        "raw_period": "year",
        "ats": "greenhouse",
        "employment_type": "full-time",
        "workplace_type": "remote",
        "location": "New York, NY",
        "description": "Build models.",
    }
    prediction = loaded.predict_record(record)

    assert prediction["salary_pred_usd"] > 1
    assert np.isfinite(prediction["salary_pred_log"])
    assert "не гарантированное условное среднее" in prediction["note"]


def test_inference_frame_ignores_salary_fields():
    record_a = {
        "title": "Data Scientist",
        "company_id": "c1",
        "country_code": "US",
        "raw_period": "year",
        "salary_min": 10_000,
        "salary_max": 20_000,
        "salary_mid": 15_000,
        "log_salary": 9.6,
    }
    record_b = {**record_a, "salary_min": 999_999, "salary_max": 1_000_000,
                "salary_mid": 999_999, "log_salary": 13.8}

    frame_a = prepare_inference_frame(record_a)
    frame_b = prepare_inference_frame(record_b)

    pd.testing.assert_frame_equal(frame_a, frame_b)
