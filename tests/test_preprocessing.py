import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CURRENCY_PER_USD,
    FX_RATES_AS_OF,
    FX_RATES_SOURCE,
    family_from_title,
    iqr_outlier,
    prepare_salary_data,
    seniority_from_title,
)


def make_frame(**overrides):
    data = {
        "salary_min": [100_000.0],
        "salary_max": [150_000.0],
        "salary_period": ["year"],
        "salary_currency": ["USD"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_prepare_salary_data_converts_to_usd_per_year():
    df = pd.DataFrame(
        {
            "salary_min": [100_000.0, 50_000.0],
            "salary_max": [150_000.0, 70_000.0],
            "salary_period": ["year", "year"],
            "salary_currency": ["USD", "EUR"],
        }
    )
    prepared = prepare_salary_data(df)

    assert prepared.loc[0, "salary_mid"] == pytest.approx(125_000.0)
    # 50 000 EUR за год -> ~56 180 USD, 70 000 EUR -> ~78 651 USD, середина ~67 415 USD
    expected_mid = (50_000 / 0.89 + 70_000 / 0.89) / 2
    assert prepared.loc[1, "salary_mid"] == pytest.approx(expected_mid, abs=1e-6)
    assert prepared.loc[0, "raw_period"] == "year"
    assert prepared.loc[0, "raw_currency"] == "USD"
    assert prepared.loc[0, "log_salary"] == pytest.approx(np.log(125_000.0))


def test_hourly_salary_is_annualized():
    df = make_frame(salary_min=[50.0], salary_max=[60.0], salary_period=["hour"])
    prepared = prepare_salary_data(df)

    assert prepared.loc[0, "salary_min"] == pytest.approx(50.0 * 2080)
    assert prepared.loc[0, "salary_max"] == pytest.approx(60.0 * 2080)


def test_currency_rates_are_documented_and_point_in_one_direction():
    assert FX_RATES_AS_OF == "2026-10-15"
    assert "НБРБ" in FX_RATES_SOURCE
    assert CURRENCY_PER_USD["USD"] == 1.0
    assert all(rate > 0 for rate in CURRENCY_PER_USD.values())

    # 1 USD = 0.89 EUR, значит 89 EUR = 100 USD.
    prepared = prepare_salary_data(make_frame(salary_min=[89.0], salary_max=[89.0],
                                              salary_currency=["EUR"]))
    assert prepared.loc[0, "salary_mid"] == pytest.approx(100.0)


def test_prepare_salary_data_rejects_unknown_period():
    with pytest.raises(ValueError, match="Неизвестные периоды"):
        prepare_salary_data(make_frame(salary_period=["fortnight"]))


def test_prepare_salary_data_rejects_unknown_currency():
    with pytest.raises(ValueError, match="Неизвестные валюты"):
        prepare_salary_data(make_frame(salary_currency=["XYZ"]))


def test_missing_values_are_allowed_and_produce_nan_without_infinities():
    df = make_frame(
        salary_min=[100_000.0, np.nan, 80_000.0],
        salary_max=[150_000.0, 120_000.0, np.nan],
        salary_period=["year", "year", "year"],
        salary_currency=["USD", "USD", None],
    )
    prepared = prepare_salary_data(df)

    # Одна граница отсутствует -> середина и лог не считаются, но вторая
    # граница остаётся доступной для диагностики.
    assert np.isnan(prepared.loc[1, "salary_mid"])
    assert np.isnan(prepared.loc[1, "log_salary"])
    assert prepared.loc[1, "salary_max"] == pytest.approx(120_000.0)
    assert np.isnan(prepared.loc[2, "salary_mid"])

    numeric_columns = ["salary_min", "salary_max", "salary_mid", "log_salary",
                       "log_salary_min", "log_salary_max", "spread", "spread_rel"]
    assert np.isfinite(prepared.loc[0, numeric_columns]).all()


@pytest.mark.parametrize("value", [0.0, -1.0, -100_000.0])
def test_zero_and_negative_salaries_become_nan_by_default(value):
    prepared = prepare_salary_data(make_frame(salary_min=[value]))

    assert np.isnan(prepared.loc[0, "salary_min"])
    assert np.isnan(prepared.loc[0, "salary_mid"])
    assert np.isnan(prepared.loc[0, "log_salary"])
    assert bool(prepared.loc[0, "invalid_salary_min"])


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_zero_and_negative_salaries_raise_in_strict_mode(value):
    with pytest.raises(ValueError, match="Некорректные значения"):
        prepare_salary_data(make_frame(salary_min=[value]), strict=True)


def test_invalid_salary_bounds_are_flagged_and_cleared():
    prepared = prepare_salary_data(make_frame(salary_min=[200_000.0], salary_max=[100_000.0]))

    assert bool(prepared.loc[0, "invalid_salary_range"])
    assert np.isnan(prepared.loc[0, "salary_min"])
    assert np.isnan(prepared.loc[0, "salary_max"])
    assert np.isnan(prepared.loc[0, "salary_mid"])


def test_invalid_salary_bounds_raise_in_strict_mode():
    with pytest.raises(ValueError, match="salary_min больше salary_max"):
        prepare_salary_data(
            make_frame(salary_min=[200_000.0], salary_max=[100_000.0]), strict=True
        )


@pytest.mark.parametrize("value", ["not-a-number", "", "100,000", np.inf])
def test_non_numeric_salaries_become_nan(value):
    prepared = prepare_salary_data(make_frame(salary_min=[value]))

    assert np.isnan(prepared.loc[0, "salary_min"])
    assert np.isnan(prepared.loc[0, "log_salary"])
    assert bool(prepared.loc[0, "invalid_salary_min"])


def test_non_numeric_salaries_raise_in_strict_mode():
    with pytest.raises(ValueError, match="Некорректные значения"):
        prepare_salary_data(make_frame(salary_min=["not-a-number"]), strict=True)


def test_prepare_salary_data_does_not_modify_input():
    df = make_frame(salary_min=[50_000.0], salary_max=[70_000.0], salary_currency=["EUR"])
    before = df.copy(deep=True)

    prepared = prepare_salary_data(df)

    pd.testing.assert_frame_equal(df, before)
    assert prepared is not df
    assert prepared.loc[0, "salary_min"] == pytest.approx(50_000 / 0.89)
    assert df.loc[0, "salary_min"] == 50_000.0


def test_iqr_outlier_flags_values_outside_fences():
    values = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 11.0, 12.0, 13.0, 11.0, 12.0, 5000.0])
    flags = iqr_outlier(values)
    assert not bool(flags.iloc[0])
    assert bool(flags.iloc[-1])


@pytest.mark.parametrize(
    ("title", "expected_seniority", "expected_family"),
    [
        ("Senior Director", "executive", "other"),
        ("Lead Data Scientist", "senior", "data_ai"),
        ("Principal Engineer", "senior", "engineering"),
        ("Head of Analytics", "executive", "data_ai"),
        ("Senior Software Engineer", "senior", "engineering"),
        ("Data Analyst Intern", "internship", "data_ai"),
        ("Junior Accountant", "entry", "finance"),
        ("Vice President of Sales", "executive", "sales"),
        ("Director of Engineering", "executive", "engineering"),
        ("Staff Machine Learning Engineer", "senior", "engineering"),
        ("Marketing Manager", "mid", "marketing"),
        ("Registered Nurse", "mid", "healthcare"),
    ],
)
def test_seniority_and_family_heuristics(title, expected_seniority, expected_family):
    assert seniority_from_title(title) == expected_seniority
    assert family_from_title(title) == expected_family


def test_seniority_pattern_priority_is_documented_by_behavior():
    # executive проверяется раньше senior/entry, internship — раньше entry:
    # это осознанный приоритет, а не случайный порядок словаря.
    assert seniority_from_title("Senior Director of Finance") == "executive"
    assert seniority_from_title("Junior Data Intern") == "internship"


def test_empty_title_falls_back_to_mid_and_other():
    assert seniority_from_title(None) == "mid"
    assert family_from_title(None) == "other"
