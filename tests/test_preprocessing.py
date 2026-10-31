import numpy as np
import pandas as pd
import pytest

from src.preprocessing import iqr_outlier, prepare_salary_data


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
    assert prepared.loc[0, "log_salary"] == pytest.approx(np.log(125_000.0))


def test_prepare_salary_data_rejects_unknown_period():
    df = pd.DataFrame(
        {
            "salary_min": [100_000.0],
            "salary_max": [150_000.0],
            "salary_period": ["fortnight"],
            "salary_currency": ["USD"],
        }
    )
    with pytest.raises(ValueError, match="Неизвестные периоды"):
        prepare_salary_data(df)


def test_prepare_salary_data_rejects_unknown_currency():
    df = pd.DataFrame(
        {
            "salary_min": [100_000.0],
            "salary_max": [150_000.0],
            "salary_period": ["year"],
            "salary_currency": ["XYZ"],
        }
    )
    with pytest.raises(ValueError, match="Неизвестные валюты"):
        prepare_salary_data(df)


def test_iqr_outlier_flags_values_outside_fences():
    values = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 11.0, 12.0, 13.0, 11.0, 12.0, 5000.0])
    flags = iqr_outlier(values)
    assert not bool(flags.iloc[0])
    assert bool(flags.iloc[-1])
