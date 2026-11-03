"""Модели, метрики и out-of-fold target encoding для прогноза зарплаты.

Модуль намеренно не зависит от ноутбуков: те же пайплайны используются
в ``salary_prediction.ipynb`` и в CLI ``python -m src.predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    MISSING_CATEGORY,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_ENCODING_FEATURES,
)


class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding без утечки таргета обучающих строк.

    ``fit_transform`` строит для каждой обучающей строки признак по статистике
    *других* фолдов (cross-fitting): таргет самой строки в её кодировании не
    участвует. ``fit`` запоминает статистику на всём train, а ``transform``
    применяет её к валидации/тесту/новым данным.

    Сглаживание: ``(sum + prior * smoothing) / (count + smoothing)``.
    Неизвестные категории получают prior обучающей выборки.
    """

    def __init__(self, columns=None, smoothing: float = 20.0, n_splits: int = 5,
                 random_state: int = RANDOM_STATE):
        self.columns = columns
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _keys(series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), MISSING_CATEGORY).astype(str)

    def _fit_mapping(self, frame: pd.DataFrame, target: pd.Series, column: str):
        keys = self._keys(frame[column])
        prior = float(np.asarray(target, dtype=float).mean())
        grouped = (
            pd.DataFrame({"key": keys, "y": np.asarray(target, dtype=float)})
            .groupby("key")["y"]
            .agg(["mean", "count"])
        )
        mapping = (
            grouped["mean"] * grouped["count"] + prior * self.smoothing
        ) / (grouped["count"] + self.smoothing)
        return mapping, prior

    def _apply_mapping(self, frame: pd.DataFrame, column: str, mapping: pd.Series, prior: float):
        return self._keys(frame[column]).map(mapping).astype(float).fillna(prior)

    def fit(self, X: pd.DataFrame, y):
        self.columns_ = list(self.columns or TARGET_ENCODING_FEATURES)
        self.mappings_ = {}
        self.priors_ = {}
        for column in self.columns_:
            self.mappings_[column], self.priors_[column] = self._fit_mapping(X, y, column)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=X.index)
        for column in self.columns_:
            encoded[f"{column}_te"] = self._apply_mapping(
                X, column, self.mappings_[column], self.priors_[column]
            )
        return encoded

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        if y is None:
            raise ValueError("OOFTargetEncoder требует y при fit_transform")

        self.fit(X, y)
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        encoded = pd.DataFrame(index=X.index, dtype=float)

        for column in self.columns_:
            oof = pd.Series(index=X.index, dtype=float)
            folds = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            for train_idx, valid_idx in folds.split(X):
                mapping, prior = self._fit_mapping(
                    X.iloc[train_idx], target.iloc[train_idx], column
                )
                oof.iloc[valid_idx] = self._apply_mapping(
                    X.iloc[valid_idx], column, mapping, prior
                ).to_numpy()
            encoded[f"{column}_te"] = oof
        return encoded

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{column}_te" for column in self.columns_], dtype=object)


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_pipeline(*, dense: bool) -> Pipeline:
    # NaN не заполняем модой: пропуск — отдельная категория (гипотеза 19),
    # а индикаторы пропусков дополнительно передаются числовыми признаками.
    return Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=not dense,
                ),
            )
        ]
    )


def make_preprocessor(*, dense: bool = False, include_te: bool = True) -> ColumnTransformer:
    """Собирает преобразование явного списка признаков.

    ``company_id`` попадает только внутрь ``OOFTargetEncoder`` — как обычная
    категория или числовой id он в модель не передаётся.
    """
    transformers = [
        ("num", _numeric_pipeline(), NUMERIC_FEATURES),
        ("cat", _categorical_pipeline(dense=dense), CATEGORICAL_FEATURES),
    ]
    if include_te:
        transformers.append(
            ("te", OOFTargetEncoder(columns=TARGET_ENCODING_FEATURES), TARGET_ENCODING_FEATURES)
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_model(name: str):
    """Создаёт модель по имени. Все модели работают на одном наборе признаков."""
    name = name.lower()
    if name in {"dummy", "baseline"}:
        return DummyRegressor(strategy="median")

    if name in {"linear", "linear_regression"}:
        return Pipeline(
            [("preprocessor", make_preprocessor(dense=False)), ("regressor", LinearRegression())]
        )

    if name in {"random_forest", "rf"}:
        return Pipeline(
            [
                ("preprocessor", make_preprocessor(dense=False)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if name in {"hist_gradient_boosting", "hgb"}:
        return Pipeline(
            [
                ("preprocessor", make_preprocessor(dense=True)),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(f"Неизвестная модель: {name}")


def make_text_model(*, alpha: float = 1.0, max_features: int = 8000,
                    min_df: int = 5) -> Pipeline:
    """Отдельный текстовый пайплайн: TF-IDF по маскированному описанию + Ridge."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    stop_words="english",
                ),
            ),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def predictions_to_usd(pred_log, *, smearing_factor: float = 1.0) -> np.ndarray:
    """Обратное преобразование логарифма с защитой от переполнения.

    ``exp(pred_log)`` — оценка условной *медианы* при примерно нормальных
    остатках, а не гарантированное условное среднее. ``smearing_factor``
    (Duan's smearing, ``mean(exp(residual))`` на train) сдвигает оценку к
    среднему, но и он не исправляет смещение выбора модели и данных.
    """
    values = np.exp(np.clip(np.asarray(pred_log, dtype=float), -20, 25))
    return np.clip(values * smearing_factor, 1.0, None)


def evaluate(y_log, pred_log, y_usd=None, *, smearing_factor: float = 1.0) -> dict[str, float]:
    """Метрики в логарифмической и исходной шкале.

    Основная метрика сравнения — MAE в лог-шкале (устойчива к дорогим
    хвостам и интерпретируется как относительная ошибка). MAE в USD
    считается по обратному преобразованию и поэтому чувствительна к
    смещению ``exp``.
    """
    y_log = np.asarray(y_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    result = {
        "MAE (log)": mean_absolute_error(y_log, pred_log),
        "RMSE (log)": mean_squared_error(y_log, pred_log) ** 0.5,
        "R2 (log)": r2_score(y_log, pred_log),
    }

    if y_usd is not None:
        actual = np.asarray(y_usd, dtype=float)
        predicted = predictions_to_usd(pred_log, smearing_factor=smearing_factor)
        result.update(
            {
                "MAE (USD)": mean_absolute_error(actual, predicted),
                "RMSE (USD)": mean_squared_error(actual, predicted) ** 0.5,
                "MAPE": mean_absolute_percentage_error(actual, predicted),
                "RMSLE": np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)),
            }
        )
    return result


@dataclass
class SalaryModel:
    """Готовая модель + метаданные, пригодная для сохранения и интеграции."""

    pipeline: Any
    feature_columns: list[str] = field(default_factory=lambda: list(BASE_FEATURES))
    include_company_id: bool = True
    smearing_factor: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)

    def predict_usd(self, X: pd.DataFrame, *, conditional_mean: bool = False) -> np.ndarray:
        factor = self.smearing_factor if conditional_mean else 1.0
        return predictions_to_usd(self.predict_log(X), smearing_factor=factor)

    def predict_record(self, record: dict | pd.DataFrame, *, conditional_mean: bool = False) -> dict:
        from src.predict import prepare_inference_frame

        frame = prepare_inference_frame(record)
        log_prediction = self.predict_log(frame)[0]
        result = {
            "salary_pred_log": float(log_prediction),
            "salary_pred_usd": float(
                predictions_to_usd([log_prediction], smearing_factor=1.0)[0]
            ),
            "salary_pred_usd_conditional_mean": float(
                predictions_to_usd(
                    [log_prediction],
                    smearing_factor=self.smearing_factor if conditional_mean else 1.0,
                )[0]
            ),
            "note": (
                "salary_pred_usd = exp(log_prediction) — оценка медианы/типичного "
                "значения; это не гарантированное условное среднее зарплаты."
            ),
        }
        return result


def compute_smearing_factor(pipeline, X_train, y_train_log) -> float:
    """Duan's smearing factor: mean(exp(residual)) на обучающей выборке."""
    residuals = np.asarray(y_train_log, dtype=float) - np.asarray(
        pipeline.predict(X_train), dtype=float
    )
    return float(np.mean(np.exp(residuals)))


def save_model(model: SalaryModel, path) -> None:
    joblib.dump(model, path)


def load_model(path) -> SalaryModel:
    return joblib.load(path)
