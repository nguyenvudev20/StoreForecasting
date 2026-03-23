import os
import pickle
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "features.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")
VALID_RESULT_PATH = os.path.join(MODEL_DIR, "valid_result.pkl")

FEATURE_COLS = [
    "store",
    "item",
    "day",
    "month",
    "year",
    "dayofweek",
    "weekofyear",
    "quarter",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return np.mean(diff) * 100


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def validate_columns(df: pd.DataFrame):
    required = {"date", "store", "item", "sales"}
    missing = required - set(df.columns)
    return list(missing)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags=(7, 14, 28, 30)) -> pd.DataFrame:
    df = df.copy().sort_values(["store", "item", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 30)) -> pd.DataFrame:
    df = df.copy().sort_values(["store", "item", "date"])
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store", "item"])["sales"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["store", "item", "date"])
    df = build_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def train_valid_split_by_date(df: pd.DataFrame, valid_days=90):
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=valid_days)
    train_df = df[df["date"] < split_date].copy()
    valid_df = df[df["date"] >= split_date].copy()
    return train_df, valid_df, split_date


def build_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )


def fit_model(train_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["sales"]

    model = build_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, valid_df: pd.DataFrame):
    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df["sales"]

    preds = model.predict(X_valid)

    metrics = {
        "MAE": mean_absolute_error(y_valid, preds),
        "RMSE": np.sqrt(mean_squared_error(y_valid, preds)),
        "R2": r2_score(y_valid, preds),
        "MAPE": safe_mape(y_valid, preds),
        "SMAPE": smape(y_valid, preds),
    }

    result_df = valid_df[["date", "store", "item", "sales"]].copy()
    result_df["pred"] = preds
    return metrics, result_df


def model_exists(
    model_path: str = MODEL_PATH,
    feature_path: str = FEATURE_PATH,
):
    return os.path.exists(model_path) and os.path.exists(feature_path)


def save_artifacts(
    model,
    feature_cols=FEATURE_COLS,
    metrics: Optional[Dict] = None,
    valid_result: Optional[pd.DataFrame] = None,
    model_path: str = MODEL_PATH,
    feature_path: str = FEATURE_PATH,
    metrics_path: str = METRICS_PATH,
    valid_result_path: str = VALID_RESULT_PATH,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(feature_path, "wb") as f:
        pickle.dump(feature_cols, f)

    if metrics is not None:
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

    if valid_result is not None:
        with open(valid_result_path, "wb") as f:
            pickle.dump(valid_result, f)


def load_artifacts(
    model_path: str = MODEL_PATH,
    feature_path: str = FEATURE_PATH,
    metrics_path: str = METRICS_PATH,
    valid_result_path: str = VALID_RESULT_PATH,
):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_path, "rb") as f:
        feature_cols = pickle.load(f)

    metrics = None
    valid_result = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

    if os.path.exists(valid_result_path):
        with open(valid_result_path, "rb") as f:
            valid_result = pickle.load(f)

    return model, feature_cols, metrics, valid_result


def get_series(df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    out = df[(df["store"] == store) & (df["item"] == item)].copy()
    return out.sort_values("date")


def build_single_feature_row(
    history_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    store: int,
    item: int
):
    hist = history_df.sort_values("date").copy()
    sales_list = hist["sales"].tolist()

    if len(sales_list) < 30:
        raise ValueError("Cần ít nhất 30 bản ghi lịch sử cho store-item để tạo lag features.")

    row = {
        "store": store,
        "item": item,
        "day": forecast_date.day,
        "month": forecast_date.month,
        "year": forecast_date.year,
        "dayofweek": forecast_date.dayofweek,
        "weekofyear": int(forecast_date.isocalendar().week),
        "quarter": forecast_date.quarter,
        "is_weekend": 1 if forecast_date.dayofweek >= 5 else 0,
        "lag_7": sales_list[-7],
        "lag_14": sales_list[-14],
        "lag_28": sales_list[-28],
        "lag_30": sales_list[-30],
        "rolling_mean_7": float(np.mean(sales_list[-7:])),
        "rolling_mean_30": float(np.mean(sales_list[-30:])),
    }

    return pd.DataFrame([row])


def recursive_forecast(
    history_df: pd.DataFrame,
    model,
    forecast_days: int,
    store: int,
    item: int,
    feature_cols=FEATURE_COLS,
):
    temp_hist = history_df.sort_values("date")[["date", "sales"]].copy()
    temp_hist["date"] = pd.to_datetime(temp_hist["date"])

    results = []
    last_date = temp_hist["date"].max()

    for _ in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)
        feature_df = build_single_feature_row(temp_hist, next_date, store, item)
        pred = float(model.predict(feature_df[feature_cols])[0])
        pred = max(0.0, pred)

        results.append({
            "date": next_date,
            "store": store,
            "item": item,
            "forecast_sales": pred
        })

        temp_hist = pd.concat(
            [temp_hist, pd.DataFrame([{"date": next_date, "sales": pred}])],
            ignore_index=True
        )
        last_date = next_date

    return pd.DataFrame(results)


def compute_inventory_suggestion(
    current_inventory: float,
    forecast_demand: float,
    lead_time_days: int,
    avg_daily_demand: float,
    demand_std: float,
    service_z: float = 1.65
):
    safety_stock = service_z * demand_std * np.sqrt(max(lead_time_days, 1))
    demand_during_lead = avg_daily_demand * lead_time_days
    reorder_point = demand_during_lead + safety_stock
    suggested_order = max(0.0, forecast_demand + safety_stock - current_inventory)

    return {
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "suggested_order": float(suggested_order),
    }