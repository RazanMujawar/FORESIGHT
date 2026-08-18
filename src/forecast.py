from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_wape(actual, forecast):

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return (
        np.abs(actual - forecast).sum()
        / denominator
    )


def calculate_mae(actual, forecast):

    return np.mean(
        np.abs(
            np.asarray(actual)
            - np.asarray(forecast)
        )
    )


def calculate_bias(actual, forecast):

    return np.sum(
        np.asarray(forecast)
        - np.asarray(actual)
    )


# ============================================================
# LOAD WEEKLY DATA
# ============================================================

def load_weekly_data():

    weekly = pd.read_parquet(
        PROCESSED_DIR / "weekly_demand.parquet"
    )

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"]
    )

    return weekly


# ============================================================
# ADD CALENDAR FEATURES
# ============================================================

def add_calendar_features(df):

    df = df.copy()

    df["week_of_year"] = (
        df["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = (
        df["week_start"]
        .dt.month
    )

    df["quarter"] = (
        df["week_start"]
        .dt.quarter
    )

    df["year"] = (
        df["week_start"]
        .dt.year
    )

    return df


# ============================================================
# ADD DEMAND LAGS
# ============================================================

def add_lag_features(df):

    df = df.sort_values(
        ["sku_id", "week_start"]
    ).copy()

    grouped = df.groupby(
        "sku_id",
        group_keys=False
    )

    for lag in [
        1,
        2,
        4,
        8,
        13,
        26,
        52
    ]:

        df[f"lag_{lag}"] = (
            grouped["units_sold"]
            .shift(lag)
        )

    return df


# ============================================================
# ADD ROLLING FEATURES
# ============================================================

def add_rolling_features(df):

    df = df.sort_values(
        ["sku_id", "week_start"]
    ).copy()

    grouped = df.groupby(
        "sku_id"
    )["units_sold"]

    # IMPORTANT:
    # Shift by 1 first so the current week's
    # target is never included in its own features.

    df["rolling_mean_4"] = (
        grouped
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_8"] = (
        grouped
        .shift(1)
        .rolling(8)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_13"] = (
        grouped
        .shift(1)
        .rolling(13)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


# ============================================================
# LOAD SKU INFORMATION
# ============================================================

def load_sku_features():

    sku = pd.read_parquet(
        PROCESSED_DIR / "sku_master.parquet"
    )

    return sku[
        [
            "sku_id",
            "category",
            "subcategory",
            "brand",
            "unit_price",
            "cost_price",
        ]
    ]


# ============================================================
# PREPARE MODEL DATASET
# ============================================================

def prepare_model_data():

    print("\nLoading weekly demand...")

    weekly = load_weekly_data()

    print(
        f"Weekly records: {len(weekly):,}"
    )

    print("\nAdding calendar features...")

    weekly = add_calendar_features(
        weekly
    )

    print("\nAdding lag features...")

    weekly = add_lag_features(
        weekly
    )

    print("\nAdding rolling features...")

    weekly = add_rolling_features(
        weekly
    )

    print("\nAdding SKU features...")

    sku = load_sku_features()

    weekly = weekly.merge(
        sku,
        on="sku_id",
        how="left"
    )

    # Encode categorical features
    categorical_columns = [
        "sku_id",
        "category",
        "subcategory",
        "brand",
    ]

    for col in categorical_columns:

        weekly[col] = (
            weekly[col]
            .astype("category")
        )

    # Remove rows where the model cannot
    # construct all required lag features.
    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_13",
        "lag_26",
        "lag_52",
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_mean_13",
    ]

    before = len(weekly)

    weekly = weekly.dropna(
        subset=feature_columns
    )

    after = len(weekly)

    print(
        f"\nRemoved {before - after:,} "
        f"rows without sufficient history."
    )

    return weekly


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_data(
    df,
    validation_weeks=12
):

    max_week = df["week_start"].max()

    validation_start = (
        max_week
        - pd.Timedelta(
            weeks=validation_weeks - 1
        )
    )

    train = df[
        df["week_start"]
        < validation_start
    ].copy()

    validation = df[
        df["week_start"]
        >= validation_start
    ].copy()

    return train, validation


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

def train_model(
    train,
    features
):

    print("\nTraining LightGBM...")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["units_sold"],
        categorical_feature=[
            "sku_id",
            "category",
            "subcategory",
            "brand",
        ],
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    validation,
    features
):

    predictions = model.predict(
        validation[features]
    )

    # Demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    validation = validation.copy()

    validation["prediction"] = (
        predictions
    )

    wape = calculate_wape(
        validation["units_sold"],
        validation["prediction"]
    )

    mae = calculate_mae(
        validation["units_sold"],
        validation["prediction"]
    )

    bias = calculate_bias(
        validation["units_sold"],
        validation["prediction"]
    )

    return validation, {
        "model": "LightGBM",
        "wape": wape,
        "mae": mae,
        "bias": bias,
        "validation_rows": len(validation),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    features
):

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        REPORTS_DIR
        / "feature_importance.csv",
        index=False
    )

    return importance


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT — LIGHTGBM FORECAST MODEL")
    print("=" * 70)

    df = prepare_model_data()

    features = [
        "sku_id",
        "category",
        "subcategory",
        "brand",
        "unit_price",
        "cost_price",
        "week_of_year",
        "month",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_13",
        "lag_26",
        "lag_52",
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_mean_13",
    ]

    print("\nSplitting data...")

    train, validation = split_data(
        df,
        validation_weeks=12
    )

    print(
        f"Training rows:   {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    model = train_model(
        train,
        features
    )

    validation, results = evaluate_model(
        model,
        validation,
        features
    )

    print("\n" + "=" * 70)
    print("LIGHTGBM RESULTS")
    print("=" * 70)

    for key, value in results.items():

        print(
            f"{key}: {value}"
        )

    importance = save_feature_importance(
        model,
        features
    )

    print("\nTOP FEATURES")

    print(
        importance.head(10)
    )

    validation[
        [
            "sku_id",
            "week_start",
            "units_sold",
            "prediction",
        ]
    ].to_parquet(
        PROCESSED_DIR
        / "validation_predictions.parquet",
        index=False
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        REPORTS_DIR
        / "lightgbm_results.csv",
        index=False
    )

    print(
        "\n✓ validation_predictions.parquet saved"
    )

    print(
        "✓ lightgbm_results.csv saved"
    )

    print(
        "✓ feature_importance.csv saved"
    )


if __name__ == "__main__":
    main()