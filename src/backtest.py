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

def wape(actual, forecast):

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return (
        np.abs(actual - forecast).sum()
        / denominator
    )


def mae(actual, forecast):

    return np.mean(
        np.abs(
            np.asarray(actual)
            - np.asarray(forecast)
        )
    )


def bias(actual, forecast):

    return np.sum(
        np.asarray(forecast)
        - np.asarray(actual)
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_parquet(
        PROCESSED_DIR / "weekly_demand.parquet"
    )

    df["week_start"] = pd.to_datetime(
        df["week_start"]
    )

    sku = pd.read_parquet(
        PROCESSED_DIR / "sku_master.parquet"
    )

    sku = sku[
        [
            "sku_id",
            "category",
            "subcategory",
            "brand",
            "unit_price",
            "cost_price",
        ]
    ]

    return df, sku


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df, sku):

    df = df.sort_values(
        ["sku_id", "week_start"]
    ).copy()

    # Calendar features
    df["week_of_year"] = (
        df["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = (
        df["week_start"].dt.month
    )

    df["quarter"] = (
        df["week_start"].dt.quarter
    )

    # Lags
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

    # Rolling features
    grouped_demand = (
        df.groupby("sku_id")["units_sold"]
    )

    df["rolling_mean_4"] = (
        grouped_demand
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    df["rolling_mean_8"] = (
        grouped_demand
        .shift(1)
        .rolling(8)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    df["rolling_mean_13"] = (
        grouped_demand
        .shift(1)
        .rolling(13)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    # SKU attributes
    df = df.merge(
        sku,
        on="sku_id",
        how="left"
    )

    # Categorical columns
    categorical_columns = [
        "sku_id",
        "category",
        "subcategory",
        "brand",
    ]

    for col in categorical_columns:

        df[col] = (
            df[col]
            .astype("category")
        )

    return df


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# LIGHTGBM
# ============================================================

def train_lightgbm(train):

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
        verbosity=-1,
    )

    model.fit(
        train[FEATURES],
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
# SEASONAL NAIVE
# ============================================================

def seasonal_naive_predictions(
    train,
    validation
):

    # Create lookup from training data:
    # same SKU, 52 weeks earlier.

    lookup = train[
        [
            "sku_id",
            "week_start",
            "units_sold"
        ]
    ].copy()

    lookup["target_week"] = (
        lookup["week_start"]
        + pd.Timedelta(weeks=52)
    )

    lookup = lookup.rename(
        columns={
            "units_sold":
            "naive_prediction"
        }
    )

    validation = validation.merge(
        lookup[
            [
                "sku_id",
                "target_week",
                "naive_prediction"
            ]
        ],
        left_on=[
            "sku_id",
            "week_start"
        ],
        right_on=[
            "sku_id",
            "target_week"
        ],
        how="left"
    )

    validation["naive_prediction"] = (
        validation["naive_prediction"]
        .fillna(0)
    )

    return validation


# ============================================================
# CREATE ROLLING FOLDS
# ============================================================

def create_folds(
    df,
    validation_weeks=12,
    n_folds=4
):

    unique_weeks = np.sort(
        df["week_start"]
        .unique()
    )

    folds = []

    total_weeks = len(unique_weeks)

    for i in range(n_folds):

        validation_end_index = (
            total_weeks
            - i * validation_weeks
        )

        validation_start_index = (
            validation_end_index
            - validation_weeks
        )

        if validation_start_index <= 0:
            break

        validation_weeks_list = (
            unique_weeks[
                validation_start_index:
                validation_end_index
            ]
        )

        train_end = (
            validation_weeks_list[0]
        )

        train = df[
            df["week_start"]
            < train_end
        ].copy()

        validation = df[
            df["week_start"]
            .isin(validation_weeks_list)
        ].copy()

        folds.append(
            (
                i + 1,
                train,
                validation
            )
        )

    folds.reverse()

    return folds


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT — ROLLING ORIGIN BACKTEST")
    print("=" * 70)

    df, sku = load_data()

    print(
        f"\nWeekly records: {len(df):,}"
    )

    print("\nCreating features...")

    df = create_features(
        df,
        sku
    )

    # Need enough history for lag_52
    df = df.dropna(
        subset=[
            "lag_52",
            "rolling_mean_13"
        ]
    ).copy()

    print(
        f"Usable records: {len(df):,}"
    )

    folds = create_folds(
        df,
        validation_weeks=12,
        n_folds=4
    )

    print(
        f"\nNumber of folds: {len(folds)}"
    )

    results = []

    for (
        fold_number,
        train,
        validation
    ) in folds:

        print("\n" + "-" * 70)

        print(
            f"FOLD {fold_number}"
        )

        print(
            f"Train:      {train['week_start'].min().date()} "
            f"→ {train['week_start'].max().date()}"
        )

        print(
            f"Validation: {validation['week_start'].min().date()} "
            f"→ {validation['week_start'].max().date()}"
        )

        # ----------------------------------------------------
        # LightGBM
        # ----------------------------------------------------

        print("\nTraining LightGBM...")

        model = train_lightgbm(
            train
        )

        predictions = model.predict(
            validation[FEATURES]
        )

        predictions = np.maximum(
            predictions,
            0
        )

        # ----------------------------------------------------
        # Seasonal naive
        # ----------------------------------------------------

        naive_validation = (
            seasonal_naive_predictions(
                train,
                validation
            )
        )

        actual = validation[
            "units_sold"
        ]

        # LightGBM metrics
        model_wape = wape(
            actual,
            predictions
        )

        model_mae = mae(
            actual,
            predictions
        )

        model_bias = bias(
            actual,
            predictions
        )

        # Naive metrics
        naive_wape = wape(
            naive_validation[
                "units_sold"
            ],
            naive_validation[
                "naive_prediction"
            ]
        )

        naive_mae = mae(
            naive_validation[
                "units_sold"
            ],
            naive_validation[
                "naive_prediction"
            ]
        )

        naive_bias = bias(
            naive_validation[
                "units_sold"
            ],
            naive_validation[
                "naive_prediction"
            ]
        )

        print(
            f"LightGBM WAPE:       {model_wape:.4f}"
        )

        print(
            f"Seasonal Naive WAPE: {naive_wape:.4f}"
        )

        results.append({

            "fold": fold_number,

            "validation_start":
                validation[
                    "week_start"
                ].min(),

            "validation_end":
                validation[
                    "week_start"
                ].max(),

            "lightgbm_wape":
                model_wape,

            "lightgbm_mae":
                model_mae,

            "lightgbm_bias":
                model_bias,

            "naive_wape":
                naive_wape,

            "naive_mae":
                naive_mae,

            "naive_bias":
                naive_bias,

            "improvement_pct":
                (
                    1
                    - model_wape
                    / naive_wape
                )
                * 100,
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("ROLLING BACKTEST RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Overall averages
    # --------------------------------------------------------

    avg_lightgbm = (
        results_df[
            "lightgbm_wape"
        ].mean()
    )

    avg_naive = (
        results_df[
            "naive_wape"
        ].mean()
    )

    avg_improvement = (
        1
        - avg_lightgbm
        / avg_naive
    ) * 100

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE")
    print("=" * 70)

    print(
        f"LightGBM WAPE:       "
        f"{avg_lightgbm:.4f}"
    )

    print(
        f"Seasonal Naive WAPE: "
        f"{avg_naive:.4f}"
    )

    print(
        f"Improvement:         "
        f"{avg_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        REPORTS_DIR
        / "backtest_results.csv",
        index=False
    )

    model_selection = {
        "selected_model":
            "LightGBM"
            if avg_lightgbm < avg_naive
            else "Seasonal Naive",

        "lightgbm_wape":
            float(avg_lightgbm),

        "seasonal_naive_wape":
            float(avg_naive),

        "improvement_pct":
            float(avg_improvement),
    }

    pd.Series(
        model_selection
    ).to_json(
        REPORTS_DIR
        / "model_selection.json",
        indent=4
    )

    print(
        "\n✓ backtest_results.csv saved"
    )

    print(
        "✓ model_selection.json saved"
    )


if __name__ == "__main__":
    main()