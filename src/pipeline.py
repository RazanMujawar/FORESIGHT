from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SALES_CHUNK_SIZE = 500_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """Standardize column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def require_columns(df, required, filename):
    """Check that required columns exist."""
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )


# ============================================================
# LOAD SKU MASTER
# ============================================================

def load_sku_master():
    print("\n[1/5] Loading SKU master...")

    path = RAW_DIR / "sku_master.csv"

    sku = pd.read_csv(path)
    sku = clean_column_names(sku)

    require_columns(
        sku,
        [
            "sku_id",
            "sku_name",
            "category",
            "subcategory",
            "unit_price",
            "cost_price",
            "brand",
        ],
        "sku_master.csv",
    )

    # Remove duplicate SKU records
    sku = sku.drop_duplicates(subset=["sku_id"])

    # Numeric conversion
    sku["unit_price"] = pd.to_numeric(
        sku["unit_price"], errors="coerce"
    )

    sku["cost_price"] = pd.to_numeric(
        sku["cost_price"], errors="coerce"
    )

    print(f"   ✓ SKUs loaded: {len(sku):,}")

    return sku


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory():
    print("\n[2/5] Loading inventory snapshot...")

    path = RAW_DIR / "inventory_snapshot.csv"

    inventory = pd.read_csv(path)
    inventory = clean_column_names(inventory)

    require_columns(
        inventory,
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "last_restock_date",
        ],
        "inventory_snapshot.csv",
    )

    inventory["stock_on_hand"] = pd.to_numeric(
        inventory["stock_on_hand"], errors="coerce"
    ).fillna(0)

    inventory["reorder_point"] = pd.to_numeric(
        inventory["reorder_point"], errors="coerce"
    ).fillna(0)

    inventory["safety_stock"] = pd.to_numeric(
        inventory["safety_stock"], errors="coerce"
    ).fillna(0)

    inventory["last_restock_date"] = pd.to_datetime(
        inventory["last_restock_date"],
        errors="coerce"
    )

    # Aggregate store-level inventory to SKU-level inventory.
    inventory_latest = (
        inventory
        .groupby("sku_id", as_index=False)
        .agg(
            on_hand_units=("stock_on_hand", "sum"),
            reorder_point=("reorder_point", "sum"),
            safety_stock=("safety_stock", "sum"),
        )
    )

    # The source dataset has no explicit on-order quantity.
    inventory_latest["on_order_units"] = 0

    print(
        f"   ✓ SKU inventory records: "
        f"{len(inventory_latest):,}"
    )

    return inventory_latest


# ============================================================
# LOAD PROMOTIONS
# ============================================================

def load_promotions():
    print("\n[3/5] Loading promotions...")

    path = RAW_DIR / "promotions.csv"

    promotions = pd.read_csv(path)
    promotions = clean_column_names(promotions)

    require_columns(
        promotions,
        [
            "promo_id",
            "start_date",
            "end_date",
            "discount_pct",
        ],
        "promotions.csv",
    )

    promotions["start_date"] = pd.to_datetime(
        promotions["start_date"],
        errors="coerce"
    )

    promotions["end_date"] = pd.to_datetime(
        promotions["end_date"],
        errors="coerce"
    )

    promotions["discount_pct"] = pd.to_numeric(
        promotions["discount_pct"],
        errors="coerce"
    ).fillna(0)

    return promotions


# ============================================================
# PROCESS SALES
# ============================================================

def process_sales():
    print("\n[4/5] Processing sales transactions...")
    print(
        f"   Reading in chunks of "
        f"{SALES_CHUNK_SIZE:,} rows..."
    )

    path = RAW_DIR / "sales_transactions.csv"

    required = [
        "date",
        "sku_id",
        "quantity",
        "unit_price",
        "total_value",
        "discount_pct",
        "promo_id",
    ]

    aggregated_chunks = []

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        chunksize=SALES_CHUNK_SIZE,
    ):

        chunk_number += 1

        chunk = clean_column_names(chunk)

        require_columns(
            chunk,
            required,
            "sales_transactions.csv",
        )

        # Date
        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        # Remove invalid rows
        chunk = chunk.dropna(
            subset=["date", "sku_id"]
        )

        # Numeric fields
        chunk["quantity"] = pd.to_numeric(
            chunk["quantity"],
            errors="coerce"
        ).fillna(0)

        chunk["unit_price"] = pd.to_numeric(
            chunk["unit_price"],
            errors="coerce"
        )

        chunk["total_value"] = pd.to_numeric(
            chunk["total_value"],
            errors="coerce"
        ).fillna(0)

        chunk["discount_pct"] = pd.to_numeric(
            chunk["discount_pct"],
            errors="coerce"
        ).fillna(0)

        # Promotion indicator
        chunk["promo_flag"] = (
            chunk["promo_id"].notna()
            & (chunk["promo_id"].astype(str).str.strip() != "")
        ).astype(int)

        # Daily SKU aggregation
        daily = (
            chunk
            .groupby(["date", "sku_id"], as_index=False)
            .agg(
                units_sold=("quantity", "sum"),
                revenue=("total_value", "sum"),
                avg_unit_price=("unit_price", "mean"),
                avg_discount_pct=("discount_pct", "mean"),
                promo_flag=("promo_flag", "max"),
            )
        )

        aggregated_chunks.append(daily)

        print(
            f"   ✓ Processed chunk {chunk_number}"
        )

    print("\n   Combining chunks...")

    sales_daily = pd.concat(
        aggregated_chunks,
        ignore_index=True
    )

    # Because the same date/SKU combination can occur
    # in different chunks, aggregate again.
    sales_daily = (
        sales_daily
        .groupby(
            ["date", "sku_id"],
            as_index=False
        )
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            avg_unit_price=("avg_unit_price", "mean"),
            avg_discount_pct=("avg_discount_pct", "mean"),
            promo_flag=("promo_flag", "max"),
        )
    )

    sales_daily = sales_daily.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    print(
        f"   ✓ Daily SKU records: "
        f"{len(sales_daily):,}"
    )

    print(
        f"   ✓ Date range: "
        f"{sales_daily['date'].min().date()} → "
        f"{sales_daily['date'].max().date()}"
    )

    return sales_daily


# ============================================================
# BUILD CALENDAR
# ============================================================

def build_calendar(start_date, end_date):
    print("\n[5/5] Building calendar...")

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    calendar = pd.DataFrame({
        "date": dates
    })

    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["week_of_year"] = (
        calendar["date"].dt.isocalendar().week.astype(int)
    )
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["is_weekend"] = (
        calendar["day_of_week"] >= 5
    ).astype(int)

    # Simple season classification.
    # We will document this derived field.
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    calendar["season"] = calendar["month"].apply(
        get_season
    )

    return calendar


# ============================================================
# BUILD UNIFIED FACT TABLE
# ============================================================

def build_fact_table(
    sales_daily,
    sku_master,
    inventory_latest,
    calendar,
):
    print("\nBuilding unified fact table...")

    # Add calendar attributes
    fact = sales_daily.merge(
        calendar,
        on="date",
        how="left",
    )

    # Add SKU attributes
    fact = fact.merge(
        sku_master[
            [
                "sku_id",
                "sku_name",
                "category",
                "subcategory",
                "brand",
                "unit_price",
                "cost_price",
            ]
        ],
        on="sku_id",
        how="left",
        suffixes=("", "_master"),
    )

    # Add inventory position
    fact = fact.merge(
        inventory_latest,
        on="sku_id",
        how="left",
    )

    # Track whether inventory information exists
    # before any numerical imputation.
    fact["inventory_data_available"] = (
        fact["on_hand_units"].notna()
    ).astype(int)

    # Convert inventory fields to numeric.
    # IMPORTANT:
    # Do not treat missing inventory as zero.
    inventory_columns = [
        "on_hand_units",
        "on_order_units",
        "reorder_point",
        "safety_stock",
    ]

    for col in inventory_columns:
        fact[col] = pd.to_numeric(
            fact[col],
            errors="coerce"
        )

    # Derived launch date:
    # first observed sales date for each SKU.
    launch_dates = (
        sales_daily
        .groupby("sku_id")["date"]
        .min()
        .rename("launch_date")
    )

    fact = fact.merge(
        launch_dates,
        on="sku_id",
        how="left",
    )

    # Reorder / inventory status
    fact["inventory_above_reorder_point"] = np.where(
    fact["inventory_data_available"] == 1,
    (
        fact["on_hand_units"]
        >= fact["reorder_point"]
    ).astype(int),
    np.nan
    )

    # Sort
    fact = fact.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return fact


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    sales_daily,
    calendar,
    sku_master,
    inventory_latest,
    fact,
):
    print("\nSaving processed datasets...")

    sales_daily.to_parquet(
        PROCESSED_DIR / "sales_daily.parquet",
        index=False,
    )

    calendar.to_parquet(
        PROCESSED_DIR / "calendar.parquet",
        index=False,
    )

    sku_master.to_parquet(
        PROCESSED_DIR / "sku_master.parquet",
        index=False,
    )

    inventory_latest.to_parquet(
        PROCESSED_DIR / "inventory_latest.parquet",
        index=False,
    )

    fact.to_parquet(
        PROCESSED_DIR / "fact.parquet",
        index=False,
    )

    print(
        f"   ✓ sales_daily.parquet"
    )

    print(
        f"   ✓ calendar.parquet"
    )

    print(
        f"   ✓ sku_master.parquet"
    )

    print(
        f"   ✓ inventory_latest.parquet"
    )

    print(
        f"   ✓ fact.parquet"
    )


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def print_quality_summary(
    sales_daily,
    sku_master,
    inventory_latest,
    fact,
):
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"Sales daily rows:       {len(sales_daily):,}"
    )

    print(
        f"Unique SKUs in sales:   "
        f"{sales_daily['sku_id'].nunique():,}"
    )

    print(
        f"SKU master records:     "
        f"{len(sku_master):,}"
    )

    print(
        f"Inventory SKU records:  "
        f"{len(inventory_latest):,}"
    )

    print(
        f"Fact table rows:        "
        f"{len(fact):,}"
    )

    print(
        f"Missing SKU category:   "
        f"{fact['category'].isna().sum():,}"
    )

    missing_inventory_skus = (
    fact.loc[
        fact["inventory_data_available"] == 0,
        "sku_id"
    ]
    .nunique()
    )

    print(
        f"SKUs missing inventory: "
        f"{missing_inventory_skus:,}"
    )

    print(
        f"Fact rows missing inventory: "
        f"{fact['inventory_data_available'].eq(0).sum():,}"
    )

    print(
        f"Duplicate fact rows:    "
        f"{fact.duplicated(['date', 'sku_id']).sum():,}"
    )

    print(
        f"Negative units sold:    "
        f"{(fact['units_sold'] < 0).sum():,}"
    )

    print(
        f"Negative revenue:       "
        f"{(fact['revenue'] < 0).sum():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FORESIGHT DATA PIPELINE")
    print("=" * 70)

    # Load reference datasets
    sku_master = load_sku_master()

    inventory_latest = load_inventory()

    load_promotions()

    # Process 10M+ sales records
    sales_daily = process_sales()

    # Calendar
    calendar = build_calendar(
        sales_daily["date"].min(),
        sales_daily["date"].max(),
    )

    # Unified dataset
    fact = build_fact_table(
        sales_daily,
        sku_master,
        inventory_latest,
        calendar,
    )

    # Quality report
    print_quality_summary(
        sales_daily,
        sku_master,
        inventory_latest,
        fact,
    )

    # Save
    save_outputs(
        sales_daily,
        calendar,
        sku_master,
        inventory_latest,
        fact,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()