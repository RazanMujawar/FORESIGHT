from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FORESIGHT API",
    description=(
        "Demand forecasting and inventory risk "
        "intelligence API"
    ),
    version="1.0.0",
)


# ============================================================
# LOAD DATA
# ============================================================

RISK_FILE = (
    PROCESSED_DIR / "risk_scoring.parquet"
)


def load_risk_data():

    if not RISK_FILE.exists():

        raise FileNotFoundError(
            f"Risk dataset not found: {RISK_FILE}"
        )

    return pd.read_parquet(
        RISK_FILE
    )


# Load once when API starts
risk = load_risk_data()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "FORESIGHT API",
        "records_loaded": len(risk),
    }


# ============================================================
# SUMMARY
# ============================================================

@app.get("/summary")
def summary():

    risk_counts = (
        risk["risk"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_skus": int(len(risk)),

        "stockout_risk": int(
            risk_counts.get(
                "STOCKOUT RISK",
                0
            )
        ),

        "watch": int(
            risk_counts.get(
                "WATCH",
                0
            )
        ),

        "healthy": int(
            risk_counts.get(
                "HEALTHY",
                0
            )
        ),

        "overstock": int(
            risk_counts.get(
                "OVERSTOCK",
                0
            )
        ),

        "no_inventory_data": int(
            risk_counts.get(
                "NO INVENTORY DATA",
                0
            )
        ),

        "stockout_exposure": float(
            risk[
                "stockout_exposure"
            ].sum()
        ),

        "excess_inventory_value": float(
            risk[
                "excess_inventory_value"
            ].sum()
        ),
    }


# ============================================================
# RISK LIST
# ============================================================

@app.get("/risks")
def risks(
    risk_type: str | None = None,
    category: str | None = None,
    limit: int = 20,
):

    filtered = risk.copy()

    # Filter by risk
    if risk_type:

        filtered = filtered[
            filtered["risk"]
            .str.upper()
            == risk_type.upper()
        ]

    # Filter by category
    if category:

        filtered = filtered[
            filtered["category"]
            .str.lower()
            == category.lower()
        ]

    # Sort by financial impact
    filtered = filtered.sort_values(
        "stockout_exposure",
        ascending=False
    )

    filtered = filtered.head(
        min(limit, 100)
    )

    records = []

    for _, row in filtered.iterrows():

        records.append({

            "sku_id":
                row["sku_id"],

            "sku_name":
                row["sku_name"],

            "category":
                row["category"],

            "risk":
                row["risk"],

            "forecast_weekly_demand":
                float(
                    row[
                        "forecast_weekly_demand"
                    ]
                ),

            "inventory_position":
                float(
                    row[
                        "inventory_position"
                    ]
                ),

            "weeks_of_cover":
                None
                if pd.isna(
                    row["weeks_of_cover"]
                )
                else float(
                    row["weeks_of_cover"]
                ),

            "stockout_exposure":
                float(
                    row[
                        "stockout_exposure"
                    ]
                ),

            "excess_inventory_value":
                float(
                    row[
                        "excess_inventory_value"
                    ]
                ),

            "recommended_action":
                row[
                    "recommended_action"
                ],
        })

    return {
        "count": len(records),
        "results": records,
    }


# ============================================================
# SKU DETAIL
# ============================================================

@app.get("/sku/{sku_id}")
def sku_detail(
    sku_id: str
):

    matches = risk[
        risk["sku_id"].str.upper()
        == sku_id.upper()
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"SKU '{sku_id}' "
                "was not found"
            ),
        )

    row = matches.iloc[0]

    return {

        "sku_id":
            row["sku_id"],

        "sku_name":
            row["sku_name"],

        "category":
            row["category"],

        "subcategory":
            row["subcategory"],

        "brand":
            row["brand"],

        "risk":
            row["risk"],

        "forecast_weekly_demand":
            float(
                row[
                    "forecast_weekly_demand"
                ]
            ),

        "forecast_horizon_demand":
            float(
                row[
                    "forecast_horizon_demand"
                ]
            ),

        "inventory_position":
            float(
                row[
                    "inventory_position"
                ]
            ),

        "weeks_of_cover":
            None
            if pd.isna(
                row["weeks_of_cover"]
            )
            else float(
                row["weeks_of_cover"]
            ),

        "reorder_point":
            float(
                row["reorder_point"]
            ),

        "safety_stock":
            float(
                row["safety_stock"]
            ),

        "stockout_exposure":
            float(
                row["stockout_exposure"]
            ),

        "excess_inventory_value":
            float(
                row[
                    "excess_inventory_value"
                ]
            ),

        "recommended_action":
            row[
                "recommended_action"
            ],
    }