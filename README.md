# FORESIGHT — Demand Forecasting & Inventory Risk Intelligence

FORESIGHT is an end-to-end demand forecasting and inventory risk decision-support system for retail operations. It converts historical sales and inventory data into weekly SKU-level demand forecasts, identifies stockout and overstock risk, quantifies financial exposure, and recommends inventory actions through an interactive dashboard and API.

## Business Problem

Retail teams face two costly problems:

- **Stockouts:** insufficient inventory can lead to missed sales and poor customer experience.
- **Overstock:** excess inventory ties up working capital and can require markdowns.

FORESIGHT addresses both by combining demand forecasting with current inventory position to help answer:

> **What should we reorder, what is at risk, and where is capital unnecessarily tied up?**

## Solution Overview

1. **Data pipeline** — ingest, clean, validate and unify the source datasets.
2. **EDA & data quality** — identify demand patterns, seasonality, top movers and inventory/data issues.
3. **Demand forecasting** — compare a seasonal-naive baseline with a LightGBM forecasting model.
4. **Inventory risk scoring** — classify SKU-level stockout/overstock risk and estimate rupee exposure.
5. **Decision support** — provide an interactive Streamlit dashboard and FastAPI scoring service.

## Data

The project uses retail datasets covering sales transactions, SKU/product master, store master, customer master, inventory snapshots, promotions and inventory risk flags.

The source sales data covers **2022-01-01 → 2025-12-31**.

Pipeline results:
- **4,143,430** daily SKU records
- **5,000** unique SKUs
- **4,495** inventory SKU records

Raw source data is intentionally not committed to Git because of its size and to keep data dumps out of version control.

## Forecasting

### Seasonal-Naive Baseline

| Metric | Seasonal Naive |
|---|---:|
| WAPE | 37.09% |
| MAE | 8.708 |

### LightGBM

Features include SKU, calendar/seasonality, lagged demand, rolling demand statistics and seasonal lag features.

| Metric | LightGBM |
|---|---:|
| Average rolling-backtest WAPE | **27.81%** |
| Improvement vs baseline | **25.02%** |
| Validation MAE | 6.818 |

Rolling-origin backtesting was used instead of a random train/test split.

| Validation Period | LightGBM WAPE | Seasonal Naive WAPE | Improvement |
|---|---:|---:|---:|
| 2025-02-03 → 2025-04-21 | 26.83% | 36.55% | 26.60% |
| 2025-04-28 → 2025-07-14 | 26.09% | 35.46% | 26.42% |
| 2025-07-21 → 2025-10-06 | 25.87% | 34.90% | 25.89% |
| 2025-10-13 → 2025-12-29 | 32.44% | 41.43% | 21.70% |

## Inventory Risk Scoring

FORESIGHT combines forecast demand with inventory position to classify each SKU into:
- `STOCKOUT RISK`
- `OVERSTOCK`
- `WATCH`
- `HEALTHY`
- `NO INVENTORY DATA`

The risk output includes weeks of cover, lead-time demand, projected inventory, stockout exposure, excess inventory value and recommended action.

Latest scoring output:

| Risk Category | SKU Count |
|---|---:|
| OVERSTOCK | 2,500 |
| HEALTHY | 888 |
| WATCH | 813 |
| NO INVENTORY DATA | 505 |
| STOCKOUT RISK | 294 |

Financial exposure:
- **Stockout exposure:** ₹52.43M
- **Excess inventory value:** ₹1.72B

These are analytical estimates from the supplied project dataset, not live business figures.

## Live Dashboard

**https://foresight-inventory-intelligence.streamlit.app/**

The dashboard provides:
- Executive KPI overview
- Risk distribution
- Financial exposure
- Inventory decisioning grid
- Stockout-risk and overstock views
- SKU drill-down
- Forecast vs actual visualization
- Filters and CSV export
- Beginner-friendly Guide page

## Live Scoring API

**https://foresight-api-aee7.onrender.com/**

Interactive documentation:

**https://foresight-api-aee7.onrender.com/docs**

Example:

```text
GET /sku/SKU04321
```

The API returns forecast and inventory-risk information for the requested SKU. Invalid SKU requests are handled with an HTTP 404 response.

## Repository Structure

```text
FORESIGHT/
├── app/
│   ├── dashboard.py
│   └── pages/
│       └── Guide.py
├── deployment_data/
│   ├── risk_scoring.parquet
│   └── validation_predictions.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_model.ipynb
├── reports/
│   ├── backtest_results.csv
│   ├── baseline_results.csv
│   ├── data_quality_eda_memo.md
│   ├── executive_readout.pptx
│   ├── feature_importance.csv
│   ├── lightgbm_results.csv
│   ├── model_selection.json
│   └── risk_scoring.csv
├── service/
│   └── api.py
├── src/
│   ├── backtest.py
│   ├── check_data.py
│   ├── forecast.py
│   ├── pipeline.py
│   ├── risk_scoring.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Clone the repository:

```bash
git clone https://github.com/RazanMujawar/FORESIGHT.git
cd FORESIGHT
```

Create and activate a virtual environment.

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Place the supplied raw datasets under:

```text
data/raw/
```

The raw data is intentionally excluded from Git.

Run:

```bash
python src/check_data.py
python src/pipeline.py
python src/forecast.py
python src/backtest.py
python src/risk_scoring.py
```

Generated analytical datasets are written to:

```text
data/processed/
```

## Running the Dashboard Locally

```bash
streamlit run app/dashboard.py
```

## Running the API Locally

```bash
uvicorn service.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Reproducibility

The analytical workflow is designed to be rerun from the supplied raw datasets:

```text
check_data.py
      ↓
pipeline.py
      ↓
forecast.py
      ↓
backtest.py
      ↓
risk_scoring.py
```

Headline forecasting result:

**LightGBM average rolling-backtest WAPE: 27.81%**

versus:

**Seasonal-naive WAPE: 37.09%**

## Key Assumptions & Limitations

- Forecasting is performed at the weekly SKU level.
- Rolling-origin validation is used because future observations must not influence historical features.
- Seasonal-naive is the benchmark that the forecasting model must beat.
- Risk scoring depends on the available inventory snapshot and forecast demand.
- SKUs without inventory records are explicitly identified rather than silently treated as zero inventory.
- Forecast accuracy varies across validation periods; the latest validation fold was weaker than earlier folds.
- The system is decision support, not an autonomous ordering system.
- Financial exposure values are estimates derived from the supplied dataset and model assumptions.

## Key Results

> LightGBM reduced average rolling-backtest WAPE from **37.09% to 27.81%**, a **25.02% improvement** over the seasonal-naive baseline.

> 294 SKUs were classified as stockout risk, with approximately **₹52.43M** in estimated stockout exposure.

> 2,500 SKUs were classified as overstock, representing approximately **₹1.72B** in estimated excess inventory value.

## Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/RazanMujawar/FORESIGHT |
| Live Dashboard | https://foresight-inventory-intelligence.streamlit.app/ |
| Live API | https://foresight-api-aee7.onrender.com/ |
| API Documentation | https://foresight-api-aee7.onrender.com/docs |

## Project Deliverables

- **D1:** Reproducible data pipeline
- **D2:** Data-quality & EDA insight memo
- **D3:** Weekly SKU-level demand forecast
- **D4:** Inventory risk scoring and recommended actions
- **D5:** Interactive planning dashboard
- **D6:** Deployed FastAPI scoring service
- **D7:** Executive readout

## Author

**Razan Sameer Mujawar**

FORESIGHT — Data Science & Analytics Project
