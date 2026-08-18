# FORESIGHT — Data Quality & EDA Insight Memo

## 1. Executive Summary

The FORESIGHT dataset contains four years of retail sales activity from January 2022 through December 2025, covering 5,000 SKUs.

The exploratory analysis identified three major characteristics of the business:

1. Demand has increased consistently over the observation period, with revenue growth accelerating in 2025.
2. Demand exhibits strong annual seasonality, with substantially higher demand toward the end of the year.
3. Demand is highly concentrated across SKUs, with a small number of products accounting for a disproportionate share of total volume and revenue.

The analysis also identified an inventory-data coverage limitation: 505 of the 5,000 SKUs do not have corresponding inventory records. These SKUs must therefore be handled carefully during inventory-risk scoring.

---

## 2. Dataset Overview

| Metric | Value |
|---|---:|
| Observation period | 2022-01-01 to 2025-12-31 |
| Unique SKUs | 5,000 |
| Daily SKU records | 4,143,430 |
| SKU master records | 5,000 |
| Inventory SKU records | 4,495 |
| SKUs missing inventory | 505 |
| Fact rows missing inventory | 417,352 |
| Duplicate date-SKU records | 0 |
| Negative units sold | 0 |
| Negative revenue | 0 |

The raw transaction data was transformed through a reproducible pipeline into a daily SKU-level analysis dataset.

---

## 3. Data Quality Findings

### 3.1 Inventory coverage

Inventory information is available for 4,495 of the 5,000 SKUs.

Therefore:

**505 SKUs, or approximately 10.1%, lack inventory information.**

These missing inventory records are not interpreted as zero stock. They are preserved as missing values so that downstream risk calculations do not incorrectly classify unavailable inventory information as a stockout.

### 3.2 Duplicate records

No duplicate `date + sku_id` records were identified in the processed fact table.

### 3.3 Negative values

No negative units sold or negative revenue records were identified.

### 3.4 Data processing

The original transaction data was processed in chunks and aggregated from transaction level to daily SKU level. This reduces the modelling dataset while preserving the demand information required for forecasting.

---

## 4. Demand Trend

Annual demand increased throughout the four-year period.

| Year | Units Sold | Revenue | Unit Growth | Revenue Growth |
|---|---:|---:|---:|---:|
| 2022 | 4.31M | ₹2.58B | — | — |
| 2023 | 4.57M | ₹2.63B | +6.09% | +1.94% |
| 2024 | 4.83M | ₹2.69B | +5.62% | +2.02% |
| 2025 | 4.99M | ₹2.98B | +3.19% | +11.02% |

### Business insight

Unit demand increased every year, while revenue growth accelerated substantially in 2025. This indicates that forecasting demand requires accounting not only for long-term volume trends but also for changes in the revenue generated per unit.

---

## 5. Seasonality

Demand shows a strong annual seasonal pattern.

| Month | Units Sold |
|---|---:|
| January | 1.30M |
| February | 1.22M |
| March | 1.45M |
| April | 1.45M |
| May | 1.53M |
| June | 1.53M |
| July | 1.45M |
| August | 1.60M |
| September | 1.61M |
| October | 1.60M |
| November | 1.85M |
| December | 2.10M |

December represents the highest-demand month, while February represents the lowest.

December demand is approximately 72% higher than February demand.

### Business insight

The strong seasonal pattern indicates that inventory planning should account for time-of-year effects. A forecasting model that ignores annual seasonality could systematically under-forecast demand during the November–December peak and over-forecast demand during weaker periods.

Day-of-week differences were substantially smaller than the month-of-year pattern, suggesting that annual seasonality is more important than weekday effects for this dataset.

---

## 6. Top Movers and Demand Concentration

Demand is highly uneven across SKUs.

The highest-volume SKU, SKU04321, recorded approximately:

- 1.186 million units sold
- ₹1.069 billion revenue

while the median SKU sold substantially fewer units.

### Business insight

The extreme concentration of demand means that forecasting errors for a small number of high-volume SKUs could have a disproportionate effect on overall business performance.

The forecasting stage should therefore evaluate SKU-level performance rather than relying only on aggregate demand accuracy.

---

## 7. Category Performance

Electronics & Accessories generated the highest category revenue, at approximately ₹2.33 billion.

Other high-revenue categories included:

- Apparel & Footwear — approximately ₹1.91B
- Personal Care — approximately ₹1.62B
- Home & Kitchen — approximately ₹1.58B

### Business insight

Revenue is concentrated across a relatively small number of categories. Electronics & Accessories is particularly important from a revenue perspective, while category-level demand differences indicate that inventory and forecasting performance should be monitored across categories rather than treated uniformly.

---

## 8. Inventory Position

Among the 4,495 SKUs with inventory records:

- 1,105 were below their reorder point.
- 3,390 were at or above their reorder point.
- Approximately 24.58% of inventory-covered SKUs were below reorder point.

### Business insight

A meaningful proportion of SKUs with available inventory information are already below their defined reorder point. This supports the need for a forecasting-driven early-warning system rather than relying only on historical inventory thresholds.

The reorder-point condition will be combined with forecast demand and inventory position during the risk-scoring stage.

---

## 9. Dataset Inventory Flags

The supplied dataset also contains inventory-risk reference flags:

| Flag | Count |
|---|---:|
| STOCKOUT_RISK | 200 |
| SLOW_MOVER | 400 |

These flags will not be used as training labels for the forecasting model. They can instead be used as a reference for sanity-checking the resulting risk classification.

---

## 10. Implications for FORESIGHT

The EDA establishes several requirements for the modelling stage:

1. The forecasting model must capture strong annual seasonality.
2. SKU-level forecasting is necessary because demand is highly concentrated.
3. High-volume SKUs require particular attention because forecasting errors can have a larger business impact.
4. Inventory availability must be handled explicitly because 505 SKUs lack inventory information.
5. Forecast performance should be evaluated against a seasonal-naive baseline using time-aware backtesting.
6. Risk scoring should combine forecast demand with the available inventory position rather than relying on historical sales alone.

---

## 11. Limitations and Assumptions

The supplied inventory data does not contain an explicit lead-time field or on-order quantity.

The pipeline therefore does not treat missing lead-time or on-order information as observed client data.

These limitations will be explicitly addressed when constructing the stockout-risk calculation.

The supplied dataset is synthetic and contains deliberately injected inventory-risk cases. The provided inventory flags are therefore treated as reference information for validation rather than as model-training labels.

---

## 12. Conclusion

The EDA confirms that the FORESIGHT problem is appropriate for demand forecasting and inventory-risk scoring.

The strongest patterns are:

- consistent multi-year demand growth,
- substantial annual seasonality,
- highly concentrated SKU-level demand,
- category-level revenue concentration,
- and a meaningful number of SKUs below their reorder point.

These findings provide the basis for the next stage: building and backtesting a weekly SKU-level demand forecast against a seasonal-naive baseline.