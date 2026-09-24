# Retail Store Sales Forecasting & Inventory Safety Stock Optimization

> **Author:** Ayush Kumar Dubey  
> **Dataset:** `retail_store_inventory.csv` (73,100 records · 5 stores · 20 products · Jan 2022 – Jan 2024)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Project Structure](#project-structure)
4. [Technologies & Libraries](#technologies--libraries)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Output Files](#output-files)
8. [Key Features](#key-features)
9. [Safety Stock Formulas](#safety-stock-formulas)
10. [Author](#author)

---

## Project Overview

This project delivers an end-to-end data science pipeline for retail inventory management:

- **Demand Forecasting** — A `RandomForestRegressor` trained on 26 engineered features (lag, rolling, seasonal, weather, pricing) produces 30-day forward demand forecasts with 90% prediction intervals.
- **Safety Stock Optimization** — Industry-standard supply chain formulas calculate safety stock buffers, reorder points, and Economic Order Quantities (EOQ) for every store–product pair.
- **Interactive Streamlit Dashboard** — A four-page web app lets business users explore EDA, run forecasts, tune safety stock parameters, and download actionable reorder recommendations — no coding required.

---

## Dataset Information

| Attribute     | Value                                                        |
|---------------|--------------------------------------------------------------|
| **File**      | `retail_store_inventory.csv`                                 |
| **Source**    | Kaggle — Retail Store Inventory Dataset                      |
| **Records**   | 73,100 rows                                                  |
| **Date Range**| 2022-01-01 to 2024-01-01 (2 years daily)                     |
| **Stores**    | 5 (S001–S005)                                                |
| **Products**  | 20 (P0001–P0020)                                             |
| **Categories**| Groceries, Toys, Electronics, Furniture, Clothing            |
| **Regions**   | North, South, East, West                                     |

**Key columns:** `Date`, `Store ID`, `Product ID`, `Category`, `Region`,
`Inventory Level`, `Units Sold`, `Units Ordered`, `Demand Forecast`, `Price`,
`Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, `Seasonality`

---

## Project Structure

```
retail_forecast/
│
├── AyushKumarDubey_RetailStock.py          ← Standalone end-to-end script
├── requirements.txt                         ← All Python dependencies
├── retail_store_inventory.csv               ← Dataset (place here)
├── README.md
│
├── app.py                                   ← Streamlit landing page
├── run.bat                                  ← Windows double-click launcher
├── run.ps1                                  ← PowerShell launcher
│
├── pages/
│   ├── 01_Overview.py                       ← EDA & KPI dashboard
│   ├── 02_Forecasting.py                    ← ML sales forecasting
│   ├── 03_Safety_Stock.py                   ← Safety stock calculator
│   └── 04_Alerts.py                         ← Inventory alerts & reorder recommendations
│
├── src/
│   ├── data_pipeline.py                     ← Data loading, cleaning, feature engineering
│   ├── forecasting.py                       ← RandomForest model & forecast generation
│   ├── safety_stock.py                      ← Safety stock, EOQ, reorder logic
│   └── utils.py                             ← Shared constants & helpers
│
└── outputs/                                 ← Auto-created on first run
    ├── eda_01_monthly_sales.png
    ├── eda_02_category_revenue.png
    ├── eda_03_region_sales.png
    ├── eda_04_seasonality.png
    ├── eda_05_correlation_heatmap.png
    ├── forecast_feature_importance.png
    ├── forecast_S001_P0001.png
    ├── forecast_S001_P0001.csv
    ├── safety_stock_chart.png
    └── inventory_optimization_report.csv
```

---

## Technologies & Libraries

| Library          | Purpose                                     | Version  |
|------------------|---------------------------------------------|----------|
| `pandas`         | Data loading, cleaning, feature engineering | ≥ 2.0    |
| `numpy`          | Numerical computation                       | ≥ 1.26   |
| `scikit-learn`   | RandomForestRegressor, TimeSeriesSplit, metrics | ≥ 1.4 |
| `scipy`          | Z-score calculation for service levels      | ≥ 1.12   |
| `statsmodels`    | Statistical tests (available for extension) | ≥ 0.14   |
| `matplotlib`     | Static EDA & safety stock charts            | ≥ 3.8    |
| `seaborn`        | Styled statistical visualisations           | ≥ 0.13   |
| `plotly`         | Interactive dashboard charts                | ≥ 5.20   |
| `streamlit`      | Interactive web dashboard frontend          | ≥ 1.32   |
| `joblib`         | Model serialisation                         | ≥ 1.3    |
| `python-docx`    | Project report generation                   | ≥ 1.1    |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or later
- `retail_store_inventory.csv` placed inside the `retail_forecast/` folder

### Step 1 — Clone or download the project

```bash
# If using git
git clone <repo-url>
cd retail_forecast
```

### Step 2 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** Use `python -m pip` (not bare `pip`) if the `pip` command is blocked by your system policy.

### Step 3 — Verify the dataset is in place

```
retail_forecast/
└── retail_store_inventory.csv   ← must be here
```

---

## How to Run

### Option A — Standalone Python Script (no browser required)

Runs the complete pipeline: data cleaning → EDA → feature engineering → model training → safety stock → saves all outputs to `outputs/`.

```bash
cd retail_forecast
python AyushKumarDubey_RetailStock.py
```

Expected console output:
```
=================================================================
  Retail Store Sales Forecasting & Safety Stock Optimization
  Author: Ayush Kumar Dubey
=================================================================
[1/6] Loading & cleaning data …
[2/6] Running EDA …
[3/6] Engineering features …
[4/6] Training demand forecasting model …
[5/6] Computing safety stock & reorder points …
[6/6] Results summary …
```

---

### Option B — Interactive Streamlit Dashboard

#### Windows — double-click launcher
```
Double-click:  retail_forecast\run.bat
```

#### PowerShell
```powershell
cd retail_forecast
.\run.ps1
# or directly:
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

#### Dashboard pages

| Page | URL path | Description |
|------|----------|-------------|
| 🏪 Overview | `/Overview` | KPI cards, revenue trends, category breakdowns, correlation heatmap |
| 📈 Forecasting | `/Forecasting` | Store/product selector, 7–90 day ML forecast with confidence intervals |
| 🛡️ Safety Stock | `/Safety_Stock` | Service-level tuning, EOQ, reorder point analysis, drill-down |
| 🚨 Alerts | `/Alerts` | Priority alerts, stockout risk heatmap, reorder timeline, CSV download |

---

## Output Files

After running `AyushKumarDubey_RetailStock.py`, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `eda_01_monthly_sales.png` | Monthly units sold trend line |
| `eda_02_category_revenue.png` | Revenue share donut chart by category |
| `eda_03_region_sales.png` | Units sold bar chart by region |
| `eda_04_seasonality.png` | Average daily demand by season |
| `eda_05_correlation_heatmap.png` | Annotated numeric feature correlation matrix |
| `forecast_feature_importance.png` | Top 15 feature importances from RandomForest |
| `forecast_S001_P0001.png` | Sample 30-day forecast chart with CI band |
| `forecast_S001_P0001.csv` | Sample forecast data (date, forecast, lower, upper) |
| `safety_stock_chart.png` | Current inventory vs safety stock bar chart |
| `inventory_optimization_report.csv` | Full table: safety stock, ROP, EOQ, status for all 100 pairs |

---

## Key Features

### Demand Forecasting
- **Model:** `RandomForestRegressor` (200 trees, depth 12, TimeSeriesSplit CV)
- **Features:** 26 engineered — lag-1/7/14/30, rolling averages/std, time components, price ratio, weather, seasonality, holiday flags
- **Output:** Point forecast + 90% prediction interval (from tree-ensemble spread)
- **Performance:** MAPE ≈ 12–16% on held-out validation fold

### Safety Stock Optimization
- **Formula:** Combined demand + lead-time variability
- **Service Levels:** 85% to 99.9% (configurable)
- **EOQ:** Wilson formula with configurable ordering and holding costs
- **Status Classification:** Stockout → Critical → Low → Healthy → Overstock

### Streamlit Dashboard
- Fully responsive multi-page layout
- All charts built with Plotly for interactivity (hover, zoom, filter)
- Colour-coded status tables using pandas Styler
- CSV export buttons on Safety Stock and Alerts pages

---

## Safety Stock Formulas

**Combined (Demand + Lead-Time Variability):**
```
SS = Z × sqrt( L_avg × σ_d² + d_avg² × σ_L² )
```

**Reorder Point:**
```
ROP = d_avg × L_avg + SS
```

**Economic Order Quantity (Wilson EOQ):**
```
EOQ = sqrt( 2 × D × S / H )
```

Where:
- `Z` = service-level Z-score (e.g. 1.65 for 95%)
- `L_avg` = average lead time (days)
- `σ_d` = std deviation of daily demand
- `d_avg` = average daily demand
- `σ_L` = std deviation of lead time
- `D` = annual demand (units)
- `S` = ordering cost per order ($)
- `H` = annual holding cost per unit ($)

---

## Author

**Ayush Kumar Dubey**  
Project: Retail Store Sales Forecasting & Inventory Safety Stock Optimization  
Dataset: `retail_store_inventory.csv`  
Stack: Python · pandas · scikit-learn · Plotly · Streamlit · scipy

---

*Built as a complete, production-ready data science project for retail inventory management.*
