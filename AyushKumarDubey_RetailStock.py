"""
=============================================================================
  AyushKumarDubey_RetailStock.py
  Retail Store Sales Forecasting & Inventory Safety Stock Optimization
  Author : Ayush Kumar Dubey
  Dataset: retail_store_inventory.csv
=============================================================================

Pipeline
--------
1. Data Loading & Cleaning
2. Exploratory Data Analysis  (EDA)
3. Feature Engineering
4. Demand Forecasting         (RandomForest + rolling-mean baseline)
5. Safety Stock & Reorder-Point Calculation
6. Results Summary & Export
"""

# -- Standard library ----------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Third-party ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -- Paths ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Locate dataset: check same folder first, then parent folder
_local  = os.path.join(BASE_DIR, "retail_store_inventory.csv")
_parent = os.path.join(BASE_DIR, "..", "retail_store_inventory.csv")
DATA_PATH  = _local if os.path.exists(_local) else _parent
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Plot style ----------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = {
    "Groceries":   "#3b82d4",
    "Toys":        "#f59e0b",
    "Electronics": "#10b981",
    "Furniture":   "#8b5cf6",
    "Clothing":    "#ef4444",
}


# =============================================================================
# SECTION 1 -- DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load retail_store_inventory.csv, rename columns to snake_case,
    coerce numeric types, drop rows with critical nulls, and sort
    by store -> product -> date.

    Returns
    -------
    pd.DataFrame  Clean, typed DataFrame ready for analysis.
    """
    print("\n[1/6] Loading & cleaning data ...")

    df = pd.read_csv(path, parse_dates=["Date"])

    # -- Standardise column names ----------------------------------------------
    col_map = {
        "Date":               "date",
        "Store ID":           "store_id",
        "Product ID":         "product_id",
        "Category":           "category",
        "Region":             "region",
        "Inventory Level":    "inventory_level",
        "Units Sold":         "units_sold",
        "Units Ordered":      "units_ordered",
        "Demand Forecast":    "demand_forecast",
        "Price":              "price",
        "Discount":           "discount",
        "Weather Condition":  "weather",
        "Holiday/Promotion":  "holiday_promo",
        "Competitor Pricing": "competitor_price",
        "Seasonality":        "seasonality",
    }
    df.rename(columns=col_map, inplace=True)

    # -- Drop rows missing critical columns ------------------------------------
    critical = ["date", "store_id", "product_id", "units_sold", "inventory_level"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped {dropped} rows with missing critical values.")

    # -- Enforce numeric types -------------------------------------------------
    num_cols = [
        "inventory_level", "units_sold", "units_ordered",
        "demand_forecast", "price", "discount",
        "competitor_price", "holiday_promo",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -- Remove negative values that make no physical sense -------------------
    df = df[df["units_sold"]      >= 0]
    df = df[df["inventory_level"] >= 0]

    # -- Sort ------------------------------------------------------------------
    df.sort_values(["store_id", "product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"   Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"   Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"   Stores: {df['store_id'].nunique()}  |  Products: {df['product_id'].nunique()}")
    return df


# =============================================================================
# SECTION 2 -- EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Print summary statistics and save EDA visualisations to outputs/.

    Figures produced
    ----------------
    eda_01_monthly_sales.png        Monthly units sold trend
    eda_02_category_revenue.png     Revenue share by category
    eda_03_region_sales.png         Units sold by region
    eda_04_seasonality.png          Avg daily demand per season
    eda_05_correlation_heatmap.png  Numeric feature correlations
    """
    print("\n[2/6] Running EDA ...")

    # -- Console summary -------------------------------------------------------
    print("\n   -- Summary Statistics (numeric columns) --")
    num_summary = df[["units_sold", "inventory_level", "price",
                       "demand_forecast", "discount"]].describe().round(2)
    print(num_summary.to_string())

    revenue = df["units_sold"] * df["price"]
    print(f"\n   Total Revenue  : ${revenue.sum():,.0f}")
    print(f"   Total Units    : {df['units_sold'].sum():,}")
    print(f"   Avg Inventory  : {df['inventory_level'].mean():.1f} units")
    print(f"   Stockout Rate  : {(df['inventory_level'] == 0).mean()*100:.2f}%")

    # -- Fig 1: Monthly units sold ---------------------------------------------
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["units_sold"]
          .sum()
          .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(monthly["date"], monthly["units_sold"], alpha=0.25, color="#3b82d4")
    ax.plot(monthly["date"], monthly["units_sold"], color="#3b82d4", linewidth=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Monthly Units Sold", fontweight="bold")
    ax.set_xlabel("Month");  ax.set_ylabel("Units Sold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_01_monthly_sales.png"), dpi=150)
    plt.close(fig)

    # -- Fig 2: Revenue by category --------------------------------------------
    cat_rev = (
        df.assign(revenue=df["units_sold"] * df["price"])
          .groupby("category")["revenue"].sum()
          .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    wedge_colors = [PALETTE.get(c, "#aaa") for c in cat_rev.index]
    ax.pie(cat_rev, labels=cat_rev.index, autopct="%1.1f%%",
           colors=wedge_colors, startangle=140,
           wedgeprops=dict(width=0.55))
    ax.set_title("Revenue Share by Category", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_02_category_revenue.png"), dpi=150)
    plt.close(fig)

    # -- Fig 3: Units sold by region -------------------------------------------
    reg_sales = df.groupby("region")["units_sold"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    reg_sales.plot(kind="bar", ax=ax, color="#3b82d4", edgecolor="white")
    ax.set_title("Total Units Sold by Region", fontweight="bold")
    ax.set_xlabel("Region");  ax.set_ylabel("Units Sold")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_03_region_sales.png"), dpi=150)
    plt.close(fig)

    # -- Fig 4: Seasonality ----------------------------------------------------
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    season_avg = (
        df.groupby("seasonality")["units_sold"]
          .mean()
          .reindex(season_order)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    season_avg.plot(kind="bar", ax=ax,
                    color=["#10b981", "#f59e0b", "#ef4444", "#3b82d4"],
                    edgecolor="white")
    ax.set_title("Avg Daily Units Sold by Season", fontweight="bold")
    ax.set_xlabel("Season");  ax.set_ylabel("Avg Units Sold")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_04_seasonality.png"), dpi=150)
    plt.close(fig)

    # -- Fig 5: Correlation heatmap --------------------------------------------
    num_cols = ["units_sold", "inventory_level", "demand_forecast",
                "price", "discount", "competitor_price", "holiday_promo"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_05_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    print("   EDA figures saved to outputs/")


# =============================================================================
# SECTION 3 -- FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time, lag, rolling-window, price, and encoded categorical features.

    New columns added
    -----------------
    year, month, week, day_of_week, quarter,
    is_month_end, is_month_start,
    lag_1 / lag_7 / lag_14 / lag_30,
    roll_7 / roll_14 / roll_30 / roll_7_std,
    price_ratio, effective_price,
    weather_enc, season_enc, category_enc, region_enc, store_enc, product_enc
    """
    print("\n[3/6] Engineering features ...")
    df = df.copy()

    # Time features
    df["year"]           = df["date"].dt.year
    df["month"]          = df["date"].dt.month
    df["week"]           = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["quarter"]        = df["date"].dt.quarter
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    # Lag & rolling features (per store-product pair to avoid data leakage)
    grp = df.groupby(["store_id", "product_id"])["units_sold"]
    df["lag_1"]      = grp.shift(1)
    df["lag_7"]      = grp.shift(7)
    df["lag_14"]     = grp.shift(14)
    df["lag_30"]     = grp.shift(30)
    df["roll_7"]     = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df["roll_14"]    = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df["roll_30"]    = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df["roll_7_std"] = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=2).std())

    # Price features
    df["price_ratio"]     = df["price"] / (df["competitor_price"] + 1e-6)
    df["effective_price"] = df["price"] * (1 - df["discount"] / 100)

    # Ordinal encodings
    df["weather_enc"]  = df["weather"].map({"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snowy": 3}).fillna(1)
    df["season_enc"]   = df["seasonality"].map({"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}).fillna(0)
    df["category_enc"] = LabelEncoder().fit_transform(df["category"].astype(str))
    df["region_enc"]   = LabelEncoder().fit_transform(df["region"].astype(str))
    df["store_enc"]    = LabelEncoder().fit_transform(df["store_id"].astype(str))
    df["product_enc"]  = LabelEncoder().fit_transform(df["product_id"].astype(str))

    df.fillna(0, inplace=True)
    print(f"   Total features: {df.shape[1]}")
    return df


FEATURE_COLS = [
    "year", "month", "week", "day_of_week", "quarter",
    "is_month_end", "is_month_start",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "roll_7", "roll_14", "roll_30", "roll_7_std",
    "price_ratio", "effective_price",
    "weather_enc", "season_enc",
    "category_enc", "region_enc", "store_enc", "product_enc",
    "holiday_promo", "discount", "inventory_level",
]


# =============================================================================
# SECTION 4 -- DEMAND FORECASTING
# =============================================================================

def train_forecast_model(df: pd.DataFrame):
    """
    Train a global RandomForestRegressor on all store-product data using
    a TimeSeriesSplit cross-validation strategy.

    Returns
    -------
    model   : fitted RandomForestRegressor
    metrics : dict  {MAE, RMSE, MAPE}  on the held-out validation fold
    """
    print("\n[4/6] Training demand forecasting model ...")

    X = df[FEATURE_COLS].values
    y = df["units_sold"].values

    # Time-series split: train on earlier periods, validate on later
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X))[-1]   # last fold = most recent

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X[train_idx], y[train_idx])

    y_pred = np.maximum(model.predict(X[val_idx]), 0)
    y_true = y[val_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: exclude near-zero actual values to avoid division inflation
    nonzero = y_true > 1
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = 0.0

    print(f"   Validation  MAE : {mae:.2f} units")
    print(f"   Validation  RMSE: {rmse:.2f} units")
    print(f"   Validation  MAPE: {mape:.2f}%")

    # Re-fit on full dataset
    model.fit(X, y)

    # Feature importance plot
    fi = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
          .sort_values(ascending=True)
          .tail(15)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#3b82d4")
    ax.set_title("Top 15 Feature Importances (RandomForest)", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "forecast_feature_importance.png"), dpi=150)
    plt.close(fig)

    return model, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def forecast_product(df: pd.DataFrame, model, store_id: str,
                     product_id: str, horizon: int = 30) -> pd.DataFrame:
    """
    Generate a `horizon`-day ahead point forecast + 90% prediction interval
    for a single store-product pair using recursive multi-step prediction.

    Returns
    -------
    pd.DataFrame  columns: date, forecast, lower, upper
    """
    mask = (df["store_id"] == store_id) & (df["product_id"] == product_id)
    sub  = df[mask].sort_values("date")

    if len(sub) < 5:
        # Fallback: flat rolling mean
        avg   = sub["units_sold"].mean() if len(sub) else 10
        dates = pd.date_range(sub["date"].max() + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame({"date": dates, "forecast": avg,
                              "lower": avg * 0.8, "upper": avg * 1.2})

    last_row   = sub.iloc[-1]
    last_sales = sub["units_sold"].tolist()

    rows = []
    for step in range(1, horizon + 1):
        fd = last_row["date"] + pd.Timedelta(days=step)

        # Build feature vector for this future date
        w7   = np.mean(last_sales[-7:])  if len(last_sales) >= 7  else np.mean(last_sales)
        w14  = np.mean(last_sales[-14:]) if len(last_sales) >= 14 else np.mean(last_sales)
        w30  = np.mean(last_sales[-30:]) if len(last_sales) >= 30 else np.mean(last_sales)
        std7 = np.std(last_sales[-7:])   if len(last_sales) >= 2  else 0

        feat = np.array([[
            fd.year, fd.month, fd.isocalendar()[1], fd.weekday(),
            (fd.month - 1) // 3 + 1,
            int(pd.Timestamp(fd).is_month_end), int(fd.day == 1),
            last_sales[-1]  if last_sales else 0,
            last_sales[-7]  if len(last_sales) >= 7  else (last_sales[0] if last_sales else 0),
            last_sales[-14] if len(last_sales) >= 14 else (last_sales[0] if last_sales else 0),
            last_sales[-30] if len(last_sales) >= 30 else (last_sales[0] if last_sales else 0),
            w7, w14, w30, std7,
            last_row.get("price_ratio",     1.0),
            last_row.get("effective_price", last_row.get("price", 30)),
            last_row.get("weather_enc",     1),
            last_row.get("season_enc",      0),
            last_row.get("category_enc",    0),
            last_row.get("region_enc",      0),
            last_row.get("store_enc",       0),
            last_row.get("product_enc",     0),
            last_row.get("holiday_promo",   0),
            last_row.get("discount",        0),
            last_row.get("inventory_level", 100),
        ]])

        pred      = float(np.maximum(model.predict(feat)[0], 0))
        tree_preds = np.array([t.predict(feat)[0] for t in model.estimators_])
        std_pred  = tree_preds.std()

        last_sales.append(pred)   # roll forward
        rows.append({
            "date":     fd,
            "forecast": round(pred, 2),
            "lower":    round(max(pred - 1.64 * std_pred, 0), 2),
            "upper":    round(pred + 1.64 * std_pred, 2),
        })

    return pd.DataFrame(rows)


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                  store_id: str, product_id: str) -> None:
    """Plot historical actuals + forecast with confidence band."""
    hist = (
        df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]
        .sort_values("date").tail(90)
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hist["date"], hist["units_sold"],
            color="#57606a", linewidth=1.5, label="Historical")
    ax.fill_between(forecast_df["date"],
                    forecast_df["lower"], forecast_df["upper"],
                    alpha=0.2, color="#3b82d4", label="90% CI")
    ax.plot(forecast_df["date"], forecast_df["forecast"],
            color="#3b82d4", linewidth=2.5, linestyle="--", label="Forecast")
    ax.axvline(hist["date"].max(), color="#ef4444",
               linestyle=":", linewidth=1.2, label="Forecast Start")
    ax.set_title(f"30-Day Demand Forecast -- {product_id} @ {store_id}",
                 fontweight="bold")
    ax.set_xlabel("Date");  ax.set_ylabel("Units Sold")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fname = f"forecast_{store_id}_{product_id}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close(fig)


# =============================================================================
# SECTION 5 -- SAFETY STOCK & REORDER POINT
# =============================================================================

def _z_score(service_level_pct: float) -> float:
    """Convert service-level percentage to Z-score (standard normal)."""
    return stats.norm.ppf(service_level_pct / 100)


def safety_stock_combined(avg_demand: float, std_demand: float,
                           avg_lead_time: float, std_lead_time: float,
                           z: float) -> float:
    """
    Combined demand-and-lead-time variability formula.

    SS = Z x ?( L?*?_d? + d??*?_L? )

    Parameters
    ----------
    avg_demand     : mean daily units sold  (d?)
    std_demand     : std-dev of daily demand (?_d)
    avg_lead_time  : mean replenishment lead time in days (L?)
    std_lead_time  : std-dev of lead time (?_L)
    z              : service-level Z-score
    """
    variance = avg_lead_time * std_demand**2 + avg_demand**2 * std_lead_time**2
    return z * np.sqrt(max(variance, 0))


def economic_order_quantity(avg_daily_demand: float, ordering_cost: float = 50.0,
                             holding_pct: float = 0.25, price: float = 30.0) -> float:
    """
    Wilson EOQ formula.   EOQ = ?( 2DS / H )

    Parameters
    ----------
    avg_daily_demand : mean units sold per day
    ordering_cost    : fixed cost per purchase order ($)
    holding_pct      : annual holding cost as fraction of unit price
    price            : unit price ($)
    """
    annual_demand = avg_daily_demand * 365
    holding_cost  = holding_pct * price
    if holding_cost <= 0 or annual_demand <= 0:
        return avg_daily_demand * 7
    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost)


def _lead_time_stats(sub: pd.DataFrame):
    """Estimate avg & std lead time from gaps between reorder events."""
    reorder_dates = sub.loc[sub["units_ordered"] > 0, "date"].sort_values()
    if len(reorder_dates) < 2:
        return 7.0, 2.0
    gaps = reorder_dates.diff().dt.days.dropna()
    return float(max(gaps.mean(), 1)), float(max(gaps.std(), 0.5))


def compute_inventory_optimization(df: pd.DataFrame,
                                    service_level: float = 95.0,
                                    ordering_cost: float = 50.0,
                                    holding_pct: float = 0.25) -> pd.DataFrame:
    """
    For every (store_id, product_id) pair compute:
      - Safety Stock    using combined demand + lead-time variability
      - Reorder Point   = avg_demand x avg_lead_time + safety_stock
      - EOQ             Economic Order Quantity
      - Days of Stock   = current_inventory / avg_demand
      - Status          Stockout / Critical / Low / Healthy / Overstock

    Returns
    -------
    pd.DataFrame  one row per store-product pair, sorted by days_of_stock.
    """
    print("\n[5/6] Computing safety stock & reorder points ...")
    z = _z_score(service_level)
    records = []

    for (store_id, product_id), grp in df.groupby(["store_id", "product_id"]):
        grp = grp.sort_values("date")

        avg_demand  = grp["units_sold"].mean()
        std_demand  = grp["units_sold"].std() if len(grp) > 1 else 1.0
        avg_lt, std_lt = _lead_time_stats(grp)

        ss  = safety_stock_combined(avg_demand, std_demand, avg_lt, std_lt, z)
        rop = avg_demand * avg_lt + ss
        avg_price = grp["price"].mean()
        eoq = economic_order_quantity(avg_demand, ordering_cost, holding_pct, avg_price)

        current_inv  = grp["inventory_level"].iloc[-1]
        days_of_stock = current_inv / (avg_demand + 1e-6)

        # Classify inventory health
        if current_inv == 0:
            status = "Stockout"
        elif current_inv < ss:
            status = "Critical"
        elif current_inv < rop:
            status = "Low"
        elif current_inv > avg_demand * 60:
            status = "Overstock"
        else:
            status = "Healthy"

        suggested_order = round(eoq, 0) if status in ("Stockout", "Critical", "Low") else 0

        records.append({
            "store_id":        store_id,
            "product_id":      product_id,
            "category":        grp["category"].iloc[-1],
            "region":          grp["region"].iloc[-1],
            "avg_daily_demand": round(avg_demand, 2),
            "std_demand":       round(std_demand, 2),
            "avg_lead_time":    round(avg_lt, 1),
            "safety_stock":     round(ss, 1),
            "reorder_point":    round(rop, 1),
            "eoq":              round(eoq, 1),
            "current_inventory": round(current_inv, 0),
            "days_of_stock":    round(days_of_stock, 1),
            "status":           status,
            "suggested_order":  suggested_order,
        })

    result = pd.DataFrame(records).sort_values("days_of_stock")

    print("\n   -- Inventory Health Distribution --")
    print(result["status"].value_counts().to_string())
    print(f"\n   Total Safety Stock Required : {result['safety_stock'].sum():,.0f} units")
    print(f"   Avg Reorder Point           : {result['reorder_point'].mean():.1f} units")

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, "inventory_optimization_report.csv")
    result.to_csv(out_path, index=False)
    print(f"   Report saved -> {out_path}")

    return result


def plot_safety_stock(inv_df: pd.DataFrame) -> None:
    """Bar chart: safety stock vs. current inventory by product (top 20)."""
    top = inv_df.head(20).copy()  # already sorted by days_of_stock asc

    x = np.arange(len(top))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, top["current_inventory"], width,
           label="Current Inventory", color="#3b82d4", alpha=0.85)
    ax.bar(x + width / 2, top["safety_stock"], width,
           label="Safety Stock", color="#ef4444", alpha=0.85)
    ax.plot(x, top["reorder_point"], "D--", color="#f59e0b",
            markersize=6, linewidth=1.5, label="Reorder Point")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['store_id']}\n{r['product_id']}" for _, r in top.iterrows()],
        fontsize=7, rotation=45, ha="right",
    )
    ax.set_title("Current Inventory vs Safety Stock & Reorder Point (Bottom 20 by Days of Stock)",
                 fontweight="bold")
    ax.set_ylabel("Units")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "safety_stock_chart.png"), dpi=150)
    plt.close(fig)
    print("   Safety stock chart saved -> outputs/safety_stock_chart.png")


# =============================================================================
# SECTION 6 -- RESULTS SUMMARY
# =============================================================================

def print_results_summary(metrics: dict, inv_df: pd.DataFrame) -> None:
    """Print a clean final results summary to the console."""
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    print("\n  -- Forecasting Model (RandomForest) --")
    for k, v in metrics.items():
        unit = "%" if k == "MAPE" else "units"
        print(f"     {k:<6}: {v} {unit}")

    print("\n  -- Inventory Optimization --")
    status_counts = inv_df["status"].value_counts()
    for status, count in status_counts.items():
        bar = "#" * count
        print(f"     {status:<12}: {count:3d}  {bar}")

    critical = inv_df[inv_df["status"].isin(["Stockout", "Critical"])]
    if len(critical):
        print(f"\n  *** {len(critical)} product(s) need IMMEDIATE replenishment:")
        for _, r in critical.iterrows():
            print(f"     {r['store_id']} / {r['product_id']}  "
                  f"({r['category']})  --  {r['days_of_stock']:.1f} days left  "
                  f"->  Order {r['suggested_order']:.0f} units")

    print("\n  Output files in ./outputs/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"     {f}")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  Retail Store Sales Forecasting & Safety Stock Optimization")
    print("  Author: Ayush Kumar Dubey")
    print("=" * 65)

    # 1. Load & clean
    df_raw = load_and_clean(DATA_PATH)

    # 2. EDA
    run_eda(df_raw)

    # 3. Feature engineering
    df_feat = engineer_features(df_raw)

    # 4. Train forecasting model
    model, metrics = train_forecast_model(df_feat)

    # Generate a sample forecast for one store-product pair
    sample_store   = df_raw["store_id"].iloc[0]
    sample_product = df_raw["product_id"].iloc[0]
    print(f"\n   Generating 30-day forecast for {sample_store} / {sample_product} ...")
    fc_df = forecast_product(df_feat, model, sample_store, sample_product, horizon=30)
    plot_forecast(df_feat, fc_df, sample_store, sample_product)
    fc_path = os.path.join(OUTPUT_DIR, f"forecast_{sample_store}_{sample_product}.csv")
    fc_df.to_csv(fc_path, index=False)
    print(f"   Forecast saved -> {fc_path}")

    # 5. Safety stock & inventory optimization
    inv_df = compute_inventory_optimization(df_raw, service_level=95.0)
    plot_safety_stock(inv_df)

    # 6. Results summary
    print_results_summary(metrics, inv_df)


if __name__ == "__main__":
    main()
