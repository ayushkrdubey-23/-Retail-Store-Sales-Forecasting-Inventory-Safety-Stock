"""
=============================================================================
  AyushKumarDubey_RetailStock_Dashboard.py
  Retail Store Sales Forecasting & Inventory Safety Stock — Streamlit Dashboard
  Author : Ayush Kumar Dubey
  Dataset: retail_store_inventory.csv

  Run:
      python -m streamlit run AyushKumarDubey_RetailStock_Dashboard.py
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RetailStock Dashboard | Ayush Kumar Dubey",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_here   = os.path.dirname(os.path.abspath(__file__))
_local  = os.path.join(_here, "retail_store_inventory.csv")
_parent = os.path.join(_here, "..", "retail_store_inventory.csv")
DATA_PATH = _local if os.path.exists(_local) else _parent

# ── Color palette ─────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Groceries":   "#3b82d4",
    "Toys":        "#f59e0b",
    "Electronics": "#10b981",
    "Furniture":   "#8b5cf6",
    "Clothing":    "#ef4444",
}
REG_COLORS = {"North": "#3b82d4", "South": "#f59e0b", "East": "#10b981", "West": "#8b5cf6"}
STATUS_COLORS = {
    "Stockout":  "#dc2626",
    "Critical":  "#f97316",
    "Low":       "#eab308",
    "Healthy":   "#22c55e",
    "Overstock": "#3b82f6",
}
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#1f2328"),
    margin=dict(t=55, b=30, l=10, r=10),
    legend=dict(orientation="h", y=1.12, x=0),
    xaxis=dict(gridcolor="#e5e7eb", showgrid=True),
    yaxis=dict(gridcolor="#e5e7eb", showgrid=True),
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f7f8fa; }
[data-testid="stMetric"] {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #57606a; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: #1f2328; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px;
    padding: 6px;
}
h1 { color: #1f4e79; }
h2 { color: #1f2328; font-size: 1.15rem !important; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #3b82d4; border-bottom: 3px solid #3b82d4; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LAYER  (all cached so filters don't retrigger heavy work)
# =============================================================================

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load, clean, and type-cast the CSV. Returns a sorted DataFrame."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.rename(columns={
        "Date": "date", "Store ID": "store_id", "Product ID": "product_id",
        "Category": "category", "Region": "region",
        "Inventory Level": "inventory_level", "Units Sold": "units_sold",
        "Units Ordered": "units_ordered", "Demand Forecast": "demand_forecast",
        "Price": "price", "Discount": "discount",
        "Weather Condition": "weather", "Holiday/Promotion": "holiday_promo",
        "Competitor Pricing": "competitor_price", "Seasonality": "seasonality",
    }, inplace=True)

    num_cols = ["inventory_level", "units_sold", "units_ordered",
                "demand_forecast", "price", "discount", "competitor_price", "holiday_promo"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df.dropna(subset=["date", "store_id", "product_id", "units_sold"], inplace=True)
    df = df[df["units_sold"] >= 0]
    df.sort_values(["store_id", "product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner="Engineering features…")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, time, and encoded features for the ML model."""
    d = df.copy()
    d["year"]           = d["date"].dt.year
    d["month"]          = d["date"].dt.month
    d["week"]           = d["date"].dt.isocalendar().week.astype(int)
    d["day_of_week"]    = d["date"].dt.dayofweek
    d["quarter"]        = d["date"].dt.quarter
    d["is_month_end"]   = d["date"].dt.is_month_end.astype(int)
    d["is_month_start"] = d["date"].dt.is_month_start.astype(int)

    grp = d.groupby(["store_id", "product_id"])["units_sold"]
    d["lag_1"]      = grp.shift(1)
    d["lag_7"]      = grp.shift(7)
    d["lag_14"]     = grp.shift(14)
    d["lag_30"]     = grp.shift(30)
    d["roll_7"]     = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    d["roll_14"]    = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    d["roll_30"]    = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    d["roll_7_std"] = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=2).std())

    d["price_ratio"]     = d["price"] / (d["competitor_price"] + 1e-6)
    d["effective_price"] = d["price"] * (1 - d["discount"] / 100)

    d["weather_enc"]  = d["weather"].map({"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snowy": 3}).fillna(1)
    d["season_enc"]   = d["seasonality"].map({"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}).fillna(0)
    d["category_enc"] = LabelEncoder().fit_transform(d["category"].astype(str))
    d["region_enc"]   = LabelEncoder().fit_transform(d["region"].astype(str))
    d["store_enc"]    = LabelEncoder().fit_transform(d["store_id"].astype(str))
    d["product_enc"]  = LabelEncoder().fit_transform(d["product_id"].astype(str))
    d.fillna(0, inplace=True)
    return d


FEATURE_COLS = [
    "year", "month", "week", "day_of_week", "quarter",
    "is_month_end", "is_month_start",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "roll_7", "roll_14", "roll_30", "roll_7_std",
    "price_ratio", "effective_price",
    "weather_enc", "season_enc", "category_enc", "region_enc",
    "store_enc", "product_enc",
    "holiday_promo", "discount", "inventory_level",
]


@st.cache_resource(show_spinner="Training forecasting model…")
def train_model(df_feat: pd.DataFrame):
    """Train a global RandomForest; return (model, metrics_dict)."""
    X = df_feat[FEATURE_COLS].values
    y = df_feat["units_sold"].values
    tscv = TimeSeriesSplit(n_splits=5)
    tr_idx, val_idx = list(tscv.split(X))[-1]

    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X[tr_idx], y[tr_idx])
    y_pred = np.maximum(rf.predict(X[val_idx]), 0)
    y_true = y[val_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nz   = y_true > 1
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.sum() else 0.0

    rf.fit(X, y)   # re-fit on full data
    return rf, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def _z(sl: float) -> float:
    return stats.norm.ppf(sl / 100)


def safety_stock_row(row, z: float) -> dict:
    """Compute SS, ROP, EOQ, days-of-stock and status for one aggregated row."""
    avg_d = row["avg_demand"]
    std_d = row["std_demand"]
    avg_l = row["avg_lead"]
    std_l = row["std_lead"]

    variance = avg_l * std_d**2 + avg_d**2 * std_l**2
    ss  = z * np.sqrt(max(variance, 0))
    rop = avg_d * avg_l + ss

    ann_d = avg_d * 365
    H     = 0.25 * row["avg_price"]
    eoq   = np.sqrt(2 * ann_d * 50 / H) if H > 0 and ann_d > 0 else avg_d * 7

    curr = row["current_inv"]
    dos  = curr / (avg_d + 1e-6)

    if curr == 0:              status = "Stockout"
    elif curr < ss:            status = "Critical"
    elif curr < rop:           status = "Low"
    elif curr > avg_d * 60:    status = "Overstock"
    else:                      status = "Healthy"

    suggest = round(eoq, 0) if status in ("Stockout", "Critical", "Low") else 0

    return dict(
        safety_stock=round(ss, 1), reorder_point=round(rop, 1),
        eoq=round(eoq, 1), days_of_stock=round(dos, 1),
        status=status, suggested_order=suggest,
    )


@st.cache_data(show_spinner="Computing safety stock…")
def compute_safety_stock(df: pd.DataFrame, service_level: float = 95.0) -> pd.DataFrame:
    """Return one row per (store_id, product_id) with all inventory KPIs."""
    z = _z(service_level)
    records = []

    for (sid, pid), grp in df.groupby(["store_id", "product_id"]):
        grp = grp.sort_values("date")
        avg_d = grp["units_sold"].mean()
        std_d = grp["units_sold"].std(ddof=1) if len(grp) > 1 else 1.0

        reorder_dates = grp.loc[grp["units_ordered"] > 0, "date"].sort_values()
        if len(reorder_dates) >= 2:
            gaps  = reorder_dates.diff().dt.days.dropna()
            avg_l = max(float(gaps.mean()), 1)
            std_l = max(float(gaps.std()),  0.5)
        else:
            avg_l, std_l = 7.0, 2.0

        base = dict(
            store_id=sid, product_id=pid,
            category=grp["category"].iloc[-1],
            region=grp["region"].iloc[-1],
            avg_demand=round(avg_d, 2), std_demand=round(std_d, 2),
            avg_lead=round(avg_l, 1), std_lead=round(std_l, 1),
            avg_price=round(grp["price"].mean(), 2),
            current_inv=grp["inventory_level"].iloc[-1],
        )
        base.update(safety_stock_row(base, z))
        records.append(base)

    return pd.DataFrame(records).sort_values("days_of_stock").reset_index(drop=True)


def forecast_one(df_feat: pd.DataFrame, model, store_id: str,
                  product_id: str, horizon: int = 30) -> pd.DataFrame:
    """Generate horizon-day recursive forecast with 90% CI."""
    mask = (df_feat["store_id"] == store_id) & (df_feat["product_id"] == product_id)
    sub  = df_feat[mask].sort_values("date")
    if len(sub) < 5:
        avg   = sub["units_sold"].mean() if len(sub) else 10
        dates = pd.date_range(sub["date"].max() + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame({"date": dates, "forecast": avg,
                              "lower": avg * 0.8, "upper": avg * 1.2})

    last   = sub.iloc[-1]
    sales  = sub["units_sold"].tolist()
    rows   = []

    for step in range(1, horizon + 1):
        fd   = last["date"] + pd.Timedelta(days=step)
        w7   = np.mean(sales[-7:])  if len(sales) >= 7  else np.mean(sales)
        w14  = np.mean(sales[-14:]) if len(sales) >= 14 else np.mean(sales)
        w30  = np.mean(sales[-30:]) if len(sales) >= 30 else np.mean(sales)
        std7 = np.std(sales[-7:])   if len(sales) >= 2  else 0

        feat = np.array([[
            fd.year, fd.month, fd.isocalendar()[1], fd.weekday(),
            (fd.month - 1) // 3 + 1,
            int(pd.Timestamp(fd).is_month_end), int(fd.day == 1),
            sales[-1]  if sales else 0,
            sales[-7]  if len(sales) >= 7  else (sales[0] if sales else 0),
            sales[-14] if len(sales) >= 14 else (sales[0] if sales else 0),
            sales[-30] if len(sales) >= 30 else (sales[0] if sales else 0),
            w7, w14, w30, std7,
            last.get("price_ratio", 1.0), last.get("effective_price", last.get("price", 30)),
            last.get("weather_enc", 1), last.get("season_enc", 0),
            last.get("category_enc", 0), last.get("region_enc", 0),
            last.get("store_enc", 0), last.get("product_enc", 0),
            last.get("holiday_promo", 0), last.get("discount", 0),
            last.get("inventory_level", 100),
        ]])

        pred      = float(np.maximum(model.predict(feat)[0], 0))
        tree_std  = np.array([t.predict(feat)[0] for t in model.estimators_]).std()
        sales.append(pred)
        rows.append({
            "date":     fd,
            "forecast": round(pred, 2),
            "lower":    round(max(pred - 1.64 * tree_std, 0), 2),
            "upper":    round(pred + 1.64 * tree_std, 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# HELPER WIDGETS
# =============================================================================

def kpi_card(col, icon: str, label: str, value: str, delta: str = "", delta_good: bool = True):
    col.metric(label=f"{icon}  {label}", value=value,
               delta=delta if delta else None,
               delta_color="normal" if delta_good else "inverse")


def plotly_chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, key=key)


def section_header(text: str):
    st.markdown(f"### {text}")
    st.markdown('<hr style="border:none;border-top:1px solid #e5e7eb;margin:2px 0 14px">', unsafe_allow_html=True)


# =============================================================================
# LOAD DATA (once, before sidebar uses it)
# =============================================================================
df_raw  = load_data()
df_feat = engineer_features(df_raw)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🏪 RetailStock")
    st.markdown("**Author:** Ayush Kumar Dubey")
    st.markdown("---")

    st.markdown("### 🔎 Filters")

    all_stores = sorted(df_raw["store_id"].unique())
    sel_stores = st.multiselect("Store ID", options=all_stores, default=all_stores)

    all_cats = sorted(df_raw["category"].unique())
    sel_cats = st.multiselect("Product Category", options=all_cats, default=all_cats)

    all_regions = sorted(df_raw["region"].unique())
    sel_regions = st.multiselect("Region", options=all_regions, default=all_regions)

    date_min = df_raw["date"].min().date()
    date_max = df_raw["date"].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    service_level = st.select_slider(
        "Service Level",
        options=[85, 90, 95, 97, 99],
        value=95,
        format_func=lambda x: f"{x}%  (Z={_z(x):.2f})",
    )
    forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, step=7)

    st.markdown("---")
    st.caption(f"Dataset: {date_min} to {date_max}\n"
               f"{df_raw['store_id'].nunique()} stores · "
               f"{df_raw['product_id'].nunique()} products · "
               f"{len(df_raw):,} records")

# ── Apply filters ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = date_range
else:
    d0, d1 = date_min, date_max

mask = (
    df_raw["store_id"].isin(sel_stores) &
    df_raw["category"].isin(sel_cats) &
    df_raw["region"].isin(sel_regions) &
    (df_raw["date"].dt.date >= d0) &
    (df_raw["date"].dt.date <= d1)
)
dff = df_raw[mask].copy()

# ── Safety stock (full dataset, not date-filtered) ────────────────────────────
ss_all  = compute_safety_stock(df_raw, service_level)
ss_filt = ss_all[
    ss_all["store_id"].isin(sel_stores) &
    ss_all["category"].isin(sel_cats)
].copy()


# =============================================================================
# PAGE HEADER
# =============================================================================
st.title("🏪 Retail Store Sales Forecasting & Inventory Safety Stock")
st.caption(
    f"Filtered view — **{len(sel_stores)} store(s)** · "
    f"**{len(sel_cats)} categor{'y' if len(sel_cats)==1 else 'ies'}** · "
    f"**{d0}  →  {d1}**  |  Service Level: **{service_level}%**"
)
st.markdown("---")

# =============================================================================
# KPI CARDS
# =============================================================================
total_rev  = (dff["units_sold"] * dff["price"]).sum()
total_units = int(dff["units_sold"].sum())
avg_inv     = dff["inventory_level"].mean()
n_critical  = int((ss_filt["status"].isin(["Stockout", "Critical"])).sum())
n_low       = int((ss_filt["status"] == "Low").sum())
stockout_r  = (dff["inventory_level"] == 0).mean() * 100
overstock_r = (dff["inventory_level"] > dff["units_sold"].mean() * 30).mean() * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_card(c1, "💰", "Total Revenue",    f"${total_rev/1e6:.1f}M")
kpi_card(c2, "📦", "Units Sold",       f"{total_units:,}")
kpi_card(c3, "🏬", "Avg Inventory",    f"{avg_inv:.0f} units")
kpi_card(c4, "🚨", "Critical / Stockout", str(n_critical),
         f"{n_critical} need action", delta_good=False)
kpi_card(c5, "⚠️", "Below Reorder Pt", str(n_low))
kpi_card(c6, "📈", "Overstock Rate",   f"{overstock_r:.1f}%")

st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview & Trends",
    "📈  Sales Forecasting",
    "🛡️  Safety Stock",
    "🚨  Alerts & Recommendations",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW & TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Monthly revenue + units dual-axis ─────────────────────────────────────
    section_header("Sales Trend Over Time")
    monthly = (
        dff.assign(revenue=dff["units_sold"] * dff["price"])
           .groupby(dff["date"].dt.to_period("M"))
           .agg(units=("units_sold", "sum"), revenue=("revenue", "sum"))
           .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly["date"], y=monthly["revenue"],
               name="Revenue ($)", marker_color="#3b82d4", opacity=0.75),
        secondary_y=False)
    fig_trend.add_trace(
        go.Scatter(x=monthly["date"], y=monthly["units"],
                   name="Units Sold", line=dict(color="#f59e0b", width=2.5),
                   mode="lines+markers"),
        secondary_y=True)
    fig_trend.update_layout(title="Monthly Revenue & Units Sold", height=360,
                             **PLOTLY_LAYOUT)
    fig_trend.update_yaxes(title_text="Revenue ($)", secondary_y=False, gridcolor="#e5e7eb")
    fig_trend.update_yaxes(title_text="Units Sold",  secondary_y=True,  showgrid=False)
    plotly_chart(fig_trend, key="trend")

    # ── Category & Region breakdown ───────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        cat_rev = (dff.assign(rev=dff["units_sold"] * dff["price"])
                     .groupby("category")["rev"].sum().reset_index()
                     .sort_values("rev", ascending=False))
        fig_cat = px.pie(cat_rev, values="rev", names="category",
                         color="category", color_discrete_map=CAT_COLORS,
                         title="Revenue Share by Category", hole=0.45)
        fig_cat.update_traces(textposition="outside", textinfo="percent+label")
        fig_cat.update_layout(height=360, showlegend=False, **{k: v for k, v in PLOTLY_LAYOUT.items()
                                                                if k not in ("xaxis","yaxis")})
        plotly_chart(fig_cat, key="cat_pie")

    with col_b:
        reg = (dff.assign(rev=dff["units_sold"] * dff["price"])
                 .groupby("region")
                 .agg(revenue=("rev", "sum"), units=("units_sold", "sum"))
                 .reset_index())
        fig_reg = px.bar(reg, x="region", y="revenue", color="region",
                         color_discrete_map=REG_COLORS,
                         title="Revenue by Region", text_auto=".2s")
        fig_reg.update_layout(height=360, showlegend=False, **PLOTLY_LAYOUT)
        plotly_chart(fig_reg, key="reg_bar")

    # ── Daily sales trend per category ───────────────────────────────────────
    section_header("Daily Sales Trend by Category")
    daily_cat = (dff.groupby([dff["date"].dt.to_period("W"), "category"])["units_sold"]
                    .sum().reset_index())
    daily_cat["date"] = daily_cat["date"].dt.to_timestamp()
    fig_dc = px.line(daily_cat, x="date", y="units_sold", color="category",
                     color_discrete_map=CAT_COLORS,
                     title="Weekly Units Sold by Category", markers=False)
    fig_dc.update_layout(height=370, **PLOTLY_LAYOUT)
    plotly_chart(fig_dc, key="daily_cat")

    # ── Seasonality + Promo ───────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        season_avg = (dff.groupby("seasonality")["units_sold"].mean()
                        .reindex(["Spring","Summer","Autumn","Winter"]).reset_index())
        fig_sea = px.bar(season_avg, x="seasonality", y="units_sold",
                         color="seasonality",
                         color_discrete_sequence=["#10b981","#f59e0b","#ef4444","#3b82d4"],
                         title="Avg Daily Units Sold by Season", text_auto=".1f")
        fig_sea.update_layout(height=340, showlegend=False, **PLOTLY_LAYOUT)
        plotly_chart(fig_sea, key="season")

    with col_s2:
        promo = (dff.groupby("holiday_promo")["units_sold"]
                   .agg(mean="mean", std="std").reset_index())
        promo["label"] = promo["holiday_promo"].map({0:"Normal Day", 1:"Holiday/Promo"})
        fig_pro = px.bar(promo, x="label", y="mean", error_y="std", color="label",
                         color_discrete_sequence=["#3b82d4","#f59e0b"],
                         title="Avg Units: Normal vs Holiday/Promo", text_auto=".1f")
        fig_pro.update_layout(height=340, showlegend=False, **PLOTLY_LAYOUT)
        plotly_chart(fig_pro, key="promo")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    section_header("Feature Correlation Heatmap")
    num_c = ["units_sold","inventory_level","demand_forecast",
             "price","discount","competitor_price","holiday_promo"]
    corr  = dff[num_c].corr().round(2)
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}", showscale=True,
        colorbar=dict(thickness=12),
    ))
    fig_hm.update_layout(height=400, title="Numeric Feature Correlations",
                          **{k: v for k, v in PLOTLY_LAYOUT.items()
                             if k not in ("xaxis","yaxis","legend","margin")},
                          xaxis=dict(tickangle=-30),
                          margin=dict(t=55, b=20, l=20, r=20))
    plotly_chart(fig_hm, key="heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SALES FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section_header("ML-Powered Demand Forecast (RandomForest)")

    col_fs1, col_fs2 = st.columns([1, 2])
    with col_fs1:
        fc_store = st.selectbox("Select Store", all_stores, key="fc_store")
        fc_prods  = sorted(df_raw[df_raw["store_id"] == fc_store]["product_id"].unique())
        fc_prod   = st.selectbox("Select Product", fc_prods, key="fc_prod")
        st.info(f"Forecasting **{forecast_horizon} days** ahead for **{fc_store} / {fc_prod}**")

    with col_fs2:
        with st.spinner("Training model…"):
            model, metrics = train_model(df_feat)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{metrics['MAE']} units")
        m2.metric("RMSE", f"{metrics['RMSE']} units")
        m3.metric("MAPE", f"{metrics['MAPE']:.1f}%")
        st.caption("Metrics on held-out TimeSeriesSplit validation fold")

    # Generate forecast
    with st.spinner(f"Generating {forecast_horizon}-day forecast…"):
        fc_df = forecast_one(df_feat, model, fc_store, fc_prod, horizon=forecast_horizon)

    hist = (df_raw[(df_raw["store_id"] == fc_store) & (df_raw["product_id"] == fc_prod)]
            .sort_values("date").tail(120))
    meta = df_raw[(df_raw["store_id"] == fc_store) & (df_raw["product_id"] == fc_prod)].iloc[-1]

    st.markdown(f"**{fc_store}** — **{fc_prod}** &nbsp;|&nbsp; "
                f"*{meta['category']}* &nbsp;|&nbsp; *{meta['region']}*")

    # Forecast chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
        y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Interval", showlegend=True,
    ))
    fig_fc.add_trace(go.Scatter(
        x=hist["date"], y=hist["units_sold"],
        mode="lines", name="Historical Sales",
        line=dict(color="#57606a", width=1.5),
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#3b82d4", width=2.5, dash="dash"),
        marker=dict(size=4),
    ))
    fig_fc.add_vline(x=hist["date"].max().isoformat(), line_dash="dot", line_color="#ef4444",
                     annotation_text="Forecast Start", annotation_position="top left")
    fig_fc.update_layout(
        title=f"{forecast_horizon}-Day Sales Forecast — {fc_prod} @ {fc_store}",
        height=420, **PLOTLY_LAYOUT,
        yaxis_title="Units Sold", xaxis_title="Date",
    )
    plotly_chart(fig_fc, key="fc_chart")

    # Forecast table + summary
    col_ft1, col_ft2 = st.columns([2, 1])
    with col_ft1:
        section_header("Forecast Details")
        fdisp = fc_df.copy()
        fdisp["date"]     = fdisp["date"].dt.strftime("%Y-%m-%d")
        fdisp.columns     = ["Date", "Forecast (units)", "Lower (90%)", "Upper (90%)"]
        st.dataframe(fdisp, use_container_width=True, hide_index=True, height=300)

    with col_ft2:
        section_header("Forecast Summary")
        st.metric("Total Forecasted Units", f"{fc_df['forecast'].sum():.0f}")
        st.metric("Avg Daily Demand",        f"{fc_df['forecast'].mean():.1f} units/day")
        peak = fc_df.loc[fc_df["forecast"].idxmax(), "date"].strftime("%b %d")
        low  = fc_df.loc[fc_df["forecast"].idxmin(), "date"].strftime("%b %d")
        st.metric("Peak Demand Day",   peak)
        st.metric("Lowest Demand Day", low)

    # Feature importance
    section_header("Feature Importance")
    fi = (pd.Series(model.feature_importances_, index=FEATURE_COLS)
            .sort_values(ascending=True).tail(15).reset_index())
    fi.columns = ["feature", "importance"]
    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale="Blues",
                    title="Top 15 Features — RandomForest Importance")
    fig_fi.update_layout(height=420, coloraxis_showscale=False,
                          **{**PLOTLY_LAYOUT, "yaxis": dict(autorange="reversed",
                                                             gridcolor="#e5e7eb")})
    plotly_chart(fig_fi, key="fi")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SAFETY STOCK
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section_header(f"Safety Stock & Reorder Points  —  Service Level {service_level}%")

    # KPIs
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Safety Stock",  f"{ss_filt['safety_stock'].sum():,.0f} units")
    s2.metric("Avg Reorder Point",   f"{ss_filt['reorder_point'].mean():.0f} units")
    s3.metric("Critical / Stockout", str(n_critical))
    s4.metric("Below Reorder Point", str(n_low))

    # Status donut + category stacked bar
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st_cnt = ss_filt["status"].value_counts().reset_index()
        st_cnt.columns = ["status", "count"]
        fig_do = px.pie(st_cnt, values="count", names="status",
                        color="status", color_discrete_map=STATUS_COLORS,
                        title="Inventory Health Distribution", hole=0.45)
        fig_do.update_traces(textposition="outside", textinfo="percent+label")
        fig_do.update_layout(height=360, **{k: v for k, v in PLOTLY_LAYOUT.items()
                                            if k not in ("xaxis","yaxis")})
        plotly_chart(fig_do, key="ss_donut")

    with col_d2:
        cst = (ss_filt.groupby(["category","status"]).size()
                      .reset_index(name="count"))
        fig_cst = px.bar(cst, x="category", y="count", color="status",
                         color_discrete_map=STATUS_COLORS,
                         title="Status by Category", barmode="stack")
        fig_cst.update_layout(height=360, **PLOTLY_LAYOUT)
        plotly_chart(fig_cst, key="ss_cat")

    # Safety stock threshold bar chart per product
    section_header("Safety Stock Thresholds per Product")

    top_n = min(20, len(ss_filt))
    worst = ss_filt.head(top_n).copy()   # sorted by days_of_stock asc

    fig_ss = go.Figure()
    fig_ss.add_trace(go.Bar(
        name="Current Inventory",
        x=worst["product_id"] + "<br>" + worst["store_id"],
        y=worst["current_inv"],
        marker_color="#3b82d4", opacity=0.85,
    ))
    fig_ss.add_trace(go.Bar(
        name="Safety Stock",
        x=worst["product_id"] + "<br>" + worst["store_id"],
        y=worst["safety_stock"],
        marker_color="#ef4444", opacity=0.85,
    ))
    fig_ss.add_trace(go.Scatter(
        name="Reorder Point",
        x=worst["product_id"] + "<br>" + worst["store_id"],
        y=worst["reorder_point"],
        mode="markers+lines",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig_ss.update_layout(
        barmode="group",
        title=f"Current Inventory vs Safety Stock Threshold (Bottom {top_n} by Days of Stock)",
        height=430,
        **PLOTLY_LAYOUT,
        yaxis_title="Units",
        xaxis_title="Product / Store",
    )
    plotly_chart(fig_ss, key="ss_bar")

    # Days of stock chart
    section_header("Days of Stock Remaining")
    fig_dos = px.bar(
        worst, y="product_id", x="days_of_stock",
        color="status", color_discrete_map=STATUS_COLORS,
        orientation="h",
        hover_data=["store_id", "category", "current_inv", "safety_stock"],
        title=f"Days of Stock — Bottom {top_n} Products",
    )
    fig_dos.add_vline(x=7, line_dash="dot", line_color="#ef4444",
                     annotation_text="7-day threshold")
    fig_dos.update_layout(height=420, **{**PLOTLY_LAYOUT,
                                         "yaxis": dict(autorange="reversed",
                                                       gridcolor="#e5e7eb")})
    plotly_chart(fig_dos, key="dos")

    # EOQ chart
    section_header("Economic Order Quantity")
    fig_eoq = px.bar(
        ss_filt.nlargest(15,"eoq"),
        y="product_id", x="eoq", color="category",
        color_discrete_map=CAT_COLORS, orientation="h",
        title="EOQ — Top 15 Products",
    )
    fig_eoq.update_layout(height=380, **{**PLOTLY_LAYOUT,
                                         "yaxis": dict(autorange="reversed",
                                                       gridcolor="#e5e7eb")})
    plotly_chart(fig_eoq, key="eoq")

    # Full table
    section_header("Full Safety Stock Table")
    disp_ss = ss_filt[[
        "store_id","product_id","category","region",
        "avg_demand","std_demand","avg_lead",
        "safety_stock","reorder_point","eoq",
        "current_inv","days_of_stock","status","suggested_order"
    ]].copy()
    disp_ss.columns = [
        "Store","Product","Category","Region",
        "Avg Demand/Day","Std Demand","Avg Lead (days)",
        "Safety Stock","Reorder Pt","EOQ",
        "Current Inv","Days of Stock","Status","Suggest Order"
    ]
    for c in ["Avg Demand/Day","Std Demand"]:
        disp_ss[c] = disp_ss[c].map("{:.1f}".format)
    for c in ["Safety Stock","Reorder Pt","EOQ","Current Inv","Suggest Order"]:
        disp_ss[c] = disp_ss[c].map("{:.0f}".format)
    disp_ss["Days of Stock"] = disp_ss["Days of Stock"].map("{:.1f}".format)

    def _style(v):
        m = {"Stockout":"background-color:#fee2e2;color:#dc2626;font-weight:bold",
             "Critical":"background-color:#ffedd5;color:#f97316;font-weight:bold",
             "Low":"background-color:#fef9c3;color:#ca8a04;font-weight:bold",
             "Healthy":"background-color:#dcfce7;color:#15803d",
             "Overstock":"background-color:#dbeafe;color:#1d4ed8"}
        return m.get(v, "")

    st.dataframe(
        disp_ss.style.map(_style, subset=["Status"]),
        use_container_width=True, hide_index=True, height=420,
    )
    st.download_button(
        "⬇️ Download Safety Stock Report (CSV)",
        data=disp_ss.to_csv(index=False).encode("utf-8"),
        file_name="safety_stock_report.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ALERTS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section_header("Inventory Alerts & Reorder Recommendations")

    # Stockout risk score (14-day window using rolling avg demand)
    ss_filt2 = ss_filt.copy()
    ss_filt2["forecast_14d"] = ss_filt2["avg_demand"] * 14
    ss_filt2["stockout_risk_pct"] = (
        (ss_filt2["forecast_14d"] - ss_filt2["current_inv"]) /
        (ss_filt2["forecast_14d"] + 1e-6) * 100
    ).clip(0, 100).round(1)
    ss_filt2["days_until_out"] = (
        ss_filt2["current_inv"] / (ss_filt2["avg_demand"] + 1e-6)
    ).round(1)

    # Alert banner
    a1, a2, a3, a4, a5 = st.columns(5)
    n_so    = int((ss_filt2["status"] == "Stockout").sum())
    n_cr    = int((ss_filt2["status"] == "Critical").sum())
    n_lo    = int((ss_filt2["status"] == "Low").sum())
    n_ov    = int((ss_filt2["status"] == "Overstock").sum())
    tot_ord = int(ss_filt2["suggested_order"].sum())
    a1.metric("🔴 Stockout",        str(n_so), delta="Immediate" if n_so else None, delta_color="inverse")
    a2.metric("🟠 Critical",        str(n_cr), delta="Order now"  if n_cr else None, delta_color="inverse")
    a3.metric("🟡 Below Reorder",   str(n_lo))
    a4.metric("🔵 Overstock Items", str(n_ov))
    a5.metric("📦 Units to Order",  f"{tot_ord:,}")

    # ── Priority alerts ───────────────────────────────────────────────────────
    st.markdown("---")
    section_header("🔴 Priority Alerts — Immediate Action Required")
    crit_df = ss_filt2[ss_filt2["status"].isin(["Stockout","Critical"])].sort_values("days_until_out")

    if len(crit_df):
        for _, row in crit_df.iterrows():
            emoji = "🔴" if row["status"] == "Stockout" else "🟠"
            with st.container(border=True):
                ca, cb, cc, cd = st.columns([2, 2, 2, 1])
                ca.markdown(
                    f"**{emoji} {row['store_id']} / {row['product_id']}**\n\n"
                    f"*{row['category']} · {row['region']}*"
                )
                cb.markdown(
                    f"📦 **Current:** {row['current_inv']:.0f} units\n\n"
                    f"🛡️ **Safety Stock:** {row['safety_stock']:.0f} units\n\n"
                    f"⏱️ **Days Left:** {row['days_until_out']:.1f}"
                )
                cc.markdown(
                    f"📈 **14d Forecast:** {row['forecast_14d']:.0f} units\n\n"
                    f"🔄 **Reorder Point:** {row['reorder_point']:.0f} units\n\n"
                    f"⚡ **Risk:** {row['stockout_risk_pct']:.0f}%"
                )
                cd.metric("Order Now", f"{row['suggested_order']:.0f} units",
                          delta="EOQ-based", delta_color="inverse")
    else:
        st.success("✅ No critical stockout items at the moment!")

    # ── Stockout risk heatmap ─────────────────────────────────────────────────
    st.markdown("---")
    section_header("🗺️ Stockout Risk Heatmap — Store × Category")
    heat = (ss_filt2.groupby(["store_id","category"])["stockout_risk_pct"]
                     .mean().reset_index())
    pivot = heat.pivot(index="category", columns="store_id", values="stockout_risk_pct").fillna(0)

    fig_heatmap = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn_r", zmid=50,
        text=pivot.values.round(1),
        texttemplate="%{text}%", showscale=True,
        colorbar=dict(title="Risk %", thickness=12),
    ))
    fig_heatmap.update_layout(
        title="14-Day Stockout Risk % (avg per Store × Category)",
        height=340,
        **{k: v for k, v in {**PLOTLY_LAYOUT,
                              "xaxis": dict(title="Store"),
                              "yaxis": dict(title="Category"),
                              "legend": dict()}.items()
           if k != "margin"},
        margin=dict(t=55, b=20, l=10, r=10),
    )
    plotly_chart(fig_heatmap, key="risk_hm")

    # ── Reorder timeline ──────────────────────────────────────────────────────
    section_header("📅 Upcoming Reorder Timeline (Next 30 Days)")
    tl = ss_filt2[
        (ss_filt2["days_until_out"] > 0) & (ss_filt2["days_until_out"] <= 30)
    ].copy()
    if len(tl):
        tl["reorder_date"] = pd.Timestamp.today().normalize() + pd.to_timedelta(
            tl["days_until_out"].clip(0).astype(int), unit="D"
        )
        fig_tl = px.scatter(
            tl, x="reorder_date", y="product_id",
            color="status", color_discrete_map=STATUS_COLORS,
            size="suggested_order",
            hover_data=["store_id","category","current_inv","days_until_out"],
            title="Reorder Timeline — Estimated Stockout Dates",
        )
        fig_tl.update_layout(height=380, **{**PLOTLY_LAYOUT,
                                            "xaxis": dict(title="Estimated Stockout Date",
                                                          gridcolor="#e5e7eb"),
                                            "yaxis": dict(autorange="reversed",
                                                          gridcolor="#e5e7eb",
                                                          title="Product")})
        plotly_chart(fig_tl, key="timeline")
    else:
        st.info("No reorder events projected in the next 30 days.")

    # ── Recommended orders table ──────────────────────────────────────────────
    st.markdown("---")
    section_header("📦 Recommended Orders")
    orders = ss_filt2[ss_filt2["suggested_order"] > 0].sort_values("days_until_out")[[
        "store_id","product_id","category","region",
        "current_inv","reorder_point","suggested_order",
        "avg_demand","days_until_out","status",
    ]].copy()
    orders["order_value"] = (orders["suggested_order"] * ss_filt2.loc[orders.index,"avg_price"]).round(0)
    orders.columns = [
        "Store","Product","Category","Region",
        "Current Inv","Reorder Pt","Order Qty",
        "Avg Demand/Day","Days Left","Status","Order Value ($)"
    ]
    orders["Order Value ($)"] = orders["Order Value ($)"].map("${:,.0f}".format)
    orders["Current Inv"]     = orders["Current Inv"].map("{:.0f}".format)
    orders["Reorder Pt"]      = orders["Reorder Pt"].map("{:.0f}".format)
    orders["Order Qty"]       = orders["Order Qty"].map("{:.0f}".format)
    orders["Avg Demand/Day"]  = orders["Avg Demand/Day"].map("{:.1f}".format)
    orders["Days Left"]       = orders["Days Left"].map("{:.1f}".format)

    st.dataframe(
        orders.style.map(_style, subset=["Status"]),
        use_container_width=True, hide_index=True, height=380,
    )
    st.download_button(
        "⬇️ Download Reorder Recommendations (CSV)",
        data=orders.to_csv(index=False).encode("utf-8"),
        file_name="reorder_recommendations.csv",
        mime="text/csv",
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "**RetailStock Dashboard** · Author: Ayush Kumar Dubey · "
    "Built with Streamlit, Plotly, scikit-learn, pandas, scipy · "
    "Dataset: retail_store_inventory.csv (73,100 records)"
)
