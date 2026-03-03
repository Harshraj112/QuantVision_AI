"""
QuantVision AI — Streamlit Dashboard
=====================================
Single-page dashboard that runs the full ML pipeline and displays all outputs:
  • Data overview & cleaning stats
  • Feature engineering preview
  • Model training & comparison
  • Best model evaluation (confusion matrix, classification report)
  • Portfolio-level backtest (cumulative returns, drawdown, risk metrics)
  • Per-company backtest explorer with charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for polished look ──────────────────────────────
st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1.5rem; }
    
    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .hero-banner h1 {
        color: #00d4ff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }
    .hero-banner p {
        color: #b0c4de;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8899aa;
        margin-top: 0.2rem;
    }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 2px solid #203a43;
        margin: 2rem 0;
    }

    /* Styled sub-header */
    .section-header {
        color: #00d4ff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 3px solid #0f3460;
        display: inline-block;
    }

    /* Positive / Negative badge */
    .badge-pos { color: #00e676; font-weight: 700; }
    .badge-neg { color: #ff5252; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Banner ────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>📈 QuantVision AI</h1>
    <p>ML-Powered Trading Signal Engine for Indian Stock Market — NIFTY 50 &amp; SENSEX</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HELPER: nice metric card
# ═══════════════════════════════════════════════════════════════
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title, icon="📌"):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CACHED PIPELINE — runs once, cached across reruns
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="🔄 Running full ML pipeline … hang tight!")
def run_pipeline():
    """Execute the full QuantVision AI pipeline and return all results."""

    BASE = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE, "dataset", "NIFTY_50_COMPANIES.csv")

    # ── Step 1: Load ──────────────────────────────────────────
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    raw_shape = df.shape
    tickers = sorted(df["Ticker"].unique()) if "Ticker" in df.columns else []
    date_range = (df["Date"].min(), df["Date"].max())

    # ── Step 2: Clean ─────────────────────────────────────────
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)

    # ── Step 3: Feature Engineering ───────────────────────────
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Return"] = df.groupby("Ticker")["Adj Close"].pct_change()
    df["Return_Lag1"] = df.groupby("Ticker")["Return"].shift(1)
    df["Return_Lag2"] = df.groupby("Ticker")["Return"].shift(2)
    df["Return_Lag3"] = df.groupby("Ticker")["Return"].shift(3)
    df["Volatility_10"] = (
        df.groupby("Ticker")["Return"].rolling(10).std().reset_index(level=0, drop=True)
    )
    df["SMA_ratio"] = df["Close"] / df["SMA_20"]
    df["EMA_ratio"] = df["Close"] / df["EMA_12"]
    df["MACD_diff"] = df["MACD"] - df["Signal_Line"]
    df["RSI_norm"] = df["RSI_14"] / 100
    bb_width = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_position"] = (df["Close"] - df["BB_Lower"]) / bb_width

    features = [
        "Return_Lag1", "Return_Lag2", "Return_Lag3",
        "Volatility_10", "SMA_ratio", "EMA_ratio",
        "MACD_diff", "RSI_norm", "BB_position",
    ]

    # ── Step 4: Target Labels ────────────────────────────────
    df["Future_Return"] = df.groupby("Ticker")["Return"].shift(-1)
    threshold = 0.005
    df["Signal"] = 0
    df.loc[df["Future_Return"] > threshold, "Signal"] = 1
    df.loc[df["Future_Return"] < -threshold, "Signal"] = -1
    df = df.dropna(subset=["Future_Return"])
    signal_dist = df["Signal"].value_counts().sort_index()

    # ── Step 5: Prepare X, y ─────────────────────────────────
    df = df.dropna(subset=features + ["Signal"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    X = df[features]
    y = df["Signal"]

    # ── Step 6: Walk-Forward Split (last fold) ───────────────
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        pass  # we want the last fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ── Step 7: Scaling ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Step 8: Train Models ─────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (Linear)": SVC(class_weight="balanced", max_iter=5000),
        "SVM (RBF)": SVC(class_weight="balanced", max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
    }

    tree_based = {"Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "HistGradientBoosting"}
    results = {}
    for name, model in models.items():
        if name in tree_based:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        else:
            model.fit(X_train_sc, y_train)
            pred = model.predict(X_test_sc)
        results[name] = accuracy_score(y_test, pred)

    # ── Step 9: Best Model ───────────────────────────────────
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    if best_name in tree_based:
        best_model.fit(X_train, y_train)
        best_pred = best_model.predict(X_test)
    else:
        best_model.fit(X_train_sc, y_train)
        best_pred = best_model.predict(X_test_sc)

    best_acc = accuracy_score(y_test, best_pred)
    cls_report = classification_report(
        y_test, best_pred,
        target_names=["SELL(-1)", "HOLD(0)", "BUY(+1)"],
        labels=[-1, 0, 1],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, best_pred, labels=[-1, 0, 1])

    # ── Step 10: Portfolio Backtest ──────────────────────────
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    bp_series = pd.Series(best_pred).reset_index(drop=True)
    df_test = df_test.iloc[: len(bp_series)].copy()
    df_test["Predicted_Signal"] = bp_series.values
    df_test["Strategy_Return"] = df_test["Predicted_Signal"] * df_test["Future_Return"]
    tc = 0.001
    df_test["Trade"] = df_test.groupby("Ticker")["Predicted_Signal"].diff().abs().fillna(0)
    df_test["Strategy_Return"] -= tc * df_test["Trade"]

    daily_strategy = df_test.groupby("Date")["Strategy_Return"].mean()
    daily_market = df_test.groupby("Date")["Future_Return"].mean()
    cum_strategy = (1 + daily_strategy).cumprod()
    cum_market = (1 + daily_market).cumprod()
    sharpe = (daily_strategy.mean() / daily_strategy.std()) * np.sqrt(252) if daily_strategy.std() != 0 else 0
    roll_max = cum_strategy.cummax()
    drawdown = cum_strategy / roll_max - 1
    max_dd = drawdown.min()
    final_s = cum_strategy.iloc[-1]
    final_m = cum_market.iloc[-1]

    portfolio_metrics = {
        "strategy_return": (final_s - 1) * 100,
        "market_return": (final_m - 1) * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
        "test_start": df_test["Date"].min(),
        "test_end": df_test["Date"].max(),
    }

    portfolio_curves = {
        "cum_strategy": cum_strategy,
        "cum_market": cum_market,
        "drawdown": drawdown,
    }

    # ── Step 11: Per-Company Backtest ────────────────────────
    all_tickers_sorted = sorted(df["Ticker"].unique())
    company_results = []

    for ticker in all_tickers_sorted:
        company_name = ticker.replace(".NS", "").replace(".BO", "")
        df_t = df[df["Ticker"] == ticker].copy().sort_values("Date").reset_index(drop=True)
        if len(df_t) < 100:
            continue
        X_t = df_t[features]
        y_t = df_t["Signal"]
        split = int(len(df_t) * 0.8)
        X_tr, X_te = X_t.iloc[:split], X_t.iloc[split:]
        y_tr, y_te = y_t.iloc[:split], y_t.iloc[split:]

        model_t = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
        )
        model_t.fit(X_tr, y_tr)
        pred_t = model_t.predict(X_te)
        acc_t = (pred_t == y_te.values).mean()

        df_bt = df_t.iloc[split:].copy().reset_index(drop=True)
        df_bt["Predicted_Signal"] = pred_t
        df_bt["Strategy_Return"] = df_bt["Predicted_Signal"] * df_bt["Future_Return"]
        df_bt["Trade"] = df_bt["Predicted_Signal"].diff().abs().fillna(0)
        df_bt["Strategy_Return"] -= tc * df_bt["Trade"]
        df_bt["Cum_Strategy"] = (1 + df_bt["Strategy_Return"]).cumprod()
        df_bt["Cum_Market"] = (1 + df_bt["Future_Return"]).cumprod()

        strat_ret = (df_bt["Cum_Strategy"].iloc[-1] - 1) * 100
        mkt_ret = (df_bt["Cum_Market"].iloc[-1] - 1) * 100
        std_s = df_bt["Strategy_Return"].std()
        sharpe_t = (df_bt["Strategy_Return"].mean() / std_s) * np.sqrt(252) if std_s != 0 else 0
        rm = df_bt["Cum_Strategy"].cummax()
        dd = df_bt["Cum_Strategy"] / rm - 1
        max_dd_t = dd.min()

        cm_t = confusion_matrix(y_te, pred_t, labels=[-1, 0, 1])

        company_results.append({
            "Ticker": ticker,
            "Company": company_name,
            "Accuracy": acc_t,
            "Strategy_Return_%": strat_ret,
            "Market_Return_%": mkt_ret,
            "Sharpe": sharpe_t,
            "Max_Drawdown_%": max_dd_t * 100,
            "df_bt": df_bt,
            "pred_t": pred_t,
            "y_te": y_te,
            "cm": cm_t,
            "dd": dd,
        })

    company_summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("df_bt", "pred_t", "y_te", "cm", "dd")}
        for r in company_results
    ]).sort_values("Strategy_Return_%", ascending=False).reset_index(drop=True)

    return {
        "raw_shape": raw_shape,
        "tickers": tickers,
        "date_range": date_range,
        "rows_dropped": rows_dropped,
        "features": features,
        "signal_dist": signal_dist,
        "X_shape": X.shape,
        "y_shape": y.shape,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_results": results,
        "best_name": best_name,
        "best_acc": best_acc,
        "cls_report": cls_report,
        "cm": cm,
        "portfolio_metrics": portfolio_metrics,
        "portfolio_curves": portfolio_curves,
        "company_results": company_results,
        "company_summary_df": company_summary_df,
    }


# ═══════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ═══════════════════════════════════════════════════════════════
data = run_pipeline()

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
section("Data Overview", "📊")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Rows × Cols", f"{data['raw_shape'][0]:,} × {data['raw_shape'][1]}")
with c2:
    metric_card("Unique Tickers", str(len(data["tickers"])))
with c3:
    metric_card("Date Range", f"{data['date_range'][0].date()} → {data['date_range'][1].date()}")
with c4:
    metric_card("Rows Dropped (NaN)", f"{data['rows_dropped']:,}")

st.write("")
with st.expander("🗂️  Ticker List", expanded=False):
    cols = st.columns(10)
    for i, t in enumerate(data["tickers"]):
        cols[i % 10].code(t.replace(".NS", "").replace(".BO", ""))

# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE ENGINEERING & TARGET LABELS
# ═══════════════════════════════════════════════════════════════
section("Feature Engineering & Target Labels", "🔧")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**9 Engineered Features**")
    feat_desc = {
        "Return_Lag1": "1-day lagged return",
        "Return_Lag2": "2-day lagged return",
        "Return_Lag3": "3-day lagged return",
        "Volatility_10": "10-day rolling std of returns",
        "SMA_ratio": "Close / SMA₂₀",
        "EMA_ratio": "Close / EMA₁₂",
        "MACD_diff": "MACD − Signal Line",
        "RSI_norm": "RSI₁₄ / 100",
        "BB_position": "(Close − BB_Lower) / (BB_Upper − BB_Lower)",
    }
    feat_rows = ""
    for f in data["features"]:
        feat_rows += f"<tr><td><code>{f}</code></td><td>{feat_desc.get(f, '')}</td></tr>"
    st.markdown(
        f"""<table style="width:100%; font-size:0.9rem;">
        <tr><th style="text-align:left;">Feature</th><th style="text-align:left;">Description</th></tr>
        {feat_rows}
        </table>""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown("**Signal Distribution (Target Variable)**")
    label_map = {1: "BUY ↑", 0: "HOLD ―", -1: "SELL ↓"}
    colors = {1: "#00e676", 0: "#90a4ae", -1: "#ff5252"}
    fig_sig, ax_sig = plt.subplots(figsize=(5, 3.5))
    labels_sorted = [-1, 0, 1]
    vals = [data["signal_dist"].get(k, 0) for k in labels_sorted]
    bar_colors = [colors[k] for k in labels_sorted]
    bars = ax_sig.bar([label_map[k] for k in labels_sorted], vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_sig.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    ax_sig.set_ylabel("Count")
    ax_sig.set_title("Target Signal Distribution", fontsize=12, fontweight="bold")
    ax_sig.spines[["top", "right"]].set_visible(False)
    fig_sig.tight_layout()
    st.pyplot(fig_sig)
    plt.close(fig_sig)

    st.caption(f"Threshold: ±0.50% — Final dataset: **{data['X_shape'][0]:,}** rows, **{data['X_shape'][1]}** features")

# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════
section("Model Training & Comparison", "🤖")

st.caption(f"Walk-Forward Validation (last fold) — Train: **{data['train_size']:,}** rows | Test: **{data['test_size']:,}** rows")

sorted_res = dict(sorted(data["model_results"].items(), key=lambda x: x[1], reverse=True))

fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
names = list(sorted_res.keys())
accs = list(sorted_res.values())
bar_palette = ["#00d4ff" if n == data["best_name"] else "#37474f" for n in names]
bars = ax_mc.barh(names[::-1], accs[::-1], color=bar_palette[::-1], edgecolor="white", linewidth=0.3)
for bar, val in zip(bars, accs[::-1]):
    ax_mc.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=10)
ax_mc.set_xlabel("Accuracy", fontsize=11)
ax_mc.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax_mc.spines[["top", "right"]].set_visible(False)
fig_mc.tight_layout()
st.pyplot(fig_mc)
plt.close(fig_mc)

# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — BEST MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════
section("Best Model Evaluation", "🏆")

bc1, bc2, bc3 = st.columns(3)
with bc1:
    metric_card("Best Model", data["best_name"])
with bc2:
    metric_card("Test Accuracy", f"{data['best_acc']:.4f}")
with bc3:
    metric_card("Test Samples", f"{data['test_size']:,}")

st.write("")
ev1, ev2 = st.columns(2)

with ev1:
    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=data["cm"], display_labels=["SELL", "HOLD", "BUY"])
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
    ax_cm.set_title(f"Confusion Matrix — {data['best_name']}", fontsize=12, fontweight="bold")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with ev2:
    st.markdown("**Classification Report**")
    cr = data["cls_report"]
    rows_html = ""
    for label in ["SELL(-1)", "HOLD(0)", "BUY(+1)"]:
        d = cr[label]
        rows_html += f"<tr><td><b>{label}</b></td><td>{d['precision']:.3f}</td><td>{d['recall']:.3f}</td><td>{d['f1-score']:.3f}</td><td>{int(d['support']):,}</td></tr>"
    # overall
    for avg_key in ["macro avg", "weighted avg"]:
        d = cr[avg_key]
        rows_html += f"<tr style='border-top:2px solid #444;'><td><b>{avg_key}</b></td><td>{d['precision']:.3f}</td><td>{d['recall']:.3f}</td><td>{d['f1-score']:.3f}</td><td>{int(d['support']):,}</td></tr>"

    st.markdown(f"""
    <table style="width:100%; text-align:center; font-size:0.95rem;">
    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
    {rows_html}
    </table>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — PORTFOLIO BACKTEST
# ═══════════════════════════════════════════════════════════════
section("Portfolio Backtest", "💰")

pm = data["portfolio_metrics"]
pc = data["portfolio_curves"]

m1, m2, m3, m4 = st.columns(4)
with m1:
    color = "badge-pos" if pm["strategy_return"] >= 0 else "badge-neg"
    metric_card("Strategy Return", f"<span class='{color}'>{pm['strategy_return']:+.2f}%</span>")
with m2:
    color = "badge-pos" if pm["market_return"] >= 0 else "badge-neg"
    metric_card("Market Return", f"<span class='{color}'>{pm['market_return']:+.2f}%</span>")
with m3:
    metric_card("Sharpe Ratio", f"{pm['sharpe']:.4f}")
with m4:
    metric_card("Max Drawdown", f"{pm['max_drawdown']:.2f}%")

st.caption(f"Test Period: **{pm['test_start'].date()}** → **{pm['test_end'].date()}** | Transaction cost: 0.1% per trade side")

# Cumulative Returns
fig_cr, ax_cr = plt.subplots(figsize=(14, 5))
ax_cr.plot(pc["cum_strategy"].index, pc["cum_strategy"].values, label="Strategy", linewidth=1.5, color="#00d4ff")
ax_cr.plot(pc["cum_market"].index, pc["cum_market"].values, label="Market (Buy & Hold)", linewidth=1.5, color="#ff9800", alpha=0.8)
ax_cr.fill_between(pc["cum_strategy"].index, 1, pc["cum_strategy"].values, alpha=0.08, color="#00d4ff")
ax_cr.set_title("Portfolio — Cumulative Returns (Strategy vs Market)", fontsize=14, fontweight="bold")
ax_cr.set_xlabel("Date"); ax_cr.set_ylabel("Growth of ₹1")
ax_cr.legend(fontsize=11); ax_cr.grid(True, alpha=0.2)
ax_cr.spines[["top", "right"]].set_visible(False)
fig_cr.tight_layout()
st.pyplot(fig_cr)
plt.close(fig_cr)

# Drawdown
fig_dd, ax_dd = plt.subplots(figsize=(14, 3))
ax_dd.fill_between(pc["drawdown"].index, pc["drawdown"].values, 0, color="#ff5252", alpha=0.5)
ax_dd.set_title("Portfolio — Drawdown", fontsize=13, fontweight="bold")
ax_dd.set_ylabel("Drawdown"); ax_dd.grid(True, alpha=0.2)
ax_dd.spines[["top", "right"]].set_visible(False)
fig_dd.tight_layout()
st.pyplot(fig_dd)
plt.close(fig_dd)

# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — COMPANY BACKTEST SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
section("Per-Company Backtest Summary", "📋")

summary_df = data["company_summary_df"].copy()
summary_df.index = range(1, len(summary_df) + 1)

# Color code
def color_return(val):
    color = "#00e676" if val > 0 else "#ff5252"
    return f"color: {color}; font-weight: bold;"

styled = summary_df.style.format({
    "Accuracy": "{:.4f}",
    "Strategy_Return_%": "{:+.2f}%",
    "Market_Return_%": "{:+.2f}%",
    "Sharpe": "{:.4f}",
    "Max_Drawdown_%": "{:.2f}%",
}).applymap(color_return, subset=["Strategy_Return_%", "Sharpe"])

st.dataframe(styled, use_container_width=True, height=450)

# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — PER-COMPANY DEEP DIVE
# ═══════════════════════════════════════════════════════════════
section("Company Deep Dive", "🔍")

company_names = [r["Company"] for r in data["company_results"]]
selected_company = st.selectbox("Select a Company", company_names, index=company_names.index("RELIANCE") if "RELIANCE" in company_names else 0)

# Find the result
comp_data = next(r for r in data["company_results"] if r["Company"] == selected_company)

# Metrics row
cc1, cc2, cc3, cc4, cc5 = st.columns(5)
with cc1:
    metric_card("Accuracy", f"{comp_data['Accuracy']:.4f}")
with cc2:
    c = "badge-pos" if comp_data["Strategy_Return_%"] >= 0 else "badge-neg"
    metric_card("Strategy Return", f"<span class='{c}'>{comp_data['Strategy_Return_%']:+.2f}%</span>")
with cc3:
    c = "badge-pos" if comp_data["Market_Return_%"] >= 0 else "badge-neg"
    metric_card("Market Return", f"<span class='{c}'>{comp_data['Market_Return_%']:+.2f}%</span>")
with cc4:
    metric_card("Sharpe Ratio", f"{comp_data['Sharpe']:.4f}")
with cc5:
    metric_card("Max Drawdown", f"{comp_data['Max_Drawdown_%']:.2f}%")

st.write("")

# Charts — 2 columns
ch1, ch2 = st.columns(2)

df_bt = comp_data["df_bt"]

with ch1:
    # Cumulative Returns
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_bt["Date"], df_bt["Cum_Strategy"], label="Strategy", linewidth=1.3, color="#00d4ff")
    ax.plot(df_bt["Date"], df_bt["Cum_Market"], label="Market", linewidth=1.3, color="#ff9800", alpha=0.7)
    ax.set_title(f"{selected_company} — Cumulative Returns", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Growth of ₹1")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with ch2:
    # Drawdown
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(df_bt["Date"], comp_data["dd"].values, 0, color="#ff5252", alpha=0.5)
    ax.set_title(f"{selected_company} — Drawdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown"); ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

ch3, ch4 = st.columns(2)

with ch3:
    # Signal Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    pred_labels = pd.Series(comp_data["pred_t"]).map({-1: "SELL", 0: "HOLD", 1: "BUY"})
    sig_counts = pred_labels.value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
    bar_colors_sig = ["#00e676", "#90a4ae", "#ff5252"]
    b = ax.bar(sig_counts.index, sig_counts.values, color=bar_colors_sig, edgecolor="white", linewidth=0.5)
    for bar, v in zip(b, sig_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{int(v)}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"{selected_company} — Predicted Signals", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with ch4:
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=comp_data["cm"], display_labels=["SELL", "HOLD", "BUY"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{selected_company} — Confusion Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — TOP / BOTTOM PERFORMERS
# ═══════════════════════════════════════════════════════════════
section("Top & Bottom Performers", "🏅")

top_n = 10
sdf = data["company_summary_df"]

t1, t2 = st.columns(2)

with t1:
    st.markdown("**🟢 Top 10 by Strategy Return**")
    top10 = sdf.head(top_n)[["Company", "Strategy_Return_%", "Sharpe", "Accuracy"]].reset_index(drop=True)
    top10.index = range(1, len(top10)+1)
    st.dataframe(top10.style.format({
        "Strategy_Return_%": "{:+.2f}%",
        "Sharpe": "{:.3f}",
        "Accuracy": "{:.4f}",
    }), use_container_width=True)

with t2:
    st.markdown("**🔴 Bottom 10 by Strategy Return**")
    bot10 = sdf.tail(top_n)[["Company", "Strategy_Return_%", "Sharpe", "Accuracy"]].reset_index(drop=True)
    bot10.index = range(1, len(bot10)+1)
    st.dataframe(bot10.style.format({
        "Strategy_Return_%": "{:+.2f}%",
        "Sharpe": "{:.3f}",
        "Accuracy": "{:.4f}",
    }), use_container_width=True)

# Visual: Strategy Return bar chart
fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
sorted_comp = sdf.sort_values("Strategy_Return_%", ascending=True)
colors_bar = ["#00e676" if v > 0 else "#ff5252" for v in sorted_comp["Strategy_Return_%"]]
ax_bar.barh(sorted_comp["Company"], sorted_comp["Strategy_Return_%"], color=colors_bar, edgecolor="white", linewidth=0.3)
ax_bar.axvline(0, color="white", linewidth=0.8, alpha=0.5)
ax_bar.set_xlabel("Strategy Return (%)", fontsize=11)
ax_bar.set_title("All Companies — Strategy Return (%)", fontsize=14, fontweight="bold")
ax_bar.spines[["top", "right"]].set_visible(False)
fig_bar.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    """<div style="text-align:center; color:#546e7a; padding:1rem 0;">
    <b>QuantVision AI</b> · Built with Streamlit · ML-Powered Trading Signal Engine<br>
    <small>Disclaimer: This is for educational purposes only. Not financial advice.</small>
    </div>""",
    unsafe_allow_html=True,
)
