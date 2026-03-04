"""
QuantVision AI — Streamlit Dashboard  (Instant-Load Edition)
=============================================================
Loads the PRE-TRAINED model (best_model.pkl), pre-generated charts
(output/), and the backtest summary CSV — zero retraining.
Loads in seconds, not minutes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, warnings
from PIL import Image

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "output")
DATASET = os.path.join(BASE, "dataset")
MODEL_PATH = os.path.join(BASE, "best_model.pkl")
SUMMARY_CSV = os.path.join(OUTPUT, "company_backtest_summary.csv")

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

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

    .section-divider {
        border: none;
        border-top: 2px solid #203a43;
        margin: 2rem 0;
    }

    .section-header {
        color: #00d4ff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 3px solid #0f3460;
        display: inline-block;
    }

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

# ─── Helpers ────────────────────────────────────────────────────
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


def load_image(path):
    """Load a PNG from disk and return a PIL Image."""
    if os.path.exists(path):
        return Image.open(path)
    return None


# ═══════════════════════════════════════════════════════════════
#  LOAD ALL PRE-BUILT ARTIFACTS (instant — no training)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📂 Loading data …")
def load_dataset():
    fp = os.path.join(DATASET, "NIFTY_50_COMPANIES.csv")
    df = pd.read_csv(fp)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data(show_spinner="📊 Loading backtest summary …")
def load_summary():
    return pd.read_csv(SUMMARY_CSV)


@st.cache_resource(show_spinner="🤖 Loading pre-trained model …")
def load_model():
    return joblib.load(MODEL_PATH)


# Load everything
df_raw = load_dataset()
summary_df = load_summary()
best_model = load_model()
best_model_name = type(best_model).__name__

# ─── Derived info from the dataset ─────────────────────────────
tickers = sorted(df_raw["Ticker"].unique()) if "Ticker" in df_raw.columns else []
company_names = sorted(set(summary_df["Company"].tolist()))
company_folders = sorted([
    d for d in os.listdir(OUTPUT)
    if os.path.isdir(os.path.join(OUTPUT, d)) and d not in ("PORTFOLIO",)
])

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
section("Data Overview", "📊")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Rows × Cols", f"{df_raw.shape[0]:,} × {df_raw.shape[1]}")
with c2:
    metric_card("Unique Tickers", str(len(tickers)))
with c3:
    d_min = df_raw["Date"].min()
    d_max = df_raw["Date"].max()
    metric_card("Date Range", f"{d_min.date()} → {d_max.date()}")
with c4:
    null_rows = df_raw.isnull().any(axis=1).sum()
    metric_card("Rows with NaN", f"{null_rows:,}")

st.write("")
with st.expander("🗂️  Ticker List", expanded=False):
    cols = st.columns(10)
    for i, t in enumerate(tickers):
        cols[i % 10].code(t.replace(".NS", "").replace(".BO", ""))

with st.expander("📋  Raw Data Preview (first 100 rows)", expanded=False):
    st.dataframe(df_raw.head(100), use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE ENGINEERING INFO
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
    for f, desc in feat_desc.items():
        feat_rows += f"<tr><td><code>{f}</code></td><td>{desc}</td></tr>"
    st.markdown(
        f"""<table style="width:100%; font-size:0.9rem;">
        <tr><th style="text-align:left;">Feature</th><th style="text-align:left;">Description</th></tr>
        {feat_rows}
        </table>""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown("**Signal Generation Rule**")
    st.markdown("""
    | Signal | Condition |
    |--------|-----------|
    | **BUY (+1)** | Future Return > +0.50% |
    | **HOLD (0)** | −0.50% ≤ Future Return ≤ +0.50% |
    | **SELL (−1)** | Future Return < −0.50% |
    """)
    st.markdown("**ML Pipeline Steps**")
    st.markdown("""
    1. Load & clean NIFTY 50 historical data
    2. Engineer 9 technical features per ticker
    3. Generate BUY / HOLD / SELL labels (±0.50% threshold)
    4. Walk-Forward Validation (TimeSeriesSplit, 5 folds)
    5. Scale features (StandardScaler — no data leakage)
    6. Train 9 classifiers → pick best by accuracy
    7. Portfolio & per-company backtesting with transaction costs
    """)

# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — PRE-TRAINED MODEL INFO
# ═══════════════════════════════════════════════════════════════
section("Best Model (Pre-Trained)", "🏆")

mc1, mc2, mc3 = st.columns(3)
with mc1:
    metric_card("Saved Model", best_model_name)
with mc2:
    metric_card("Model File", "best_model.pkl")
with mc3:
    metric_card("Companies Tested", f"{len(summary_df)}")

st.write("")

# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — MODEL COMPARISON & PORTFOLIO CHARTS
# ═══════════════════════════════════════════════════════════════
section("Model Comparison & Portfolio Backtest", "💰")

portfolio_dir = os.path.join(OUTPUT, "PORTFOLIO")

# Model comparison chart
model_cmp_img = load_image(os.path.join(portfolio_dir, "model_comparison.png"))
if model_cmp_img:
    st.image(model_cmp_img, caption="Model Accuracy Comparison (all 9 classifiers)", use_container_width=True)

st.write("")

# Portfolio charts in 2 columns
pc1, pc2 = st.columns(2)

with pc1:
    cm_img = load_image(os.path.join(portfolio_dir, "confusion_matrix.png"))
    if cm_img:
        st.image(cm_img, caption=f"Confusion Matrix — {best_model_name}", use_container_width=True)

with pc2:
    cr_img = load_image(os.path.join(portfolio_dir, "cumulative_returns.png"))
    if cr_img:
        st.image(cr_img, caption="Portfolio — Cumulative Returns (Strategy vs Market)", use_container_width=True)

dd_img = load_image(os.path.join(portfolio_dir, "drawdown.png"))
if dd_img:
    st.image(dd_img, caption="Portfolio — Drawdown", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — COMPANY BACKTEST SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
section("Per-Company Backtest Summary", "📋")

display_df = summary_df.copy()
display_df.index = range(1, len(display_df) + 1)

def color_return(val):
    color = "#00e676" if val > 0 else "#ff5252"
    return f"color: {color}; font-weight: bold;"

styled = display_df.style.format({
    "Accuracy": "{:.4f}",
    "Strategy_Return_%": "{:+.2f}%",
    "Market_Return_%": "{:+.2f}%",
    "Sharpe": "{:.4f}",
    "Max_Drawdown_%": "{:.2f}%",
}).map(color_return, subset=["Strategy_Return_%", "Sharpe"])

st.dataframe(styled, use_container_width=True, height=450)

# Quick stats
n_positive = (summary_df["Strategy_Return_%"] > 0).sum()
n_total = len(summary_df)
avg_sharpe = summary_df["Sharpe"].mean()
best_co = summary_df.loc[summary_df["Strategy_Return_%"].idxmax()]
worst_co = summary_df.loc[summary_df["Strategy_Return_%"].idxmin()]

s1, s2, s3, s4 = st.columns(4)
with s1:
    metric_card("Profitable Strategies", f"{n_positive} / {n_total}")
with s2:
    metric_card("Avg Sharpe", f"{avg_sharpe:.4f}")
with s3:
    c = "badge-pos" if best_co["Strategy_Return_%"] >= 0 else "badge-neg"
    metric_card("Best Company", f"<span class='{c}'>{best_co['Company']}<br>{best_co['Strategy_Return_%']:+.1f}%</span>")
with s4:
    c = "badge-neg"
    metric_card("Worst Company", f"<span class='{c}'>{worst_co['Company']}<br>{worst_co['Strategy_Return_%']:+.1f}%</span>")

# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — COMPANY DEEP DIVE (select + show 4 charts)
# ═══════════════════════════════════════════════════════════════
section("Company Deep Dive", "🔍")

selected = st.selectbox(
    "Select a Company",
    company_folders,
    index=company_folders.index("RELIANCE") if "RELIANCE" in company_folders else 0,
)

# Show metrics for selected company
row = summary_df[summary_df["Company"] == selected]
if not row.empty:
    r = row.iloc[0]
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1:
        metric_card("Accuracy", f"{r['Accuracy']:.4f}")
    with cc2:
        c = "badge-pos" if r["Strategy_Return_%"] >= 0 else "badge-neg"
        metric_card("Strategy Return", f"<span class='{c}'>{r['Strategy_Return_%']:+.2f}%</span>")
    with cc3:
        c = "badge-pos" if r["Market_Return_%"] >= 0 else "badge-neg"
        metric_card("Market Return", f"<span class='{c}'>{r['Market_Return_%']:+.2f}%</span>")
    with cc4:
        metric_card("Sharpe Ratio", f"{r['Sharpe']:.4f}")
    with cc5:
        metric_card("Max Drawdown", f"{r['Max_Drawdown_%']:.2f}%")

st.write("")

# 4 charts in 2×2 grid
comp_dir = os.path.join(OUTPUT, selected)

ch1, ch2 = st.columns(2)
with ch1:
    img = load_image(os.path.join(comp_dir, "cumulative_returns.png"))
    if img:
        st.image(img, caption=f"{selected} — Cumulative Returns", use_container_width=True)
    else:
        st.info("cumulative_returns.png not found")

with ch2:
    img = load_image(os.path.join(comp_dir, "drawdown.png"))
    if img:
        st.image(img, caption=f"{selected} — Drawdown", use_container_width=True)
    else:
        st.info("drawdown.png not found")

ch3, ch4 = st.columns(2)
with ch3:
    img = load_image(os.path.join(comp_dir, "signal_distribution.png"))
    if img:
        st.image(img, caption=f"{selected} — Predicted Signals", use_container_width=True)
    else:
        st.info("signal_distribution.png not found")

with ch4:
    img = load_image(os.path.join(comp_dir, "confusion_matrix.png"))
    if img:
        st.image(img, caption=f"{selected} — Confusion Matrix", use_container_width=True)
    else:
        st.info("confusion_matrix.png not found")

# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — TOP & BOTTOM PERFORMERS
# ═══════════════════════════════════════════════════════════════
section("Top & Bottom Performers", "🏅")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

top_n = 10
sdf = summary_df.sort_values("Strategy_Return_%", ascending=False).reset_index(drop=True)

t1, t2 = st.columns(2)

with t1:
    st.markdown("**🟢 Top 10 by Strategy Return**")
    top10 = sdf.head(top_n)[["Company", "Strategy_Return_%", "Sharpe", "Accuracy"]].reset_index(drop=True)
    top10.index = range(1, len(top10) + 1)
    st.dataframe(top10.style.format({
        "Strategy_Return_%": "{:+.2f}%",
        "Sharpe": "{:.3f}",
        "Accuracy": "{:.4f}",
    }), use_container_width=True)

with t2:
    st.markdown("**🔴 Bottom 10 by Strategy Return**")
    bot10 = sdf.tail(top_n)[["Company", "Strategy_Return_%", "Sharpe", "Accuracy"]].reset_index(drop=True)
    bot10.index = range(1, len(bot10) + 1)
    st.dataframe(bot10.style.format({
        "Strategy_Return_%": "{:+.2f}%",
        "Sharpe": "{:.3f}",
        "Accuracy": "{:.4f}",
    }), use_container_width=True)

# Full bar chart
sorted_comp = sdf.sort_values("Strategy_Return_%", ascending=True)
fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
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
#  SECTION 8 — QUICK PREDICT (use saved model)
# ═══════════════════════════════════════════════════════════════
section("Quick Signal Prediction (Live)", "⚡")

st.markdown("Enter feature values to get a **BUY / HOLD / SELL** signal from the pre-trained model:")

features = [
    "Return_Lag1", "Return_Lag2", "Return_Lag3",
    "Volatility_10", "SMA_ratio", "EMA_ratio",
    "MACD_diff", "RSI_norm", "BB_position",
]

defaults = {
    "Return_Lag1": 0.005, "Return_Lag2": -0.002, "Return_Lag3": 0.001,
    "Volatility_10": 0.015, "SMA_ratio": 1.02, "EMA_ratio": 1.01,
    "MACD_diff": 0.5, "RSI_norm": 0.55, "BB_position": 0.6,
}

cols_inp = st.columns(3)
user_vals = {}
for i, feat in enumerate(features):
    with cols_inp[i % 3]:
        user_vals[feat] = st.number_input(feat, value=defaults[feat], format="%.6f")

if st.button("🔮 Predict Signal", type="primary", use_container_width=True):
    input_df = pd.DataFrame([user_vals])
    try:
        prediction = best_model.predict(input_df)[0]
        label_map = {1: ("BUY ↑", "#00e676"), 0: ("HOLD ―", "#90a4ae"), -1: ("SELL ↓", "#ff5252")}
        label, color = label_map.get(prediction, (str(prediction), "#ffffff"))
        st.markdown(
            f"""<div style="text-align:center; padding:1.5rem; background:linear-gradient(145deg,#1a1a2e,#16213e);
            border-radius:12px; margin-top:1rem;">
            <span style="font-size:3rem; color:{color}; font-weight:800;">{label}</span><br>
            <span style="color:#8899aa;">Predicted by {best_model_name}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    """<div style="text-align:center; color:#546e7a; padding:1rem 0;">
    <b>QuantVision AI</b> · Built with Streamlit · Pre-trained model loaded instantly<br>
    <small>Disclaimer: This is for educational purposes only. Not financial advice.</small>
    </div>""",
    unsafe_allow_html=True,
)
