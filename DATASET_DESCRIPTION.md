# Dataset Description

This document describes all CSV files in the `dataset/` folder, their purpose, and the meaning of each column.

---

## 1. NIFTY_50.csv

**Description:** Historical daily price data and technical indicators for the **NIFTY 50 index** (National Stock Exchange of India's benchmark index comprising 50 large-cap stocks).

- **Rows:** ~4,453 (from 2007-09-17 onwards)
- **Granularity:** One row per trading day

---

## 2. NIFTY_50_COMPANIES.csv

**Description:** Historical daily price data and technical indicators for each of the **50 individual companies** that constitute the NIFTY 50 index. Data is identified by a `Ticker` column (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`). The `.NS` suffix denotes the National Stock Exchange.

- **Rows:** ~304,543
- **Granularity:** One row per company per trading day
- **Companies (51 tickers):** ADANIENT, ADANIGREEN, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BHARTIARTL, BPCL, BRITANNIA, CIPLA, COALINDIA, DIVISLAB, DRREDDY, EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HDFCLIFE, HEROMOTOCO, HINDALCO, HINDUNILVR, ICICIBANK, INDUSINDBK, INFY, ITC, JSWSTEEL, KOTAKBANK, LT, M&M, MARUTI, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE, SBIN, SHREECEM, SUNPHARMA, TATACONSUM, TATAMOTORS, TATASTEEL, TCS, TECHM, TITAN, ULTRACEMCO, UPL, WIPRO

---

## 3. SENSEX.csv

**Description:** Historical daily price data and technical indicators for the **BSE SENSEX index** (Bombay Stock Exchange's benchmark index comprising 30 large-cap stocks).

- **Rows:** ~6,988 (from 1997-07-01 onwards)
- **Granularity:** One row per trading day

---

## 4. SENSEX_COMPANIES.csv

**Description:** Historical daily price data and technical indicators for each of the **30 individual companies** that constitute the SENSEX index. Data is identified by a `Ticker` column (e.g., `RELIANCE.BO`, `TCS.BO`). The `.BO` suffix denotes the Bombay Stock Exchange.

- **Rows:** ~180,327
- **Granularity:** One row per company per trading day
- **Companies (30 tickers):** ASIANPAINT, AXISBANK, BAJAJFINSV, BAJFINANCE, BHARTIARTL, HCLTECH, HDFCBANK, HINDUNILVR, ICICIBANK, INDUSINDBK, INFY, ITC, KOTAKBANK, LT, M&M, MARUTI, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBIN, SUNPHARMA, TATAMOTORS, TATASTEEL, TCS, TECHM, TITAN, ULTRACEMCO, WIPRO

---

## Column Descriptions

All four files share the same set of columns (company files have one extra column: **Ticker**).

### Price & Volume Data

| Column | Meaning |
|---|---|
| **Date** | Trading date in `YYYY-MM-DD` format. |
| **Open** | The price at which the stock/index opened for trading on that day. |
| **High** | The highest price reached during the trading day. |
| **Low** | The lowest price reached during the trading day. |
| **Close** | The official closing price at the end of the trading day. |
| **Adj Close** | The closing price adjusted for corporate actions such as dividends, stock splits, and rights issues. This is the most accurate price for historical analysis. |
| **Volume** | The total number of shares/units traded during the day. A value of `0` typically appears for index-level data where volume is not directly applicable. |
| **Ticker** | *(Company files only)* The stock exchange ticker symbol identifying the company (e.g., `RELIANCE.NS` for NSE, `RELIANCE.BO` for BSE). |

### Technical Indicators

| Column | Full Name | Meaning |
|---|---|---|
| **SMA_20** | Simple Moving Average (20-day) | The average closing price over the last 20 trading days. Used to identify short-term trends. |
| **SMA_50** | Simple Moving Average (50-day) | The average closing price over the last 50 trading days. Used to identify medium-term trends. |
| **EMA_12** | Exponential Moving Average (12-day) | A weighted moving average giving more importance to recent prices (12-day span). Reacts faster to price changes than SMA. |
| **EMA_26** | Exponential Moving Average (26-day) | A weighted moving average giving more importance to recent prices (26-day span). Used alongside EMA_12 to compute MACD. |
| **MACD** | Moving Average Convergence Divergence | The difference between EMA_12 and EMA_26 (`EMA_12 − EMA_26`). A momentum indicator that signals bullish/bearish trends. Positive MACD = bullish momentum; negative = bearish. |
| **Signal_Line** | MACD Signal Line | A 9-day EMA of the MACD value. When MACD crosses above the Signal Line it indicates a potential buy signal; crossing below indicates a potential sell signal. |
| **RSI_14** | Relative Strength Index (14-day) | A momentum oscillator ranging from 0 to 100 measuring the speed and magnitude of price changes. Values above 70 suggest the asset is **overbought**; below 30 suggests it is **oversold**. |
| **BB_Mid** | Bollinger Band – Middle | The 20-day SMA, serving as the centerline of the Bollinger Bands. |
| **BB_Upper** | Bollinger Band – Upper | `BB_Mid + 2 × standard deviation (20-day)`. Price touching or exceeding this band may indicate the asset is overbought. |
| **BB_Lower** | Bollinger Band – Lower | `BB_Mid − 2 × standard deviation (20-day)`. Price touching or falling below this band may indicate the asset is oversold. |
| **Daily_Return_%** | Daily Percentage Return | The percentage change in the **Adj Close** price compared to the previous trading day: `((Today − Yesterday) / Yesterday) × 100`. |

---

## Notes

- Empty/blank values in technical indicator columns (e.g., SMA_20, RSI_14, BB_*) appear at the beginning of each time series because those indicators require a minimum number of historical data points (lookback period) before they can be calculated.
- Index files (`NIFTY_50.csv`, `SENSEX.csv`) have `Volume = 0` because volume is not tracked at the index level.
- The SENSEX dataset has a longer history (starting from **1997**) compared to NIFTY 50 (starting from **2007**).
