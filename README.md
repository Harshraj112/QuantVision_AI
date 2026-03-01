# QuantVision AI: Feature Engineering and Modeling Pipeline

## Overview

This repository contains the core predictive modeling pipeline for **QuantVision AI**. The project focuses on generating trading signals (BUY, HOLD, SELL) for Indian stock market companies (such as NIFTY 50 and SENSEX constituents) using a suite of technical indicators and machine learning models.

## Project Structure

- `feature_engineering.ipynb`: The main Jupyter notebook outlining the complete end-to-end data processing, feature engineering, modeling, and backtesting pipeline.
- `dataset/`: Directory containing historical stock data. Notable datasets include `NIFTY_50_COMPANIES.csv`.
- `myenv/`: Python virtual environment containing required dependencies.

## Pipeline Steps (Detailed in the Notebook)

### 1. Data Loading and Cleaning

- Historical stock features (Open, High, Low, Close, Adj Close, Volume) along with pre-calculated technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands) are loaded from the datasets.
- Missing values caused by indicator lookback periods are dropped.

### 2. Feature Engineering

The pipeline engineers 9 robust predictive features:

- **Lags**: `Return_Lag1`, `Return_Lag2`, `Return_Lag3`
- **Volatility**: 10-day rolling volatility (`Volatility_10`)
- **Trend and Momentum**: `SMA_ratio`, `EMA_ratio`, `MACD_diff`, Normalized RSI (`RSI_norm`), Bollinger Band Position (`BB_position`)

### 3. Target Label Generation

Future daily returns are calculated, and signals are generated based on a **±0.50% threshold**:

- **BUY (1)**: If future return > 0.50%
- **SELL (-1)**: If future return < -0.50%
- **HOLD (0)**: If return is between -0.50% and 0.50%

### 4. Walk-Forward Validation and Scaling

- To prevent data leakage and handle time-series data appropriately, **Walk-Forward Validation (`TimeSeriesSplit`)** with 5 folds is employed.
- Features are scaled using `StandardScaler` (fit strictly on the training set and applied to the test set).

### 5. Model Training and Evaluation

Several models are evaluated to find the best performer:

- Logistic Regression, KNN, SVM (Linear & RBF)
- Decision Tree, Random Forest
- Gradient Boosting, AdaBoost, HistGradientBoosting

_(In initial runs, Gradient Boosting emerges as the strongest modeling approach)._

### 6. Backtesting and Risk Assessment

The pipeline includes a realistic vectorised backtest module that evaluates the model's trading performance based on:

- Strategy vs Market Returns
- Max Drawdown
- Sharpe Ratio
- Transaction Costs (0.1% factored per trade side)

The notebook also features a dedicated single-ticker backtest (e.g., for `RELIANCE`).

## Dependencies

- pandas, numpy
- scikit-learn
- matplotlib
- joblib

Ensure all requirements are installed in your active environment (`myenv`) before running the notebook.
