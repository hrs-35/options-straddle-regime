# Intraday Regime-Based Short Straddle Strategy

## Overview
This project implements a systematic intraday short straddle strategy on NIFTY ATM options using regime-based filtering.

## Key Features
- IV Rank + Trend-based regime detection
- Volatility compression signal
- Dynamic position sizing
- Realistic cost modeling (STT, brokerage, GST)
- Leg-wise stop loss

## Strategy Logic
- Trade only in low-volatility (GREEN/YELLOW) regimes
- Enter during volatility compression
- Exit via target, stop-loss, timeout, or EOD

## Results
- Profit Factor: X.X
- Max Drawdown: XX%
- Win Rate: XX%

## How to Run

```bash
pip install -r requirements.txt
python main.py
