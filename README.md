# NIFTY 50 Hurst Exponent Trading Strategy

Quantitative trading strategy using the Hurst exponent to identify market regimes and generate trading signals for the NIFTY 50 index.

## Overview

This project implements a machine learning trading strategy based on the Hurst exponent, a statistical measure that classifies market behavior into trending (H > 0.5), mean-reverting (H < 0.5), or random walk (H ≈ 0.5) regimes. A Random Forest classifier generates trading signals using the Hurst exponent and technical indicators.

**Key Result:** The strategy is not profitable on NIFTY 50 test data (2019 onwards), significantly underperforming buy-and-hold with an annual return of -0.32%, Sharpe ratio of -0.01, and maximum drawdown of -21.98%.

## Installation

```bash
git clone https://github.com/YourUsername/nifty50-hurst-strategy.git
cd nifty50-hurst-strategy
pip install -r requirements.txt
python main.py
```

## Methodology

1. Download historical NIFTY 50 data (training: pre-2019, test: 2019+)
2. Calculate rolling Hurst exponent
3. Engineer features including market regime labels and technical indicators
4. Train Random Forest classifier
5. Backtest on out-of-sample data

## Results

The model shows overfitting on training data with poor generalization to the test period. Market regime detection via Hurst exponent does not provide predictive edge for this application.

## Disclaimer

For educational purposes only. Not financial advice.
