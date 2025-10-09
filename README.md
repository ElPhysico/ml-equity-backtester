# ML Equity Backtester

A modular framework for **building and evaluating ML-driven Top-N equity strategies** with walk-forward validation and benchmark comparison.

This repository demonstrates a complete workflow — from raw equity data to machine-learning predictions and backtesting — using a curated *DEMO15* universe of clean, split-safe stocks.

---

## Quickstart (DEMO15 One-Button Run)

```bash
# 1. Clone the repository
git clone https://github.com/ElPhysico/ml-equity-backtester.git
cd ml-equity-backtester

# 2. Create the environment (Pixi recommended)
pixi install

# 3. Initialize and update the demo universe
pixi run demo:init

# 4. Run the ElasticNet Top-N pipeline (one-button demo)
pixi run demo:elasticnet
