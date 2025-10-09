# ML Equity Backtester

This is a small project which I use to explore workflows around **backtesting trading strategies** with a focus on **ML driven strategies**, including:
- Fetching market data via API
- Managing, curating, storing, and preparing data
- Computing features (Lookback-skip momentum, volatility, ...)
- Generating predictions/signals to inform trading strategies
- Backtesting strategies against historical market data
- Evaluating strategies using typical metrics (TR, CAGR, Sharpe, Vol, MaxDD, ...)
- Documenting and reporting results

So far, I've implemented basic Buy-and-Hold, Monthly-Rebalance, and Select-Top-N strategies.

I am using a Pixi for my python environment, but you can use any other environment. However, by using Pixi you can just follow along and run pre-defined Pixi tasks. If you are using any other environment, you can recreate the tasks by running the commands documented in the pixi.toml.

Currently, to fetch market data an Alpha Vantage API key is needed (they provide a free tier). I am planning on providing synthetic market data in a future update such that anyone can run tests without requiring an API key.
    
---

## Quickstart Demo (requires Alpha Vantage API key)

1. Clone the repository and install the environment:
```bash
git clone https://github.com/ElPhysico/ml-equity-backtester.git
cd ml-equity-backtester

pixi install
```

2. Create the dotenv file `.env`
```bash
touch .env
```
and paste your API key inside, using the format `ALPHA_VANTAGE_API_KEY=yourkey`.

3. Download the demo universe (15 tickers):
```bash
pixi run demo_download init
```

4. [Optional] Download benchmarks (2 tickers: CSPX.L (SP500), IWDA.AS (MSCI World)):
```bash
pixi run benchmarks_download init
```

5. Backtest the demo strategy:
```bash
pixi run demo_elasticnet_topn
```

By default the demo uses the config file `config/DEMO15_config.yaml` and you should see an output log similar to this:
```bash
10-10-2025 00:09:39 | INFO | Run ID: 20251009-220938+0000_87c9ac90 | Total return: 392.30% | Sharpe: 0.73 | CAGR: 20.35% | MaxDD: 46.68% | Ann. Volatility: 33.09% | Avg. ann. turnover: 104.83%
10-10-2025 00:09:39 | INFO | Output files saved to outputs/backtests/20251009-220938+0000_87c9ac90
10-10-2025 00:09:39 | INFO | vs BH_EW_DEMO15 -5.20% CAGR, -0.10 Sharpe | vs BH_EW_CSPX.L 5.85% CAGR, -0.15 Sharpe | vs BH_EW_IWDA.AS 9.20% CAGR, 0.04 Sharpe |
```