# ML Equity Backtester — a learning journey in quant research

> From “how do I backtest a strategy?” to **building a market simulator**, studying **equity theory**, and **stress‑testing** strategies across regimes.

I started this project to learn workflows around **backtesting** and **machine learning for equity strategies**. Along the way I had to wrangle APIs, data management, metadata, docs, and reproducibility. The scope then expanded: I’m now **studying price theory**, **generating synthetic market data**, and **evaluating strategies** in controlled scenarios to build intuition and basic knowledge.

---

## What’s inside (at a glance)

- **Backtesting framework** for ML‑based and rule‑based strategies (walk‑forward).  
- **Built‑in synthetic data simulator** (GBM baseline; correlation, fat tails, regimes upcoming).  
- **Strategies:** Buy‑and‑Hold (BH), Monthly Rebalance (MR), ElasticNet Top‑N (ML), and more coming.  
- **Reproducibility:** per‑run metadata, deterministic seeds, artifacts.  
- **Notebooks:** 
  - Theory: derivations and intuition — see `notebooks/theory/`
    - Start here: [`01_gbm.ipynb`](notebooks/theory/01_gbm.ipynb)
  - Tutorials: learn how to use my framework for yourself - see `notebooks/tutorials/`
    - Start here: [`01_simulating_universes.ipynb`](notebooks/tutorials/01_simulating_universes.ipynb)
  <!-- - Insights: strategy behavior in different environments — see `notebooks/insights/`
    - Start here: [`01_gbm_regimes_overview.ipynb`](notebooks/insights/01_gbm_regimes_overview.ipynb) -->

---

## The simulator (why & how)

To make experiments reproducible and API‑free, I generate **synthetic price universes**. The baseline is **Geometric Brownian Motion (GBM)**:

$$
dS_t = \mu S_tdt + \sigma S_tdW_t
$$

which implies **lognormal prices**,
$\mathbb{E}[S_t]=S_0 e^{\mu t}$ and a **typical (geometric) growth** of
$g=\mu-\tfrac12\sigma^2$ so the median path behaves like $S_0e^{gt}$.

What GBM gets right: positivity, compounding, tractability.  
What it misses: fat tails, skew, volatility clustering, jumps, default.  
I progressively add those features (correlation, t‑tails, regime switches, etc.) so strategies can be stress‑tested beyond the Gaussian world.

- **Singular regimes:** fixed $(\mu,\sigma)$ over time (e.g. Bull/Low‑Vol, Bear/High‑Vol).  
- **Paradigm shifts:** piecewise constant $(\mu,\sigma)$ with switch points to mimic cycles.
- **Mixed baskets:** combine different sub-baskets to a rich universe.

<!-- See the visuals in `notebooks/insights/01_gbm_regimes_overview.ipynb`. -->

---

## Quickstart — Demo backtest synthetic data (no API key)

```bash
git clone https://github.com/ElPhysico/ml-equity-backtester.git
cd ml-equity-backtester
pixi install && pixi run dev-install

# One‑command demo on GBM universe
pixi run demo_synth_elasticnet_topn
```

What it does:
- Builds a business‑day calendar (TDY configurable, defaults noted in logs).  
- Simulates an i.i.d. GBM universe (for now).  
- Trains a walk‑forward ElasticNet Top‑N and benchmarks vs BH/MR.  
- Saves results to `outputs/backtests/<run_id>/` (including `run_meta.json`).
- An example overlay vs benchmarks can be found in `docs/images/overlay_demo_synth.png`

---

## Quickstart — Demo backtest real market data (Alpha Vantage)

```bash
pixi install && pixi run dev-install
echo "ALPHA_VANTAGE_API_KEY=yourkey" > .env

# Download demo universe and (optionally) benchmarks
pixi run demo_download init
pixi run benchmarks_download init  # optional

# Backtest the demo strategy
pixi run demo15_elasticnet_topn
```

Outputs mirror the synthetic run. An example overlay vs benchmarks can be found in `docs/images/overlay_demo15.png`.

---

## Strategies (brief)

- **BH (Buy‑and‑Hold), MR (Monthly Rebalance):** baselines.  
- **ElasticNet Top‑N (walk‑forward):** cross‑sectional ML on monthly features.  
  - **Label:** next‑month simple return from month‑end $t$ to $t{+}1$.  
  - **Features:** momentum windows $[P,\text{skip}]$, realized volatility (currently $1\mathrm{m}$; parameterized variants coming).  
  - **Selection:** rank scores, hold Top‑$N$, apply `cost_bps`.  
  - **No look‑ahead:** train $<t$, predict at $t$, evaluate on $t{+}1$.

<!-- For a minimal spec, see the README section in the repo or `docs/` as it evolves. -->

---

## Insights (why this matters)

Even in GBM worlds, the **mean vs median** split matters, diversification reduces risk like $\sim 1/\sqrt{N}$ when names are i.i.d., and **regime switches** can whipsaw naive strategies. The **Insights** notebooks run BH/MR/ENet across bull/bear and low/high‑vol regimes and discuss expected behavior.

<!-- - Start: `notebooks/insights/01_gbm_regimes_overview.ipynb`  
- Planned: `notebooks/insights/02_strategy_perf_by_regime.ipynb` (per‑regime KPIs)
- Planned: `notebooks/insights/03_cross_regime_summary.ipynb` (side‑by‑side comparison) -->

---

## Roadmap (theory → simulator → tests)

- **Update Demo:** include statistical comparison between strategy and benchmarks.
- **Correlation:** one-factor $\rightarrow$ multi-factor $\rightarrow$ full correlation-matrix GBM.  
- **Fat tails:** Student‑t innovations; tail quantile validation.  
- **Skew:** skew‑t or downward jump‑diffusion.  
- **Vol clustering:** simple two‑regime or stochastic volatility; leverage effect.  
- **Signals with target IC:** synthetic cross‑sectional predictors with controllable IC.
- **Stresstesting:** evaluate strategy robustness across regimes.
- **Docs:** curated SVGs under `docs/images/` and concise write‑ups.

---

## Outputs & reproducibility

Each run under `outputs/backtests/<run_id>/` includes:
- `run_meta.json` (params, seed, TDY, simulator version, hash).  
- Equity curves, metrics, selections/weights/turnovers, overlays.  
<!-- - Synthetic runs also record regime specs. -->

Artifacts are deterministic under fixed seeds.

---

## Setup

Key libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyarrow`, `jupyter`, `python-dotenv`, `alpha_vantage`, managed via **Pixi**.  
Install: `pixi install && pixi run dev-install`

---

## License

MIT — use and adapt with attribution.

---

## Author

Kevin Klein  
LinkedIn: https://www.linkedin.com/in/kevin-klein-9a2342195/  
GitHub: https://github.com/ElPhysico/
