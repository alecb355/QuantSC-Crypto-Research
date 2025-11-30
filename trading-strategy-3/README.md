# Strategy 3 – Celeb Coin Launch & First-Dip Backtest

This folder contains the **celebrity/meme coin launch pattern study** and a simple **“First Dip” trading strategy** backtester.

The workflow:

1. Align all celeb coins to a common launch time (`T0`) and normalize their price paths.
2. Optionally compare each coin to a TRUMP template using correlation + DTW.
3. Run a **grid search** to find good parameters for a “buy the first dip after a moon” strategy.
4. Run a full backtest of the First-Dip strategy using the best parameters.

---

## Folder structure

```text
trading-strategy-3/
├── analysis/
│   ├── celeb_vs_trump.py                # aligns coins and (optionally) compares to TRUMP
│   ├── aligned_prices.csv               # aligned, normalized prices after launch (output)
│   ├── celeb_vs_trump_results.csv       # corr/DTW stats vs TRUMP (output)
│   └── overlay_aligned_prices.png       # visualization of aligned paths (output)
├── data/
│   └── coin_data/
│       ├── TRUMP.csv
│       ├── MELANIA.csv
│       ├── DADDY.csv
│       ├── JENNER.csv
│       └── LIBRA.csv
└── strategy/
    ├── first_dip_backtest.py            # core First-Dip trading logic for a single param set
    ├── grid_search_first_dip.py         # grid search over First-Dip params
    ├── first_dip_results.csv            # per-coin trade results for chosen params (output)
    ├── first_dip_grid_search_results.csv# all param combos + performance stats (output)
    └── run_first_dip_full.py            # one-click pipeline for Strategy 3
```

## 1. Prepare aligned launch data

You need minute OHLCV CSVs for each celeb coin in:
`trading-strategy-3/data/coin_data/<COIN>.csv`
with at least:
- a timestamp column
- a close column

Then run the alignment script.

**From the repo root**
```
cd trading-strategy-3

python analysis/celeb_vs_trump.py \
  --data-dir data/coin_data \
  --ref TRUMP.csv \
  --horizon 360 \
  --plot
```
This will:
- read all `*.csv` in `data/coin_data/`
- align them so that `t = 0` is each coin’s launch time
- normalize each series to 1.0 at `t = 0`
- write:
    - `analysis/aligned_prices.csv`
    - `analysis/celeb_vs_trump_results.csv`
    - `analysis/overlay_aligned_prices.png`

If you get `aligned_prices.csv not found` later, it means this step hasn’t been run (or it failed).

## 2. Run First-Dip grid search

The First-Dip strategy tries to capture the first “buyable dip” after a big launch moon.
Parameters include:
- `moon_mult` – how big the initial moon must be (e.g. 5×, 8×)
- `dip_min`, `dip_max` – retrace zone as fraction of the peak (e.g. 0.6–0.7)
- `tp_level` – take-profit level relative to the peak (e.g. 0.8 = 80% of peak)
- `stop_loss_pct` – max loss from entry (e.g. 0.3 = 30%)
- `max_minutes` – how long to hold at most

To search across a grid of these:

**From the repo root**

```
cd trading-strategy-3
python strategy/grid_search_first_dip.py
```

This script:
- loads `analysis/aligned_prices.csv`
- loops over a grid of parameter values
- runs the First-Dip backtest for each combination
- writes `strategy/first_dip_grid_search_results.csv` with:
    - the parameter combo
    - average return per trade
    - number of trades
    - win rate

If this file is missing, `run_first_dip_full.py` will raise a `FileNotFoundError`.


## 3. Run the full First-Dip backtest with best params

Once you have `first_dip_grid_search_results.csv`, run the one-click pipeline:

**From the repo root**
```
cd trading-strategy-3
python strategy/run_first_dip_full.py
```

What this script does:
- Pick best parameters from `first_dip_grid_search_results.csv`
    - by default: maximize average return, with a minimum number of trades.
- Run First-Dip backtest with those parameters on all coins in `aligned_prices.csv` using `first_dip_backtest.py`.
- Print a summary to the console.
- Save detailed results to:
    - `strategy/first_dip_results.csv` – per-coin trade_taken/ret/reason

(optionally) plots or additional diagnostics if added later.

## 4. Typical run sequence (everything from scratch)

**From the repo root:**
```
cd trading-strategy-3

# 1. Align and normalize launch data
python analysis/celeb_vs_trump.py \
  --data-dir data/coin_data \
  --ref TRUMP.csv \
  --horizon 360 \
  --plot

# 2. Grid search to tune First-Dip parameters
python strategy/grid_search_first_dip.py

# 3. Use best params and run final backtest
python strategy/run_first_dip_full.py
```