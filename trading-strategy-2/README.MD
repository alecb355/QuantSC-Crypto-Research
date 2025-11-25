We are testing if we can make a strategy based on lead-lag relationships where BTC is the leader and alt coins lag behind

# Lead–Lag (DTW) Strategy

## What this is
We test whether **BTC leads other coins** on short horizons.

1. Measure lead/lag using **Dynamic Time Warping (DTW)** on recent 1-minute **log-returns**.
2. Convert the estimated lag \(k\) into a **market-neutral spread trade**: long/short the alt vs BTC when BTC’s k-bar move is big but the alt hasn’t moved yet.

---

## Folder layout
trading-strategy-2/
├─ analysis/
│ ├─ dynamic_time_warping_analysis.py # builds per-coin DTW lag (k) on TRAIN only
│ ├─ dtw_train_lags.csv # output: coin → k and train_end
│ └─ sample_results_dtw.csv # optional DTW summary
├─ data/
│ └─ data_1m/ # 1-min CSVs (BTCUSD.csv, DOGEUSD.csv, …)
├─ strategy/
│ ├─ leadlag_spread.py # backtest: uses k to trade TEST window
│ ├─ run_pipeline.py # end-to-end: DTW → backtest all coins
│ ├─ dtw_out/ # equity curves & summary
│ └─ strategy_plots/ # optional PNGs
└─ README.md

---

## Data format (1-minute bars)
Place CSVs in `trading-strategy-2/data/data_1m/` with at least:
```csv
timestamp,open,high,low,close,volume
2025-02-19 13:00:00,0.2845,0.2845,0.2845,0.2845,0.0
- Timestamps should be UTC or convertible to UTC.
- Data are aligned to 1-minute; tiny gaps are forward-filled.
```

## Environment
Python ≥ 3.9

Packages: pandas, numpy, fastdtw, scipy, matplotlib
(We use fastdtw if available; otherwise fall back to exact DTW.)

Install:
`pip install pandas numpy fastdtw scipy matplotlib`

## Quick start
### 1) Find lags k with DTW (TRAIN window)
```python trading-strategy-2/analysis/dynamic_time_warping_analysis.py \
  --data-dir trading-strategy-2/data/data_1m \
  --ref BTCUSD.csv \
  --lookback 1440 \
  --train-ratio 0.7 \
  --out trading-strategy-2/analysis/sample_results_dtw.csv \
  --k-out trading-strategy-2/analysis/dtw_train_lags.csv
```

This creates analysis/dtw_train_lags.csv with:

symbol — coin (e.g., DOGEUSD)

k — lag in minutes (rounded)

train_end — train cutoff (TEST starts after this)

### 2) Backtest the spread on the TEST window
```
python trading-strategy-2/strategy/leadlag_spread.py \
  --coin DOGEUSD.csv \
  --ref BTCUSD.csv \
  --k-from trading-strategy-2/analysis/dtw_train_lags.csv \
  --z-lookback 1440 \
  --beta-lookback 1440 \
  --out trading-strategy-2/strategy/dtw_out/equity_DOGE.csv
```

Example output:
```
[stats] {'coin':'DOGEUSD','k':45,'trades':485,'ret_total':0.1483,
         'ret_annualized':1.5583,'vol_annualized':0.5748,'sharpe':2.71,'winrate':0.189}
[saved] equity curve -> trading-strategy-2/strategy/dtw_out/equity_DOGE.csv
```

## End-to-end pipeline (all coins)

Runs DTW (TRAIN) → backtests (TEST) for every coin, + optional plots.
```
python trading-strategy-2/strategy/run_pipeline.py \
  --data-dir trading-strategy-2/data/data_1m \
  --ref BTCUSD.csv \
  --lookback 1440 \
  --train-ratio 0.7 \
  --z-lookback 1440 \
  --beta-lookback 1440 \
  --out-dir trading-strategy-2/strategy/dtw_out \
  --plots \
  --plot-collective \
  --no-per-coin-plots \
  --collective-resample 5min
```

Outputs:

`analysis/dtw_train_lags.csv` — per-coin `k` and `train_end`

`strategy/dtw_out/equity_<COIN>.csv` — per-coin equity curve

`strategy/dtw_out/summary_stats.csv` — stats across coins

`strategy/strategy_plots/equity_collective.png` — all coins overlaid in one figure
(omit `--no-per-coin-plots` to also save individual `equity_<COIN>.png` files)

## Plotting options
- `--plot-collective`
Produce the single overlay PNG: `strategy/strategy_plots/equity_collective.png`.
- `--no-per-coin-plots`
Skip individual coin PNGs; still writes each coin’s equity CSV.
- `--collective-resample <rule>`
Optional smoothing of the overlay by resampling returns before re-cumsumming.
Examples: `5min`, `15min`, `1H`, `1D`. If omitted, uses native 1-minute bars.

### Examples
```
# Overlay only (no per-coin PNGs), 5-minute smoothing
python trading-strategy-2/strategy/run_pipeline.py --plots --plot-collective --no-per-coin-plots --collective-resample 5min

# Overlay + per-coin PNGs (unsmoothed)
python trading-strategy-2/strategy/run_pipeline.py --plots --plot-collective
```

## How the strategy works

1. Measure delay: DTW on 1-min returns finds a lag k where BTC leads each coin (TRAIN only).

2. Wait for setup: compute k-minute cumulative returns and convert to z-scores.
If BTC’s z ≥ z_entry and the coin’s z < z_follow, the coin hasn’t reacted yet.

3. Trade the spread:
- BTC up, coin hasn’t moved → long coin / short BTC.
- BTC down, coin hasn’t moved → short coin / long BTC.
Exit when the z-gap closes (z_close) or after k minutes.

4. Hedge: use rolling beta (coin vs BTC) to be roughly market-neutral.

## Key flags

- DTW: --lookback, --train-ratio or --train-end

- Backtest: --k or --k-from analysis/dtw_train_lags.csv

- Rolling windows: --z-lookback, --beta-lookback

- Thresholds: --z-entry, --z-follow, --z-close

- Costs: --fee-bps (round-trip basis points per position change)

## Interpreting stats

- `ret_total`: cumulative log return over TEST

- `ret_annualized`, `vol_annualized`: minute→year scaling

- `sharpe`: annualized Sharpe ratio

- `winrate`: share of profitable bar-level increments while in a position

- `trades`: number of entries (not bar-by-bar flips)

## Assumptions & notes

- All timestamps treated as UTC.

- DTW trained on TRAIN only; backtest uses TEST only.

- Rolling stats are past-only (no look-ahead).

- Beta is clipped to ±5 for stability.

- Minute gaps are forward-filled to keep alignment.

## Repro example (single coin)
```# 1) Build k table (TRAIN)
python trading-strategy-2/analysis/dynamic_time_warping_analysis.py \
  --data-dir trading-strategy-2/data/data_1m --ref BTCUSD.csv \
  --lookback 1440 --train-ratio 0.7 \
  --k-out trading-strategy-2/analysis/dtw_train_lags.csv

# 2) Backtest DOGE (TEST)
python trading-strategy-2/strategy/leadlag_spread.py \
  --coin DOGEUSD.csv --ref BTCUSD.csv \
  --k-from trading-strategy-2/analysis/dtw_train_lags.csv \
  --z-lookback 1440 --beta-lookback 1440 \
  --out trading-strategy-2/strategy/dtw_out/equity_DOGE.csv
```
