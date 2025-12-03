import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# PATHS
# ---------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]

DATA_DIR = ROOT / "data" / "data_1m"
LAG_FILE = ROOT / "analysis" / "all_tokens_results.json"

OUT_DIR = ROOT / "strategy" / "plots"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------
# STRATEGY PARAMS
# ---------------------------
THRESHOLDS = [0.0005, 0.001, 0.002]
MAX_POSITION = 10
LOOKBACK_DAYS = 90     # ~3 months
ANNUALIZE = 365 * 24 * 60
TRANSACTION_COST = 0.0005   # 5 bps per exposure change
TOP_N = 10
# ---------------------------


def load_series(path, label):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]]
    df.columns = ["timestamp", f"{label}_close"]
    return df


def add_returns(df, col):
    df[f"{col}_ret"] = df[col].pct_change()
    return df.dropna().reset_index(drop=True)


def align(btc, coin, label):
    df = coin.merge(btc, on="timestamp", how="inner").sort_values("timestamp")
    df = add_returns(df, "BTC_close")
    df = add_returns(df, f"{label}_close")
    return df.dropna().reset_index(drop=True)


# ---------------------------
# CORE STRATEGY ENGINE
# ---------------------------
def simulate(df, symbol, lag, threshold, transaction_cost, max_position):
    if lag == 0:
        raise ValueError("Lag cannot be zero")

    trade_btc = lag < 0
    lag = abs(int(lag))

    if trade_btc:
        signal_ret = df[f"{symbol}_close_ret"]
        trade_ret = df["BTC_close_ret"]
        traded = "BTC"
    else:
        signal_ret = df["BTC_close_ret"]
        trade_ret = df[f"{symbol}_close_ret"]
        traded = symbol

    signals = np.where(
        signal_ret.shift(lag) > threshold, 1,
        np.where(signal_ret.shift(lag) < -threshold, -1, 0)
    )

    position = 0
    prev_exposure = 0.0

    exposures = []
    returns = []
    turnovers = []

    for t in range(len(df) - 1):
        sig = signals[t]

        if sig == 1 and position < max_position:
            position += 1
        elif sig == -1 and position > -max_position:
            position -= 1

        exposure = position / max_position
        turnover = abs(exposure - prev_exposure)

        r = exposure * trade_ret.iloc[t + 1] - transaction_cost * turnover

        exposures.append(exposure)
        turnovers.append(turnover)
        returns.append(r)

        prev_exposure = exposure

    result = df.iloc[1:].copy()
    result["exposure"] = exposures
    result["strategy_ret"] = returns
    result["turnover"] = turnovers

    wealth = (1 + result["strategy_ret"]).cumprod()
    result["wealth"] = wealth
    result["cumret"] = wealth - 1

    mu = result["strategy_ret"].mean()
    vol = result["strategy_ret"].std()
    sharpe = mu / vol * np.sqrt(ANNUALIZE) if vol > 0 else 0

    return result, {
        "symbol": symbol,
        "trade_asset": traded,
        "threshold": threshold,
        "sharpe": sharpe,
        "total_return": result["cumret"].iloc[-1]
    }


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    with open(LAG_FILE) as f:
        lag_data = json.load(f)

    # sort by absolute correlation
    ranked = sorted(
        lag_data.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    )

    top10 = ranked[:TOP_N]

    print("\nTOP 10 BY CORRELATION")
    for s, info in top10:
        print(s, "corr:", round(info["correlation"], 3))

    # ---------------------------
    # Load BTC
    # ---------------------------
    btc_file = DATA_DIR / "BTCUSDT.csv"
    btc = load_series(btc_file, "BTC")

    portfolio_returns = []

    results = {}

    for symbol, info in top10:
        path = DATA_DIR / f"{symbol}.csv"
        if not path.exists():
            print(f"Skipping {symbol} (missing data)")
            continue

        lag = info.get("lag_with_highest_corr", 0)
        if lag == 0:
            lag = info.get("second_best_lag", 0)

        if lag == 0:
            print(f"Skipping {symbol} (no usable lag)")
            continue

        coin = load_series(path, symbol)
        df = align(btc, coin, symbol)

        # restrict to past 3 months
        cutoff = df["timestamp"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

        if len(df) < 100:
            print(f"Skipping {symbol} (too few rows)")
            continue

        best = None
        best_df = None

        for thresh in THRESHOLDS:
            sim, metrics = simulate(
                df, symbol, lag, thresh,
                transaction_cost=TRANSACTION_COST,
                max_position=MAX_POSITION
            )

            if best is None or metrics["sharpe"] > best["sharpe"]:
                best = metrics
                best_df = sim

        results[symbol] = best
        portfolio_returns.append(best_df["strategy_ret"])

        print(f"{symbol}: Sharpe={round(best['sharpe'],2)} | "
              f"Return={round(best['total_return'],2)} | lag={lag}")

    # ---------------------------
    # BUILD PORTFOLIO
    # ---------------------------
    portfolio_df = pd.concat(portfolio_returns, axis=1)
    portfolio_df.columns = results.keys()

    # Equal weight portfolio
    portfolio_df["portfolio_ret"] = portfolio_df.mean(axis=1)

    wealth = (1 + portfolio_df["portfolio_ret"]).cumprod()
    portfolio_df["wealth"] = wealth
    portfolio_df["cumret"] = wealth - 1

    # metrics
    mu = portfolio_df["portfolio_ret"].mean()
    vol = portfolio_df["portfolio_ret"].std()
    sharpe = mu / vol * np.sqrt(ANNUALIZE)

    print("\n==== PORTFOLIO RESULTS ====")
    print("Sharpe:", round(sharpe, 2))
    print("Total Return:", round(portfolio_df["cumret"].iloc[-1], 3))

    # ---------------------------
    # PLOT
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df["cumret"], linewidth=2)
    plt.title("Top 10 Correlation Portfolio (3 months)")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "TOP10_PORTFOLIO.png", dpi=200)
    plt.close()

    # ---------------------------
    # SAVE RESULTS
    # ---------------------------
    with open(OUT_DIR / "top10_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(" - plots/TOP10_PORTFOLIO.png")
    print(" - plots/top10_results.json")


if __name__ == "__main__":
    main()
