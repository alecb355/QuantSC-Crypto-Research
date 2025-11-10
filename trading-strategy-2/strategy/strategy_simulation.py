import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path constants
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA = ROOT / "data"
OUT_DIR = ROOT / "strategy" / "plots"
OUT_DIR.mkdir(exist_ok=True)

BTC_PATH = DATA / "example_btc.csv"
COINS = ["ETH", "SOL"]
COIN_PATHS = [DATA / f"example_{c.lower()}.csv" for c in COINS]

# Buy/sell thresholds
LAGS = {"ETH": 1, "SOL": 2} 
THRESHOLDS = [0.0, 0.0005, 0.001, 0.002]
MAX_POSITION = 10


def load_price_series(path, label):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].rename(columns={"close": f"{label}_close"})
    return df

def compute_log_returns(df, col):
    # df[f"{col}_ret"] = np.log(df[col]).diff()

    # df[f"{col}_ret"] = df[col]
    df[f"{col}_ret"] = df[col].pct_change()
    return df.dropna().reset_index(drop=True)

def align_and_returns(btc_df, coin_df, coin_label):
    merged = coin_df.merge(btc_df, on="timestamp", how="inner").sort_values("timestamp")
    merged = compute_log_returns(merged, "BTC_close")
    merged = compute_log_returns(merged, f"{coin_label}_close")
    return merged.dropna().reset_index(drop=True)

def simulate_strategy(df, coin_label, lag=1, buy_thresh=0.0, sell_thresh=0.0, max_pos=10):
    """
    Simulate threshold-based BTC→altcoin lag strategy.
    """
    btc_ret = df["BTC_close_ret"]
    coin_ret = df[f"{coin_label}_close_ret"]

    sig_raw = np.where(
        btc_ret.shift(lag) > buy_thresh, 1,
        np.where(btc_ret.shift(lag) < sell_thresh, -1, 0)
    )

    position = 0
    positions = []
    strategy_rets = []

    for i in range(len(df) - 1):
        sig = sig_raw[i]

        if sig == 1 and position < max_pos:
            position += 1
        elif sig == -1 and position > -max_pos:
            position -= 1

        positions.append(position)

        strat_ret = position * coin_ret.iloc[i + 1]
        strategy_rets.append(strat_ret)

    df = df.iloc[1:].copy()
    df["position"] = positions
    df["strategy_ret"] = strategy_rets

    # df["cumret"] = np.exp(np.cumsum(df["strategy_ret"])) - 1
    df["cumret"] = (1 + df["strategy_ret"]).cumprod() - 1

    total_return = df["cumret"].iloc[-1]
    sharpe = (
        np.mean(df["strategy_ret"]) / np.std(df["strategy_ret"]) * np.sqrt(365*24*60)
        if np.std(df["strategy_ret"]) > 0 else 0
    )
    cummax = np.maximum.accumulate(df["cumret"])
    max_drawdown = np.max(1 - (1 + df["cumret"]) / (1 + cummax))

    metrics = {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_drawdown), 4),
    }

    return df, metrics

# Plot graphs
def plot_strategy(df, coin_label, buy_thresh, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["timestamp"], df["cumret"], label="Cumulative Return")
    axes[0].set_title(f"{coin_label} Strategy Performance (threshold={buy_thresh})")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["timestamp"], df["position"], label="Position", color="orange")
    axes[1].set_title("Position Over Time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Position Size")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{coin_label}_threshold_{buy_thresh:.4f}.png", dpi=200)
    plt.close(fig)


def main():
    btc_df = load_price_series(BTC_PATH, "BTC")
    results = {}

    for coin, path in zip(COINS, COIN_PATHS):
        coin_df = load_price_series(path, coin)
        df = align_and_returns(btc_df, coin_df, coin)

        best_metrics = None
        best_thresh = None
        best_df = None

        for thresh in THRESHOLDS:
            sim_df, metrics = simulate_strategy(
                df, coin_label=coin, lag=LAGS[coin],
                buy_thresh=thresh, sell_thresh=-thresh, max_pos=MAX_POSITION
            )

            plot_strategy(sim_df, coin, thresh, OUT_DIR)

            # keep best Sharpe
            if best_metrics is None or metrics["sharpe"] > best_metrics["sharpe"]:
                best_metrics, best_thresh, best_df = metrics, thresh, sim_df

        results[coin] = {
            "lag_used": LAGS[coin],
            "best_threshold": best_thresh,
            **best_metrics
        }

        # plot best threshold separately
        plot_strategy(best_df, coin, best_thresh, OUT_DIR)

    out_path = OUT_DIR / "strategy_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
