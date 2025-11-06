import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Path constants
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA = ROOT / "data"
OUT_PATH = ROOT / "strategy" / "strategy_results.json"
PLOTS_DIR = ROOT / "strategy" / "strategy_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Plot configurations
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

# Strategy Functions
def load_close_series(path, label):
    """Load CSV with timestamp + close columns."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].rename(columns={"close": f"{label}_close"})
    return df

def compute_log_returns(df, col):
    df[f"{col}_ret"] = np.log(df[col]).diff()
    return df.dropna()

# Align BTC and coin by timestamp and compute log returns.
def align_returns(btc_df, coin_df, label):
    merged = btc_df.merge(coin_df, on="timestamp", how="inner").dropna()
    merged = compute_log_returns(merged, "BTC_close")
    merged = compute_log_returns(merged, f"{label}_close")
    return merged.dropna()


# Compute performance metrics for strategy returns.
def compute_metrics(strategy_returns):
    cumret = strategy_returns.cumsum()
    total_return = np.exp(cumret.iloc[-1]) - 1
    sharpe = np.sqrt(len(strategy_returns)) * strategy_returns.mean() / strategy_returns.std()
    roll_max = cumret.cummax()
    drawdown = cumret - roll_max
    max_dd = -drawdown.min()
    return total_return, sharpe, max_dd, cumret

# Simulate BTC-lag trading strategy for one coin.
def simulate_strategy(df, coin_label, lag):
    coin_ret = df[f"{coin_label}_close_ret"]
    btc_ret = df["BTC_close_ret"]

    signal = np.sign(btc_ret.shift(lag))
    strat_ret = signal * coin_ret
    strat_ret = strat_ret.dropna()

    total_return, sharpe, max_dd, cumret = compute_metrics(strat_ret)
    return {
        "lag_used": lag,
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4)
    }, cumret, strat_ret

# Make plots
def plot_all(merged, coin, lag, cumret, strat_ret):
    """Generate and save diagnostic plots for each coin."""
    coin_ret = merged[f"{coin}_close_ret"]
    btc_ret = merged["BTC_close_ret"]

    # Cum return plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumret.index, cumret, label=f"{coin} Strategy (lag={lag})", linewidth=2)
    plt.plot(coin_ret.cumsum().index, coin_ret.cumsum(), label=f"{coin} Passive (HODL)", linestyle="--")
    plt.title(f"{coin} Strategy vs HODL (lag={lag})")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{coin}_1_cum_returns.png")
    plt.close()

    # rolling sharpe ratio plot
    window = 200
    rolling_sharpe = strat_ret.rolling(window).mean() / strat_ret.rolling(window).std() * np.sqrt(window)
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe, color="purple")
    plt.title(f"{coin} Rolling Sharpe Ratio (window={window})")
    plt.xlabel("Time")
    plt.ylabel("Sharpe Ratio")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{coin}_2_rolling_sharpe.png")
    plt.close()

    # BTC vs. Coin plots
    signal = np.sign(btc_ret.shift(lag))
    plt.figure(figsize=(6, 6))
    plt.scatter(signal, merged[f"{coin}_close_ret"], alpha=0.3)
    plt.title(f"{coin}: BTC Signal vs Coin Returns (lag={lag})")
    plt.xlabel("BTC Signal (-1 short, +1 long)")
    plt.ylabel(f"{coin} Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{coin}_3_signal_scatter.png")
    plt.close()

    #Rolling correlatino plots
    window_corr = 300
    rolling_corr = btc_ret.rolling(window_corr).corr(coin_ret)
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_corr, color="teal")
    plt.title(f"{coin} Rolling Correlation with BTC (window={window_corr})")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{coin}_4_rolling_corr.png")
    plt.close()

    print(f"Saved 4 plots for {coin} in {PLOTS_DIR}")

def main():
    btc_df = load_close_series(DATA / "example_btc.csv", "BTC")
    eth_df = load_close_series(DATA / "example_eth.csv", "ETH")
    sol_df = load_close_series(DATA / "example_sol.csv", "SOL")

    # Lags identified from prev. correlation analysis
    lags = {"ETH": 1, "SOL": 2}

    results = {}
    for coin, lag in lags.items():
        coin_df = eth_df if coin == "ETH" else sol_df
        merged = align_returns(btc_df, coin_df, coin)

        res, cumret, strat_ret = simulate_strategy(merged, coin, lag)
        results[coin] = res
        plot_all(merged, coin, lag, cumret, strat_ret)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Strategy Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
