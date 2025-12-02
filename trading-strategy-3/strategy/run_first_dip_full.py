"""
Run full Strategy-3 First-Dip workflow in one go:

1. Load analysis/aligned_prices.csv (normalized prices after launch)
2. Grid-search first-dip parameters using ALL coins
3. Pick best parameter set
4. Run final backtest with best parameters on ALL coins
5. Save results CSVs under strategy/ and print Sharpe ratio
"""

from pathlib import Path
import itertools
import pandas as pd
import numpy as np


# -----------------------------
# 1. Data loading
# -----------------------------

def load_aligned_prices(base_dir: Path) -> pd.DataFrame:
    """
    Load analysis/aligned_prices.csv.

    Expected format:
        minute_since_launch,TRUMP,DADDY,JENNER,LIBRA,MELANIA,...
        0,1.0,1.0,1.0,...
        1,1.01,1.05,...

    Returns a DataFrame indexed by minute_since_launch with one column per coin.
    """
    path = base_dir / "analysis" / "aligned_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"aligned_prices.csv not found at {path}")
    df = pd.read_csv(path)
    if "minute_since_launch" not in df.columns:
        raise ValueError("aligned_prices.csv must have 'minute_since_launch' column.")
    df = df.set_index("minute_since_launch")
    return df


# -----------------------------
# 2. First-dip trade logic
# -----------------------------

def first_dip_trade(
    series: pd.Series,
    moon_mult: float,
    dip_min: float,
    dip_max: float,
    tp_level: float,
    stop_loss_pct: float,
    max_minutes: int,
):
    """
    Execute one first-dip trade on a single coin's normalized price series.

    series: normalized price (1.0 at t=0) indexed by minutes since launch.

    Returns:
        trade_taken (bool),
        ret (float, 0 if no trade),
        reason (str)
    """
    s = series.dropna()
    if s.empty:
        return False, 0.0, "no_data"

    # Already normalized: value at t0 should be ~1.0
    max_norm = s.max()
    if max_norm < moon_mult:
        return False, 0.0, "no_moon"

    # Time of peak within the first max_minutes from launch
    t_peak = s.idxmax()
    if t_peak >= max_minutes:
        return False, 0.0, "peak_too_late"

    # Find dip after the peak: price in [dip_min * max_norm, dip_max * max_norm]
    start_t = t_peak + 1
    end_t = min(s.index.max(), t_peak + max_minutes)
    dip_window = s.loc[start_t:end_t]
    dip_mask = (dip_window >= dip_min * max_norm) & (dip_window <= dip_max * max_norm)

    if not dip_mask.any():
        return False, 0.0, "no_dip"

    t_entry = dip_mask.idxmax()  # first time dip condition is true
    entry_norm = s.loc[t_entry]

    # Exit search window
    t_last = min(s.index.max(), t_entry + max_minutes)

    reason = "timeout"
    t_exit = t_last
    exit_norm = s.loc[t_exit]

    # Walk forward bar-by-bar and check TP / SL
    for t in range(t_entry + 1, t_last + 1):
        val = s.loc[t]
        # Take profit: back to some fraction of the original peak
        if val >= tp_level * max_norm:
            reason = "take_profit"
            t_exit = t
            exit_norm = val
            break
        # Stop loss: drop relative to entry
        if val <= entry_norm * (1.0 - stop_loss_pct):
            reason = "stop_loss"
            t_exit = t
            exit_norm = val
            break

    ret = exit_norm / entry_norm - 1.0
    return True, float(ret), reason


def run_backtest_on_all(
    prices: pd.DataFrame,
    moon_mult: float,
    dip_min: float,
    dip_max: float,
    tp_level: float,
    stop_loss_pct: float,
    max_minutes: int,
) -> pd.DataFrame:
    """
    Apply first_dip_trade to each coin column in prices.
    Returns a DataFrame with per-coin results.
    """
    results = []
    for coin in prices.columns:
        series = prices[coin]
        taken, ret, reason = first_dip_trade(
            series,
            moon_mult=moon_mult,
            dip_min=dip_min,
            dip_max=dip_max,
            tp_level=tp_level,
            stop_loss_pct=stop_loss_pct,
            max_minutes=max_minutes,
        )
        results.append(
            {
                "coin": coin,
                "trade_taken": taken,
                "ret": ret,
                "reason": reason,
            }
        )
    return pd.DataFrame(results)


# -----------------------------
# 3. Grid search
# -----------------------------

def grid_search_first_dip(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Brute-force grid search over parameter combinations.

    Returns a DataFrame with columns:
        moon_mult, dip_min, dip_max, tp_level, stop_loss_pct, max_minutes,
        avg_ret, num_trades, win_rate
    """

    # --- define search grid ---
    moon_mult_grid = [5.0, 8.0]
    dip_min_grid = [0.4, 0.5, 0.6]
    dip_max_grid = [0.7, 0.8, 0.9]
    tp_level_grid = [0.7, 0.8]
    stop_loss_grid = [0.1, 0.2, 0.3]
    max_minutes_grid = [180, 360]

    rows = []

    for (
        moon_mult,
        dip_min,
        dip_max,
        tp_level,
        stop_loss_pct,
        max_minutes,
    ) in itertools.product(
        moon_mult_grid,
        dip_min_grid,
        dip_max_grid,
        tp_level_grid,
        stop_loss_grid,
        max_minutes_grid,
    ):

        # Require dip_min < dip_max
        if dip_min >= dip_max:
            continue

        df_res = run_backtest_on_all(
            prices,
            moon_mult=moon_mult,
            dip_min=dip_min,
            dip_max=dip_max,
            tp_level=tp_level,
            stop_loss_pct=stop_loss_pct,
            max_minutes=max_minutes,
        )

        traded = df_res[df_res["trade_taken"]]
        num_trades = len(traded)
        if num_trades == 0:
            avg_ret = 0.0
            win_rate = 0.0
        else:
            avg_ret = traded["ret"].mean()
            win_rate = (traded["ret"] > 0).mean()

        rows.append(
            {
                "moon_mult": moon_mult,
                "dip_min": dip_min,
                "dip_max": dip_max,
                "tp_level": tp_level,
                "stop_loss_pct": stop_loss_pct,
                "max_minutes": max_minutes,
                "avg_ret": avg_ret,
                "num_trades": num_trades,
                "win_rate": win_rate,
            }
        )

    return pd.DataFrame(rows)


def pick_best_params(df_grid: pd.DataFrame, min_trades: int = 2):
    """
    Pick the best row from the grid search results.

    Criteria:
        - num_trades >= min_trades
        - highest avg_ret
        - tie-breaker: higher win_rate
    """
    filt = df_grid[df_grid["num_trades"] >= min_trades].copy()
    if filt.empty:
        # Fall back to overall best avg_ret, even if very few trades
        filt = df_grid.copy()

    filt = filt.sort_values(
        by=["avg_ret", "win_rate"], ascending=[False, False]
    ).reset_index(drop=True)

    best = filt.iloc[0]
    return best


# -----------------------------
# 4. Sharpe helpers
# -----------------------------

def compute_sharpe_from_returns(rets: pd.Series) -> float:
    """
    Compute per-trade Sharpe ratio from a Series of returns.

    Sharpe (not annualized) = mean(ret) / std(ret).

    Returns NaN if fewer than 2 trades or std == 0.
    """
    r = pd.Series(rets).dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    mean_ret = r.mean()
    std_ret = r.std(ddof=1)
    if std_ret == 0:
        return float("nan")
    return float(mean_ret / std_ret)


def compute_sharpe_from_results(results_path: Path) -> float:
    """
    Convenience wrapper: compute Sharpe from first_dip_results.csv.
    Uses only rows where trade_taken == True.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}")

    df = pd.read_csv(results_path)
    if "trade_taken" not in df.columns or "ret" not in df.columns:
        raise ValueError(
            "results CSV must have 'trade_taken' and 'ret' columns."
        )

    rets = df.loc[df["trade_taken"] == True, "ret"]
    return compute_sharpe_from_returns(rets)


# -----------------------------
# 5. Main
# -----------------------------

def main():
    # This file lives in trading-strategy-3/strategy/
    base_dir = Path(__file__).resolve().parents[1]
    strategy_dir = base_dir / "strategy"
    strategy_dir.mkdir(exist_ok=True)

    print(f"Base dir: {base_dir}")

    # 1) Load data
    prices = load_aligned_prices(base_dir)
    print(f"Loaded aligned_prices with shape {prices.shape}")

    # 2) Grid search
    print("Running grid search over first-dip parameters on ALL coins...")
    df_grid = grid_search_first_dip(prices)
    grid_path = strategy_dir / "first_dip_grid_results_auto.csv"
    df_grid.to_csv(grid_path, index=False)
    print(f"Saved grid search results to {grid_path}")

    # 3) Pick best params
    best = pick_best_params(df_grid, min_trades=2)
    print("\nBest parameter set:")
    print(best)

    best_params = dict(
        moon_mult=float(best["moon_mult"]),
        dip_min=float(best["dip_min"]),
        dip_max=float(best["dip_max"]),
        tp_level=float(best["tp_level"]),
        stop_loss_pct=float(best["stop_loss_pct"]),
        max_minutes=int(best["max_minutes"]),
    )

    print("\nUsing best params to run final backtest:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 4) Run final backtest with best parameters
    df_final = run_backtest_on_all(prices, **best_params)
    out_path = strategy_dir / "first_dip_results_auto.csv"
    df_final.to_csv(out_path, index=False)
    print(f"\nSaved per-coin first-dip results to {out_path}\n")
    print(df_final)

    # 5) Compute Sharpe ratio across all trades
    sharpe = compute_sharpe_from_results(out_path)
    print(f"\nIn-sample Sharpe ratio (per trade, not annualized): {sharpe:.2f}")


if __name__ == "__main__":
    main()
