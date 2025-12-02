"""
Leave-one-coin-out cross-validation for the Strategy-3 First-Dip setup.

For each coin:
    - Hold it out
    - Grid search parameters on the remaining coins
    - Pick best params
    - Run first-dip trade ONLY on the held-out coin (true out-of-sample)
    - Record trade result + params

Saves:
    strategy/first_dip_cv_results_all_coins.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

from run_first_dip_full import (
    load_aligned_prices,
    first_dip_trade,
    grid_search_first_dip,
    pick_best_params,
    compute_sharpe_from_returns,
)


def main():
    # This file lives in trading-strategy-3/strategy/
    base_dir = Path(__file__).resolve().parents[1]
    strategy_dir = base_dir / "strategy"
    strategy_dir.mkdir(exist_ok=True)

    print(f"Base dir: {base_dir}")

    # 1) Load aligned prices
    prices = load_aligned_prices(base_dir)
    coins = list(prices.columns)
    print(f"Loaded aligned_prices with shape {prices.shape}")
    print(f"Coins: {coins}")

    cv_rows = []
    oos_returns = []  # all out-of-sample returns across held-out coins

    # -------------------------------------------------
    # 2) Leave-one-coin-out loop
    # -------------------------------------------------
    for held_out in coins:
        print("\n==============================")
        print(f"Held-out coin: {held_out}")
        print("==============================")

        train_coins = [c for c in coins if c != held_out]
        train_prices = prices[train_coins]

        # 2a) Grid search on training coins only
        df_grid = grid_search_first_dip(train_prices)
        best = pick_best_params(df_grid, min_trades=2)

        best_params = dict(
            moon_mult=float(best["moon_mult"]),
            dip_min=float(best["dip_min"]),
            dip_max=float(best["dip_max"]),
            tp_level=float(best["tp_level"]),
            stop_loss_pct=float(best["stop_loss_pct"]),
            max_minutes=int(best["max_minutes"]),
        )

        print("Best params from training coins:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # 2b) Apply these params to held-out coin only
        series_ho = prices[held_out]
        taken, ret, reason = first_dip_trade(series_ho, **best_params)

        print(f"Held-out trade_taken={taken}, ret={ret:.4f}, reason={reason}")

        if taken:
            oos_returns.append(ret)

        row = {
            "held_out_coin": held_out,
            "trade_taken": taken,
            "ret": ret,
            "reason": reason,
            "moon_mult": best_params["moon_mult"],
            "dip_min": best_params["dip_min"],
            "dip_max": best_params["dip_max"],
            "tp_level": best_params["tp_level"],
            "stop_loss_pct": best_params["stop_loss_pct"],
            "max_minutes": best_params["max_minutes"],
        }
        cv_rows.append(row)

    # -------------------------------------------------
    # 3) Save per-coin CV results
    # -------------------------------------------------
    df_cv = pd.DataFrame(cv_rows)

    # Per-coin Sharpe is statistically undefined (≤1 trade each),
    # but we'll add a column that is NaN unless you later decide
    # to use some biased definition.
    per_coin_sharpe = []
    for _, r in df_cv.iterrows():
        if not r["trade_taken"]:
            per_coin_sharpe.append(float("nan"))
        else:
            # Only 1 trade per coin → Sharpe (mean/std) is undefined (std=0).
            per_coin_sharpe.append(float("nan"))
    df_cv["sharpe_per_coin"] = per_coin_sharpe

    out_path = strategy_dir / "first_dip_cv_results_all_coins.csv"
    df_cv.to_csv(out_path, index=False)
    print(f"\nSaved CV results to {out_path}")
    print(df_cv)

    # -------------------------------------------------
    # 4) Overall CV Sharpe across all held-out trades
    # -------------------------------------------------
    if len(oos_returns) == 0:
        print("\nNo out-of-sample trades taken in CV; Sharpe is undefined.")
        return

    overall_sharpe = compute_sharpe_from_returns(pd.Series(oos_returns))

    print("\nOut-of-sample CV Sharpe (per trade, not annualized):")
    print(f"  N trades = {len(oos_returns)}")
    print(f"  Sharpe   = {overall_sharpe:.2f}")


if __name__ == "__main__":
    main()
