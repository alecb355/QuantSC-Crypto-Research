from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os

# ---- project-relative paths (portable) ----
STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_DIR  = STRATEGY_DIR.parent
DATA_DIR     = PROJECT_DIR / "data" / "data_1m"
ANALYSIS_DIR = PROJECT_DIR / "analysis"
DEFAULT_REF  = "BTCUSD.csv"

# default rolling window lengths (in 1-minute bars)
Z_LOOKBACK = 720          # 12 hours for z-scores
BETA_LOOKBACK = 720       # 12 hours for beta

# ---------- helpers ----------
TIME_COLS = ["bar","timestamp","time","Datetime","date","Date","datetime"]

def load_close(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"timestamp column not found in {csv_path.name}")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    s = pd.Series(close.values, index=ts).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index().asfreq("1min", method="pad")
    return s

def roll_z(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std()
    return (x - mu) / (sd + 1e-12)

# ---------- core backtest ----------
def backtest_leadlag_spread(
    coin_file, ref_file, k, z_entry, z_follow, z_close, fee_bps,
    test_start=None, test_end=None, z_lookback=Z_LOOKBACK, beta_lookback=BETA_LOOKBACK
):
    # Load ALL data first to calculate rolling windows properly
    ref_full = load_close(DATA_DIR / ref_file)
    coi_full = load_close(DATA_DIR / coin_file)
    
    # Align on full dataset
    idx_full = ref_full.index.intersection(coi_full.index)
    ref_full = ref_full.loc[idx_full]
    coi_full = coi_full.loc[idx_full]
    
    # Calculate returns and rolling stats on FULL data
    r_btc  = np.log(ref_full).diff().fillna(0.0)
    r_coin = np.log(coi_full).diff().fillna(0.0)

    Rb = r_btc.rolling(k).sum()
    Ri = r_coin.rolling(k).sum()
    z_btc  = roll_z(Rb, z_lookback)
    z_coin = roll_z(Ri, z_lookback)

    cov = r_coin.rolling(beta_lookback).cov(r_btc)
    var = r_btc.rolling(beta_lookback).var()
    beta = (cov / (var + 1e-12)).clip(-5, 5).fillna(0.0)

    # NOW filter to test period for actual trading
    if test_start is not None and not pd.isna(test_start):
        test_mask = idx_full >= test_start
    else:
        test_mask = pd.Series(True, index=idx_full)
    
    if test_end is not None and not pd.isna(test_end):
        test_mask = test_mask & (idx_full <= test_end)
    
    # Apply mask to get test period
    idx = idx_full[test_mask]
    ref = ref_full.loc[idx]
    coi = coi_full.loc[idx]
    r_btc_test = r_btc.loc[idx]
    r_coin_test = r_coin.loc[idx]
    z_btc = z_btc.loc[idx]
    z_coin = z_coin.loc[idx]
    beta = beta.loc[idx]

    # Basic safety
    if len(idx) < k + 100:
        raise ValueError(f"Not enough TEST data: {len(idx)} bars (need at least {k+100})")

    # ORIGINAL signals (we'll test both directions)
    long_pair  = (z_btc >= z_entry)  & (z_coin <  z_follow)
    short_pair = (z_btc <= -z_entry) & (z_coin > -z_follow)

    # THE KEY FIX: Proper position management to prevent overtrading
    pos = pd.Series(0, index=idx, dtype=int)
    in_trade = False
    entry_bar = -1
    
    for t in range(len(idx)):
        # If we're in a trade, manage it
        if in_trade:
            bars_held = t - entry_bar
            
            # Exit conditions:
            # 1. Held for k bars (the lag period)
            # 2. Spread has converged (both z-scores near zero)
            spread_converged = (abs(z_btc.iloc[t]) < z_close and abs(z_coin.iloc[t]) < z_close)
            time_expired = bars_held >= k
            
            if spread_converged or time_expired:
                # Exit the trade
                pos.iloc[t] = 0
                in_trade = False
                entry_bar = -1
            else:
                # Continue holding
                pos.iloc[t] = pos.iloc[t-1]
        
        # If we're not in a trade, look for entry
        else:
            # Require BOTH:
            # 1. Strong signal (high z threshold)
            # 2. Not already in a position recently (implicit from in_trade=False)
            if long_pair.iloc[t]:
                pos.iloc[t] = 1
                in_trade = True
                entry_bar = t
            elif short_pair.iloc[t]:
                pos.iloc[t] = -1
                in_trade = True
                entry_bar = t
            else:
                pos.iloc[t] = 0

    # Pair returns calculation
    h = beta.fillna(0.0)
    pair_ret = pos * (r_coin_test - h * r_btc_test)

    # Fee calculation - charge on BOTH entry AND exit
    position_change = pos.diff().fillna(pos.iloc[0])
    
    # Entry = nonzero position from zero
    # Exit = zero position from nonzero
    # Both incur fees on both legs
    entries = ((pos != 0) & (pos.shift(1).fillna(0) == 0)).astype(float)
    exits = ((pos == 0) & (pos.shift(1).fillna(0) != 0)).astype(float)
    
    # Each entry or exit trades both legs (coin + BTC hedge)
    # At 1bp per leg = 2bp per side = 4bp round trip
    fee = (entries + exits) * (2 * fee_bps / 1e4)

    # Final equity curve
    eq = (pair_ret - fee).cumsum()
    
    # Calculate stats only on test period
    total_minutes = len(idx)
    total_days = total_minutes / (60 * 24)
    
    # Count actual round trips
    actual_trades = int(entries.sum())
    
    stats = {
        "coin": Path(coin_file).stem,
        "k": int(k),
        "test_bars": len(idx),
        "test_days": round(total_days, 1),
        "trades": actual_trades,
        "ret_total": float(eq.iloc[-1]) if len(eq) > 0 else 0.0,
        "ret_annualized": float(pair_ret.mean() * 60 * 24 * 365),
        "vol_annualized": float(pair_ret.std() * np.sqrt(60 * 24 * 365)),
        "sharpe": float((pair_ret.mean() / (pair_ret.std() + 1e-12)) * np.sqrt(60 * 24 * 365)),
        "winrate": float((pair_ret[pos != 0] > 0).mean()) if (pos != 0).any() else 0.0,
        "max_drawdown": float((eq - eq.cummax()).min()),
        "avg_bars_per_trade": float((pos != 0).sum() / max(actual_trades, 1)),
    }
    return eq, pair_ret, pos, stats

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Lead–Lag spread backtest (portable).")
    ap.add_argument("--coin", required=True, help="e.g., ADAUSD.csv")
    ap.add_argument("--ref",  default=DEFAULT_REF, help="e.g., BTCUSD.csv")
    
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--k", type=int, help="lag minutes from DTW (rounded)")
    mx.add_argument("--k-from", type=str,
                    help="Path to k-table CSV (from DTW). If provided, k is read from there "
                         "for the chosen --coin and its train_end is used as default --test-start.")
    ap.add_argument("--z-lookback", type=int, default=Z_LOOKBACK,
                    help="lookback for z-scores (in 1-minute bars)")
    ap.add_argument("--beta-lookback", type=int, default=BETA_LOOKBACK,
                    help="lookback for beta (in 1-minute bars)")
    
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-follow", type=float, default=0.5)
    ap.add_argument("--z-close", type=float, default=0.5)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--out", default=str(STRATEGY_DIR / "sample_equity.csv"))
    ap.add_argument("--test-start", type=str, default=None,
                    help="ISO8601 UTC start for TEST trading window.")
    ap.add_argument("--test-end", type=str, default=None,
                    help="ISO8601 UTC end for TEST trading window (optional).")
    args = ap.parse_args()

    # Resolve k and TEST window
    test_start = pd.NaT
    test_end   = pd.NaT
    k = args.k

    if args.k_from:
        ktab = pd.read_csv(args.k_from)
        row = ktab.loc[ktab["symbol"] == os.path.splitext(args.coin)[0]]
        if row.empty:
            raise SystemExit(f"No entry for {args.coin} in {args.k_from}")
        k = int(row["k"].iloc[0])
        if args.test_start is None:
            test_start = pd.Timestamp(row["train_end"].iloc[0])
        else:
            test_start = pd.Timestamp(args.test_start)
    else:
        if args.test_start:
            test_start = pd.Timestamp(args.test_start)

    if args.test_end:
        test_end = pd.Timestamp(args.test_end)

    # Normalize to UTC
    if pd.notna(test_start) and test_start.tzinfo is None:
        test_start = test_start.tz_localize("UTC")
    if pd.notna(test_end) and test_end.tzinfo is None:
        test_end = test_end.tz_localize("UTC")

    eq, _, _, st = backtest_leadlag_spread(
        coin_file=args.coin,
        ref_file=args.ref,
        k=k,
        z_entry=args.z_entry,
        z_follow=args.z_follow,
        z_close=args.z_close,
        fee_bps=args.fee_bps,
        test_start=test_start,
        test_end=test_end,
        z_lookback=args.z_lookback,
        beta_lookback=args.beta_lookback
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(out_path, header=["equity"])
    print("[stats]", st)
    print(f"[saved] equity curve -> {out_path}")

if __name__ == "__main__":
    main()