from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os

# ---- project-relative paths (portable) ----
STRATEGY_DIR = Path(__file__).resolve().parent                # .../strategy
PROJECT_DIR  = STRATEGY_DIR.parent                            # .../trading-strategy-2
DATA_DIR     = PROJECT_DIR / "data" / "data_1m"               # CSVs live here
ANALYSIS_DIR = PROJECT_DIR / "analysis"                       # DTW output lives here
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
    ref = load_close(DATA_DIR / ref_file)
    coi = load_close(DATA_DIR / coin_file)

    # --- apply TEST window first ---
    if test_start is not None and not pd.isna(test_start):
        ref = ref.loc[test_start:]
        coi = coi.loc[test_start:]
    if test_end is not None and not pd.isna(test_end):
        ref = ref.loc[:test_end]
        coi = coi.loc[:test_end]

    # re-align after slicing
    idx = ref.index.intersection(coi.index)
    ref, coi = ref.loc[idx], coi.loc[idx]

    # basic safety: need enough bars after the split
    if len(ref) < k + 100 or len(coi) < k + 100:
        raise ValueError("Not enough TEST data after split for this k.")

    # --- now do the rest ---
    r_btc  = np.log(ref).diff().fillna(0.0)
    r_coin = np.log(coi).diff().fillna(0.0)

    Rb = r_btc.rolling(k).sum()
    Ri = r_coin.rolling(k).sum()
    z_btc  = roll_z(Rb, z_lookback)
    z_coin = roll_z(Ri, z_lookback)

    cov = r_coin.rolling(beta_lookback).cov(r_btc)
    var = r_btc.rolling(beta_lookback).var()
    beta = (cov / (var + 1e-12)).clip(-5, 5).fillna(0.0)

    long_pair  = (z_btc >= z_entry)  & (z_coin <  z_follow)
    short_pair = (z_btc <= -z_entry) & (z_coin > -z_follow)

    pos = pd.Series(0, index=idx, dtype=int)  # +1 long coin/short BTC, -1 short coin/long BTC
    hold = 0
    for t in range(len(idx)):
        if pos.iloc[t-1] != 0 and hold > 0:
            pos.iloc[t] = pos.iloc[t-1]
            hold -= 1
            if abs(z_coin.iloc[t] - z_btc.iloc[t]) < z_close:
                pos.iloc[t] = 0; hold = 0
        else:
            if long_pair.iloc[t]:
                pos.iloc[t] = 1;  hold = k
            elif short_pair.iloc[t]:
                pos.iloc[t] = -1; hold = k
            else:
                pos.iloc[t] = 0

    h = beta.reindex(idx).fillna(0.0)
    pair_ret = pos * (r_coin - h * r_btc)

    turns = (pos != pos.shift(1)).fillna(False)
    fee = turns.astype(float) * (2 * fee_bps / 1e4)

    eq = (pair_ret - fee).cumsum()
    stats = {
        "coin": Path(coin_file).stem,
        "k": int(k),
        "trades": int((turns & (pos != 0)).sum()),
        "ret_total": float(eq.iloc[-1]),
        "ret_annualized": float(pair_ret.mean()*60*24*365),
        "vol_annualized": float(pair_ret.std()*np.sqrt(60*24*365)),
        "sharpe": float((pair_ret.mean()/(pair_ret.std()+1e-12))*np.sqrt(60*24*365)),
        "winrate": float((pair_ret[pos!=0] > 0).mean())
    }
    return eq, pair_ret, pos, stats

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Lead–Lag spread backtest (portable).")
    ap.add_argument("--coin", required=True, help="e.g., ADAUSD.csv")
    ap.add_argument("--ref",  default=DEFAULT_REF, help="e.g., BTCUSD.csv")
    
    # --- change begins ---
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--k", type=int, help="lag minutes from DTW (rounded)")
    mx.add_argument("--k-from", type=str,
                    help="Path to k-table CSV (from DTW). If provided, k is read from there "
                         "for the chosen --coin and its train_end is used as default --test-start.")
    ap.add_argument("--z-lookback", type=int, default=Z_LOOKBACK,
                    help="lookback for z-scores (in 1-minute bars)")
    ap.add_argument("--beta-lookback", type=int, default=BETA_LOOKBACK,
                    help="lookback for beta (in 1-minute bars)")
    # --- change ends ---
    
    ap.add_argument("--z-entry", type=float, default=1.5)
    ap.add_argument("--z-follow", type=float, default=0.5)
    ap.add_argument("--z-close", type=float, default=0.3)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--out", default=str(STRATEGY_DIR / "sample_equity.csv"))
    ap.add_argument("--test-start", type=str, default=None,
                    help="ISO8601 UTC start for TEST trading window.")
    ap.add_argument("--test-end", type=str, default=None,
                    help="ISO8601 UTC end for TEST trading window (optional).")
    args = ap.parse_args()

    # --- Resolve k and TEST window ---
    test_start = pd.NaT
    test_end   = pd.NaT
    k = args.k

    if args.k_from:
        ktab = pd.read_csv(args.k_from)
        row = ktab.loc[ktab["symbol"] == os.path.splitext(args.coin)[0]]
        if row.empty:
            raise SystemExit(f"No entry for {args.coin} in {args.k_from}")
        k = int(row["k"].iloc[0])
        # default TEST window starts at train_end if user didn't pass --test-start
        if args.test_start is None:
            test_start = pd.Timestamp(row["train_end"].iloc[0])
        else:
            test_start = pd.Timestamp(args.test_start)
    else:
        # no k-table; user must provide --k and (optionally) test window
        if args.test_start:
            test_start = pd.Timestamp(args.test_start)

    if args.test_end:
        test_end = pd.Timestamp(args.test_end)

    # normalize to UTC
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
