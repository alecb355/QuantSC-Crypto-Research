import argparse, os, glob, math
import numpy as np, pandas as pd
from pathlib import Path

# Paths relative to this script, so it works anywhere
SCRIPT_DIR   = Path(__file__).resolve().parent                  # …/trading-strategy-2/analysis
PROJECT_DIR  = SCRIPT_DIR.parent                                 # …/trading-strategy-2
DEFAULT_DATA = PROJECT_DIR / "data" / "data_1m"                  # …/trading-strategy-2/data/data_1m
DEFAULT_OUT  = SCRIPT_DIR / "sample_results_dtw.csv"

# Try fast DTW; fall back to exact O(L^2) DTW if not installed
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    _USE_FASTDTW = True
except Exception:
    _USE_FASTDTW = False

# -------------------------
# Utilities
# -------------------------
COMMON_TIME_COLS = ["bar","timestamp","time","Datetime","date"]
COMMON_CLOSE_COLS = ["close","Close","closing_price"]

def load_close_series(path: str) -> pd.Series:
    """Load a CSV and return UTC 1-min close series with a DatetimeIndex."""
    df = pd.read_csv(path)
    # time col
    tcol = next((c for c in COMMON_TIME_COLS if c in df.columns), None)
    if tcol is None:
        # try index if reset_index() produced 'index'
        tcol = "index" if "index" in df.columns else None
    if tcol is None:
        raise ValueError(f"Could not find a timestamp column in {path}")
    # close col
    ccol = next((c for c in COMMON_CLOSE_COLS if c in df.columns), None)
    if ccol is None:
        raise ValueError(f"Could not find a close column in {path}")

    # parse timestamps -> DatetimeIndex
    ser = pd.to_datetime(df[tcol], utc=True, errors="coerce")          # already a Series
    idx = pd.DatetimeIndex(ser.values, tz="UTC")
    
    # close as numeric series aligned to the datetime index
    vals = pd.to_numeric(df[ccol], errors="coerce")
    out = pd.Series(vals.values, index=idx)
    
    # clean: drop NaNs/dupes, sort, enforce 1-minute freq (pad tiny gaps)
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.asfreq("1min", method="pad")

    return out

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-12)

def logret(series: pd.Series) -> pd.Series:
    return np.log(series).diff().dropna()

def dtw_distance_and_lag(a, b):
    """
    a, b: array-like (price returns or similar). We coerce to 1-D float.
    Returns (distance, effective_lag) with lag>0 meaning 'a leads b'.
    """
    A = np.asarray(a, dtype=float).reshape(-1)   # <-- force 1-D float
    B = np.asarray(b, dtype=float).reshape(-1)

    # z-normalize (still 1-D)
    muA, sdA = np.nanmean(A), np.nanstd(A)
    muB, sdB = np.nanmean(B), np.nanstd(B)
    A = (A - muA) / (sdA + 1e-12)
    B = (B - muB) / (sdB + 1e-12)

    if _USE_FASTDTW:
        # scalar distance (works for 1-D time series)
        _scalar_dist = lambda x, y: abs(float(x) - float(y))
        dist, path = fastdtw(A, B, dist=_scalar_dist)
        lag = float(np.mean([j - i for (i, j) in path])) if path else 0.0
        return float(dist), lag

    # exact DTW fallback (unchanged)
    n, m = len(A), len(B)
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(A[i-1] - B[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else: i -= 1; j -= 1
    path.reverse()
    lag = float(np.mean([j - i for (i, j) in path])) if path else 0.0
    return float(D[n, m]), lag

def pick_train_cutoff(overlap_index: pd.DatetimeIndex, train_end: str | None, train_ratio: float) -> pd.Timestamp:
    if train_end:
        ts = pd.Timestamp(train_end)
        cut = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
        return cut
    n = len(overlap_index)
    j = max(1, int(n * train_ratio))
    return overlap_index[min(j, n-1)]

# -------------------------
# Main scan
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="DTW scan vs BTC for 1m CSVs")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA), help="folder with *.csv")
    ap.add_argument("--ref",      default="BTCUSD.csv",     help="reference CSV filename for BTC")
    ap.add_argument("--lookback", type=int, default=1440,   help="bars to compare (e.g., 1440=1d of 1m)")
    ap.add_argument("--out",      default=str(DEFAULT_OUT), help="output CSV")
    ap.add_argument("--lead_sigma", type=float, default=1.0,
                    help="z-threshold for simple lead-lag momentum hint")
    ap.add_argument("--train-end", type=str, default=None,
                    help="ISO8601 cutoff in UTC for training (e.g., 2025-10-15T00:00:00Z). "
                         "DTW is computed only up to this time.")
    ap.add_argument("--train-ratio", type=float, default=0.7,
                    help="If --train-end is not given, use the first train_ratio of overlapping bars as train.")
    ap.add_argument("--k-out", default="analysis/dtw_train_lags.csv",
                    help="Where to save the per-symbol k (lag_bars_abs) estimated on the train window.")
    args = ap.parse_args()

    # find files
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {args.data_dir}")
    ref_path = os.path.join(args.data_dir, args.ref)
    if not os.path.exists(ref_path):
        raise SystemExit(f"Reference file not found: {ref_path}")

    # load reference
    ref_close = load_close_series(ref_path)
    ref_ret = logret(ref_close)

    results = []
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        if os.path.samefile(f, ref_path):
            continue
        try:
            s = load_close_series(f)
            r = logret(s)
        except Exception as e:
            print(f"[skip] {sym}: {e}")
            continue

        # align last lookback window
        joined = pd.concat([ref_ret.rename("btc"), r.rename("x")], axis=1).dropna()
        if len(joined) < args.lookback + 10:
            print(f"[skip] {sym}: not enough overlapping bars ({len(joined)})")
            continue

        # ----- TRAIN/TEST SPLIT -----
        # choose train cutoff (explicit --train-end or by ratio)
        cutoff = pick_train_cutoff(joined.index, args.train_end, args.train_ratio)

        # TRAIN window: only data <= cutoff, last lookback bars
        joined_train = joined.loc[:cutoff]
        if len(joined_train) < args.lookback:
            print(f"[skip] {sym}: not enough TRAIN bars before cutoff ({len(joined_train)})")
            continue
        window_train = joined_train.tail(args.lookback)
        a = window_train["btc"].values
        b = window_train["x"].values

        dist, lag = dtw_distance_and_lag(a, b)   # lag>0 ⇒ BTC leads sym (on TRAIN only)
        k = int(max(1, round(abs(lag))))

        # TEST window is everything after cutoff (we don’t use it here, just record sizes)
        joined_test = joined.loc[cutoff:]
        test_len = int(len(joined_test))
        # simple lead-lag hint: if BTC up over |lag| bars by > lead_sigma, follower might follow
        r_btc_k = window_train["btc"].tail(k).to_numpy(dtype=float).ravel()
        z_btc_k = (r_btc_k.sum() - r_btc_k.mean()*k) / (r_btc_k.std()*math.sqrt(k) + 1e-12)

        if lag > 0 and z_btc_k > args.lead_sigma:
            hint = "BTC_up→long_symbol"
        elif lag > 0 and z_btc_k < -args.lead_sigma:
            hint = "BTC_down→short_symbol"
        else:
            hint = ""

        results.append({
            "symbol": sym,
            "dtw_distance": dist,
            "effective_lag_btc_leads": lag,
            "lag_bars_abs": k,
            "train_end": cutoff.isoformat(),
            "train_size": int(len(joined_train)),
            "test_size": test_len,
            "overlap_bars": int(len(joined))
        })

    out = pd.DataFrame(results).sort_values(["dtw_distance","symbol"])
    # Also save compact per-symbol K table for downstream execution
    ktab = out[["symbol", "lag_bars_abs", "train_end"]].rename(columns={"lag_bars_abs":"k"})
    os.makedirs(os.path.dirname(args.k_out), exist_ok=True)
    ktab.to_csv(args.k_out, index=False)
    print(f"Saved k table: {args.k_out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
