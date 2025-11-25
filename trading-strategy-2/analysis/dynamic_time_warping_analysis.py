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
        window = joined.tail(args.lookback)
        a = window["btc"].to_numpy(dtype=float).ravel()
        b = window["x"].to_numpy(dtype=float).ravel()


        dist, lag = dtw_distance_and_lag(a, b)   # lag>0 ⇒ BTC leads sym
        # simple lead-lag hint: if BTC up over |lag| bars by > lead_sigma, follower might follow
        k = int(max(1, round(abs(lag))))
        r_btc_k = window["btc"].tail(k).to_numpy(dtype=float).ravel()
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
            "btc_kbar_zmove": float(z_btc_k),
            "leadlag_hint": hint,
            "overlap_bars": int(len(window))
        })

    out = pd.DataFrame(results).sort_values(["dtw_distance","symbol"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
