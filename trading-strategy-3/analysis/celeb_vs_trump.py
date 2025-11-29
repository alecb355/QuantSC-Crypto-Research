import argparse, glob, os, math
from pathlib import Path
import numpy as np
import pandas as pd

# Try fast DTW; fall back to exact O(L^2)
try:
    from fastdtw import fastdtw
    _USE_FASTDTW = True
except Exception:
    _USE_FASTDTW = False

# ---------- IO helpers ----------
TIME_COLS  = ["time", "timestamp", "Datetime", "date", "bar"]
CLOSE_COLS = ["close", "Close", "price"]

def load_price_1m(csv_path: Path, resample="1min") -> pd.Series:
    """Read a time,close CSV → UTC DatetimeIndex series on a uniform 1-minute grid."""
    df = pd.read_csv(csv_path)
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    ccol = next((c for c in CLOSE_COLS if c in df.columns), None)
    if tcol is None or ccol is None:
        raise ValueError(f"{csv_path.name}: need time + close columns; got {df.columns.tolist()}")

    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    px = pd.to_numeric(df[ccol], errors="coerce")
    s = pd.Series(px.values, index=ts).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if resample:
        s = s.asfreq(resample, method="pad")
    return s.dropna()

def to_T0_window(s: pd.Series, horizon: int, freq: str = "1min") -> pd.Series:
    """
    Align a price series to T0 (its first timestamp), resample to 1-minute bars,
    convert the index to 'minutes since T0', clip to [0, horizon], and normalize.
    """
    # clean + resample
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s.asfreq(freq, method="pad")

    # relative minutes since T0  (this line is the key fix)
    t0 = s.index[0]
    rel_min = ((s.index - t0) / pd.Timedelta(minutes=1)).astype(int)  # <- no timedelta64[m] cast

    # reindex to 0..horizon and normalize to 1.0 at T0
    s.index = rel_min
    s = s.loc[:horizon]
    return s / float(s.iloc[0])

def norm_start_1(s: pd.Series) -> pd.Series:
    """Normalize so value at minute 0 equals 1.0."""
    if len(s) == 0:
        return s
    base = float(s.iloc[0])
    return s / (base if base != 0 else 1.0)

# ---------- similarity metrics ----------
def dtw_distance_and_lag(a: np.ndarray, b: np.ndarray):
    """
    Returns (distance, effective_lag_minutes). Positive lag ⇒ a leads b.
    """
    A = np.asarray(a, float).reshape(-1)
    B = np.asarray(b, float).reshape(-1)

    # z-normalize to compare shape
    def z(x):
        m, s = np.nanmean(x), np.nanstd(x)
        return (x - m) / (s + 1e-12)
    A, B = z(A), z(B)

    if _USE_FASTDTW:
        # scalar distance between points
        dist, path = fastdtw(A, B, dist=lambda x, y: abs(float(x) - float(y)))
        lag = float(np.mean([j - i for (i, j) in path])) if path else 0.0
        return float(dist), lag

    # exact DTW (O(N^2)) fallback
    n, m = len(A), len(B)
    D = np.full((n+1, m+1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        ai = A[i-1]
        for j in range(1, m+1):
            cost = abs(ai - B[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # backtrack path to estimate lag
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

# ---------- main ----------
def main():
    here = Path(__file__).resolve().parent
    project = here.parent
    default_data = project / "data" / "coin_data"
    default_out  = here / "celeb_vs_trump_results.csv"
    default_aligned = here / "aligned_prices.csv"

    ap = argparse.ArgumentParser(description="Compare celeb coins to TRUMP after launch (T0 alignment).")
    ap.add_argument("--data-dir", default=str(default_data))
    ap.add_argument("--ref", default="TRUMP.csv", help="reference filename inside data-dir")
    ap.add_argument("--horizon", type=int, default=360, help="minutes after launch to compare")
    ap.add_argument("--out", default=str(default_out), help="per-coin similarity table CSV")
    ap.add_argument("--aligned", default=str(default_aligned), help="aligned normalized price paths CSV")
    ap.add_argument("--plot", action="store_true", help="save overlay plot PNG")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ref_path = data_dir / args.ref
    if not ref_path.exists():
        raise SystemExit(f"Reference file not found: {ref_path}")

    # Load & prepare TRUMP path (T0-aligned, normalized)
    trump = load_price_1m(ref_path)
    trump = to_T0_window(trump, args.horizon)
    trump = norm_start_1(trump)

    # Gather all other CSVs
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    rows = []
    aligned = {"TRUMP": trump}

    for f in files:
        fname = os.path.basename(f)
        sym = os.path.splitext(fname)[0]
        if fname == args.ref:
            continue
        try:
            s = load_price_1m(Path(f))
            s = to_T0_window(s, args.horizon)
            s = norm_start_1(s)

            # align indices (minutes since each coin's launch)
            # inner join on common minutes [0..H-1]
            join = pd.concat([trump.rename("TRUMP"), s.rename(sym)], axis=1).dropna()
            if len(join) < min(60, args.horizon//6):  # require a little overlap
                print(f"[skip] {sym}: too little overlap with TRUMP window ({len(join)} mins)")
                continue

            a = join["TRUMP"].to_numpy(float)
            b = join[sym].to_numpy(float)

            # similarity metrics
            corr = float(pd.Series(a).corr(pd.Series(b)))
            dist, lag = dtw_distance_and_lag(a, b)

            rows.append({
                "coin": sym,
                "overlap_min": int(len(join)),
                "corr_vs_TRUMP": corr,
                "dtw_distance": dist,
                "dtw_lag_minutes_(+TRUMP_leads)": lag
            })

            # store aligned path on the canonical 0..H-1 grid for plotting/export
            aligned[sym] = s.reindex(trump.index)  # same minute grid
        except Exception as e:
            print(f"[skip] {sym}: {e}")

    # Save aligned normalized prices
    aligned_df = pd.DataFrame(aligned).sort_index()
    aligned_df.to_csv(args.aligned, index_label="minute_since_launch")
    print(f"[saved] aligned normalized prices -> {args.aligned}")

    # Save summary
    out = pd.DataFrame(rows).sort_values(["dtw_distance", "coin"])
    out.to_csv(args.out, index=False)
    print(f"[saved] similarity table -> {args.out}")
    if not out.empty:
        print(out.head(12).to_string(index=False))

    # Optional overlay plot
    if args.plot and not aligned_df.empty:
        try:
            import matplotlib.pyplot as plt
            ax = aligned_df.plot(figsize=(10,4), linewidth=1)
            ax.set_title(f"Normalized price after launch (first {args.horizon} minutes)")
            ax.set_ylabel("normalized price (T0 = 1.0)")
            ax.set_xlabel("minutes since launch (T0)")
            ax.grid(True, alpha=0.3)
            png_path = here / "overlay_aligned_prices.png"
            plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close()
            print(f"[saved] overlay plot -> {png_path}")
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

if __name__ == "__main__":
    main()
