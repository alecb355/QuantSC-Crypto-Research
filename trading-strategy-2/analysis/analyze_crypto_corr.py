import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]        # .../trading-strategy-1
DATA = ROOT / "data"

# Optional: Granger causality (skip gracefully if statsmodels isn't installed)
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ---------- config ----------
# Path to BTC file and any number of altcoin files:
BTC_PATH   = DATA / "example_btc.csv"
COIN_FILES = [DATA / "example_eth.csv", DATA / "example_sol.csv"]
MAX_LAG = 60     # minutes (lags) to search for cross-corr and Granger
OUT_PATH = ROOT / "analysis" / "sample_results_analysis.json"
# ----------------------------

def load_close_series(path, label):
    """Load CSV with columns: timestamp, open, high, low, close, volume."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].rename(columns={"close": f"{label}_close"})
    return df

def compute_log_returns(df, col):
    """Append log returns of price column to df."""
    df = df.copy()
    df[f"{col}_ret"] = np.log(df[col]).diff()
    return df

def align_and_returns(btc_df, coin_df, coin_label):
    """Inner-join on timestamp, compute BTC/coin log-returns."""
    merged = (
        coin_df.merge(btc_df, on="timestamp", how="inner")
               .sort_values("timestamp")
               .dropna()
               .reset_index(drop=True)
    )
    merged = compute_log_returns(merged, f"{coin_label}_close")
    merged = compute_log_returns(merged, "BTC_close")
    merged = merged.dropna().reset_index(drop=True)
    return merged

def best_cross_corr(coin_ret, btc_ret, max_lag=10):
    """
    Return:
      (best_lag, best_corr), (second_lag, second_corr)
    Positive lag => BTC leads coin by 'lag' periods.
    """
    lags = range(-max_lag, max_lag + 1)
    items = []
    for k in lags:
        if k < 0:
            x = coin_ret.iloc[-k:]
            y = btc_ret.iloc[:len(x)]
        elif k > 0:
            x = coin_ret.iloc[:len(coin_ret) - k]
            y = btc_ret.iloc[k:]
        else:
            x = coin_ret
            y = btc_ret
        if len(x) > 5:
            val = float(np.corrcoef(x, y)[0, 1])
            if not np.isnan(val):
                items.append((k, val))

    if not items:
        return (None, None), (None, None)

    # sort by absolute correlation (desc); tie-breaker prefers smaller |lag|
    items.sort(key=lambda kv: (abs(kv[1]), -1.0/(abs(kv[0]) + 1e-9)), reverse=True)

    best_lag, best_val = items[0]
    # pick first entry with a different lag as "second best"
    second_lag, second_val = (None, None)
    for lag, val in items[1:]:
        if lag != best_lag:
            second_lag, second_val = lag, val
            break

    return (best_lag, best_val), (second_lag, second_val)

def granger_pvalue(coin_ret, btc_ret, maxlag=10):
    """Min F-test p-value for H0: BTC does NOT Granger-cause coin."""
    if not HAS_STATSMODELS:
        return None
    data = pd.concat([coin_ret, btc_ret], axis=1)
    data.columns = ["coin_ret", "btc_ret"]
    try:
        res = grangercausalitytests(data, maxlag=min(maxlag, 10), verbose=False)
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, min(maxlag, 10) + 1)]
        return float(np.nanmin(pvals))
    except Exception:
        return None

def analyze_coin(btc_df, coin_df, coin_label, max_lag=10):
    df = align_and_returns(btc_df, coin_df, coin_label)
    coin_ret = df[f"{coin_label}_close_ret"]
    btc_ret = df["BTC_close_ret"]

    corr = float(coin_ret.corr(btc_ret))
    (best_lag, best_corr), (second_lag, second_corr) = best_cross_corr(coin_ret, btc_ret, max_lag=max_lag)
    pval = granger_pvalue(coin_ret, btc_ret, maxlag=max_lag)

    return {
        coin_label.upper(): {
            "correlation": round(corr, 4) if not np.isnan(corr) else None,
            "lag_with_highest_corr": int(best_lag) if best_lag is not None else None,
            "cross_corr_at_best_lag": round(best_corr, 4) if best_corr is not None and not np.isnan(best_corr) else None,
            "second_best_lag": int(second_lag) if second_lag is not None else None,
            "cross_corr_at_second_best_lag": round(second_corr, 4) if second_corr is not None and not np.isnan(second_corr) else None,
            "granger_p_value": round(pval, 4) if pval is not None else None,
        }
    }


btc_df = load_close_series(BTC_PATH, "BTC")
results = {}
for path in COIN_FILES:
    label = os.path.splitext(os.path.basename(path))[0].split("_")[-1]  # e.g., "eth" from "example_eth.csv"
    coin_df = load_close_series(path, label.upper())
    res = analyze_coin(btc_df, coin_df, label.upper(), max_lag=MAX_LAG)
    results.update(res)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {OUT_PATH}")
print(json.dumps(results, indent=2))