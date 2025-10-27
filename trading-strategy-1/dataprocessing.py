import pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- 1) Load wallets & transactions ---
wallets = pd.read_json("example_wallet_list.json")
tx = pd.read_json("example_transaction_list.json")

# wallet labels
wallets["address"] = wallets["address"].astype(str).str.lower()
wallets["label"]   = wallets["label"].astype(str).str.lower()
wallets["type"]    = wallets["type"].astype(str).str.lower()

addr2type  = wallets.set_index("address")["type"].to_dict()
label2type = wallets.set_index("label")["type"].to_dict()

# normalize tx columns
tx["from_address"] = tx["from_address"].astype(str).str.lower()
tx["to_address"]   = tx["to_address"].astype(str).str.lower()
tx["from_type"]    = tx.get("from_type", "").astype(str).str.lower()
tx["to_type"]      = tx.get("to_type", "").astype(str).str.lower()
tx["amount"]       = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)

mask_from = (tx["from_type"].isna()) | (tx["from_type"]=="") | (tx["from_type"]=="other")
mask_to   = (tx["to_type"].isna())   | (tx["to_type"]=="")   | (tx["to_type"]=="other")

tx.loc[mask_from, "from_type"] = tx.loc[mask_from, "from_address"].map(addr2type)
tx.loc[mask_from, "from_type"] = tx.loc[mask_from, "from_type"].fillna(
    tx.loc[mask_from, "from_address"].map(label2type)
)
tx.loc[mask_to, "to_type"] = tx.loc[mask_to, "to_address"].map(addr2type)
tx.loc[mask_to, "to_type"] = tx.loc[mask_to, "to_type"].fillna(
    tx.loc[mask_to, "to_address"].map(label2type)
)

tx["from_type"] = tx["from_type"].fillna("other")
tx["to_type"]   = tx["to_type"].fillna("other")

# directional flows per bar
# --- build three signed-flow variants (work even if only one side is labeled) ---
# bar size must match OHLCV (5m)
tx["ts"]  = pd.to_datetime(tx["timestamp"], utc=True)
tx["bar"] = tx["ts"].dt.floor("5min")

tx["flow_strict"] = 0.0
tx.loc[(tx["from_type"]=="whale") & (tx["to_type"]=="exchange"), "flow_strict"]  =  tx["amount"]
tx.loc[(tx["from_type"]=="exchange") & (tx["to_type"]=="whale"), "flow_strict"]  = -tx["amount"]

tx["flow_whale_only"] = 0.0
tx.loc[(tx["from_type"]=="whale") & (tx["to_type"]!="whale"), "flow_whale_only"] =  tx["amount"]
tx.loc[(tx["to_type"]=="whale")   & (tx["from_type"]!="whale"), "flow_whale_only"] = -tx["amount"]

tx["flow_ex_only"] = 0.0
tx.loc[(tx["to_type"]=="exchange")   & (tx["from_type"]!="exchange"), "flow_ex_only"] =  tx["amount"]
tx.loc[(tx["from_type"]=="exchange") & (tx["to_type"]!="exchange"),   "flow_ex_only"] = -tx["amount"]

flows = (
    tx.groupby("bar")[["flow_strict","flow_whale_only","flow_ex_only"]]
      .sum()
      .reset_index()
      .rename(columns={
          "flow_strict":"netflow_strict",
          "flow_whale_only":"netflow_whale_only",
          "flow_ex_only":"netflow_ex_only"
      })
).sort_values("bar")

# debugging helpers (optional)
inflow  = tx[(tx["from_type"]=="whale") & (tx["to_type"]=="exchange")].groupby("bar")["amount"].sum().rename("wh_to_ex_in")
outflow = tx[(tx["from_type"]=="exchange") & (tx["to_type"]=="whale")].groupby("bar")["amount"].sum().rename("ex_to_wh_out")
flows = flows.merge(pd.concat([inflow, outflow], axis=1).reset_index(), on="bar", how="left")

for c in flows.columns:
    if c != "bar":
        flows[c] = pd.to_numeric(flows[c], errors="coerce").fillna(0.0)

# ensure bar dtype
flows["bar"] = pd.to_datetime(flows["bar"], utc=True)

# --- 2) Import OHLCV (clean the 'sol-usd' label row, coerce numerics) ---
ohlcv = pd.read_csv("SOLUSD_5m_yahoo.csv")

# drop any junk label rows (e.g., ",sol-usd,sol-usd,...")
for c in ["open","high","low","close","volume"]:
    if c in ohlcv.columns:
        ohlcv[c] = pd.to_numeric(ohlcv[c], errors="coerce")

# parse bar as UTC and keep only valid rows
ohlcv["bar"] = pd.to_datetime(ohlcv["bar"], utc=True, errors="coerce")
ohlcv = ohlcv.dropna(subset=["bar","open","high","low","close"])
ohlcv = ohlcv[["bar","open","high","low","close","volume"]].sort_values("bar")

# --- 3) MERGE prices + flows → this creates df ---
flows["bar"] = pd.to_datetime(flows["bar"], utc=True)
df = ohlcv.merge(flows, on="bar", how="left")

# choose which netflow variant to use (you added these earlier)
if "netflow_strict" in df.columns:
    df["netflow"] = df["netflow_strict"]
elif "netflow_whale_only" in df.columns:
    df["netflow"] = df["netflow_whale_only"]
elif "netflow_ex_only" in df.columns:
    df["netflow"] = df["netflow_ex_only"]
else:
    df["netflow"] = 0.0  # fallback if none present

# fill only numeric columns; keep 'bar' as datetime
num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].fillna(0.0)
df = df.sort_values("bar")

# Targets
df["ret"] = df["close"].pct_change()
y = df["ret"].shift(-1)

def scan_single_flow_lag(df: pd.DataFrame,
                         flow_col: str = "netflow",
                         max_lag: int = 12,
                         include_ret_lb1: bool = True,
                         cov_type: str = "HC1") -> pd.DataFrame:
    """
    For k=0..max_lag, fit:  ret_{t+1} = a + b_k * flow_{t-k} + d * ret_t (+ const)
    Returns a table with k, rows used, beta_k, se, t, p, R2.
    """
    df = df.sort_values("bar").copy()
    if "ret" not in df:
        df["ret"] = df["close"].pct_change()
    y = df["ret"].shift(-1)

    out = []
    for k in range(max_lag + 1):
        # Build X with exactly one flow lag: flow_{t-k}
        X = pd.DataFrame({f"{flow_col}_lag{k}": df[flow_col].shift(k)})
        if include_ret_lb1:
            X["ret_lb1"] = df["ret"].shift(1)

        # Align X & y; drop any NaNs
        M = pd.concat([X, y.rename("y")], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
        if M.shape[0] <= (M.shape[1] + 2):
            out.append({"k": k, "rows": int(M.shape[0]),
                        "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                        "R2": np.nan, "adjR2": np.nan})
            continue

        Y = M["y"].astype(float)
        Z = sm.add_constant(M.drop(columns=["y"]).astype(float))
        # Drop any constant columns (safety)
        keep = ["const"] + [c for c in Z.columns if c != "const" and Z[c].std() > 0]
        Z = Z[keep]

        # Fit
        if cov_type.lower() == "nonrobust":
            res = sm.OLS(Y, Z).fit()
        else:
            res = sm.OLS(Y, Z).fit(cov_type="HC1")

        flow_name = f"{flow_col}_lag{k}"
        beta = res.params.get(flow_name, np.nan)
        se   = res.bse.get(flow_name, np.nan) if hasattr(res, "bse") else np.nan
        tval = res.tvalues.get(flow_name, np.nan) if hasattr(res, "tvalues") else np.nan
        pval = res.pvalues.get(flow_name, np.nan) if hasattr(res, "pvalues") else np.nan

        out.append({
            "k": k, "rows": int(res.nobs),
            "beta": float(beta), "se": float(se), "t": float(tval), "p": float(pval),
            "R2": float(res.rsquared), "adjR2": float(res.rsquared_adj)
        })

    return pd.DataFrame(out).sort_values("k").reset_index(drop=True)


def plot_flow_sensitivity(tbl, title="Flow → Next-bar Return (β by lag)"):
    # keep only rows with finite estimates
    td = tbl.replace([np.inf, -np.inf], np.nan).dropna(subset=["k","beta","se","t","p"])
    if td.empty:
        print("[warn] nothing to plot (all NaN).")
        return

    # 95% CI
    td = td.copy()
    td["ci_lo"] = td["beta"] - 1.96*td["se"]
    td["ci_hi"] = td["beta"] + 1.96*td["se"]

    # --- Plot 1: beta ± CI vs lag ---
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.axhline(0, lw=1, ls="--")
    ax.errorbar(td["k"], td["beta"], yerr=1.96*td["se"], fmt="o-", capsize=3)
    ax.set_xlabel("lag k (bars)")
    ax.set_ylabel("beta on flow_{t-k}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # annotate N used
    for _, r in td.iterrows():
        ax.annotate(int(r["rows"]), (r["k"], r["beta"]), xytext=(0,6),
                    textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- Plot 2 (optional): t-stat vs lag ---
    fig2, ax2 = plt.subplots(figsize=(8,3.6))
    ax2.axhline(0, lw=1, ls="--")
    ax2.plot(td["k"], td["t"], "o-")
    ax2.set_xlabel("lag k (bars)")
    ax2.set_ylabel("t-stat")
    ax2.set_title("t-stat by lag (HC1 or nonrobust)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example run
# 1) build the table for k = 0..12 (uses ret_{t+1} target)
tbl = scan_single_flow_lag(df, flow_col="netflow", max_lag=12,
                           include_ret_lb1=True, cov_type="HC1")
print(tbl.to_string(index=False))

# 2) plot
plot_flow_sensitivity(tbl, title="SOL: flow_t-k → ret_{t+1} (5-min)")

# Predictors for a single flow lag
# lags = 1  # start with 0 or 1 for stability; increase later
# X = pd.concat([df["netflow"].shift(j) for j in range(lags)], axis=1) if lags>0 else pd.DataFrame()
# if lags>0: X.columns = [f"net_lag{j}" for j in range(lags)]
# X["ret_lb1"] = df["ret"].shift(1)
# 
# # Align X and y together, drop rows with any NaN
# M = pd.concat([X, y.rename("y")], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
# 
# if len(M) > 5:
#     Y = M["y"].astype(float)
#     Z = sm.add_constant(M.drop(columns=["y"]).astype(float))
# 
#     print("Z std:\n", Z.drop(columns=["const"]).std())
# 
#     # (A) plain OLS (should give finite params even if robust fails)
#     res_nr = sm.OLS(Y, Z).fit()
#     print("\n--- NONROBUST OLS ---")
#     print(res_nr.params)
# 
#     # (B) robust (may be flaky with ultra-sparse spikes)
#     res_hc1 = sm.OLS(Y, Z).fit(cov_type="HC1")
#     print("\n--- HC1 OLS ---")
#     print(res_hc1.params)
#     print(res_hc1.tvalues)
#     print(res_hc1.pvalues)
# else:
#     print("[warn] Not enough aligned rows; reduce lags or widen window.")
