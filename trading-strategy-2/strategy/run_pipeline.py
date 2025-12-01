from __future__ import annotations
import argparse, sys, subprocess, os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

# ---- Project-relative defaults (portable) ----
STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_DIR  = STRATEGY_DIR.parent
ANALYSIS_DIR = PROJECT_DIR / "analysis"
DATA_DIR     = PROJECT_DIR / "data" / "data_1m"
OUT_DIR      = STRATEGY_DIR / "dtw_out"
PLOTS_DIR    = STRATEGY_DIR / "strategy_plots"

DTW_SCRIPT   = ANALYSIS_DIR / "dynamic_time_warping_analysis.py"
LEADLAG_MOD  = STRATEGY_DIR / "leadlag_spread.py"

DEFAULT_REF  = "BTCUSD.csv"

# ---------- helpers ----------
def run_dtw_train(data_dir: Path,
                  ref_file: str,
                  lookback: int,
                  train_end: str|None,
                  train_ratio: float,
                  out_results: Path,
                  out_k_table: Path):
    """
    Call the DTW scanner to produce (a) per-symbol DTW metrics and
    (b) a compact k-table with columns [symbol, k, train_end].
    """
    cmd = [
        sys.executable, str(DTW_SCRIPT),
        "--data-dir", str(data_dir),
        "--ref", ref_file,
        "--lookback", str(lookback),
        "--out", str(out_results),
        "--k-out", str(out_k_table),
        "--train-ratio", str(train_ratio),
    ]
    if train_end:
        cmd += ["--train-end", train_end]
    print("[pipeline] running DTW:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def import_leadlag():
    """
    Import the backtest function from strategy/leadlag_spread.py without
    executing its CLI.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("leadlag_spread", str(LEADLAG_MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def list_coins(data_dir: Path, ref_file: str, only: list[str] | None) -> list[str]:
    if only:
        wanted = {c if c.endswith(".csv") else f"{c}.csv" for c in only}
        files = [f for f in wanted if (data_dir / f).exists()]
    else:
        files = [Path(f).name for f in glob.glob(str(data_dir / "*.csv"))]
    files = [f for f in files if f != ref_file]
    files.sort()
    return files

def resample_equity(eq: pd.Series, rule: str) -> pd.Series:
    """
    Resample a cumulative equity series by first converting to per-bar returns,
    summing within the resample bucket, then re-cumsumming back to equity.
    """
    r = eq.diff().fillna(0.0)
    r_r = r.resample(rule).sum()
    return r_r.cumsum()

# ---------- pipeline ----------
def main():
    all_eq = []
    ap = argparse.ArgumentParser(description="DTW → LeadLag pipeline (portable)")
    # DTW stage
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--lookback", type=int, default=1440, help="DTW window in 1m bars (e.g., 1440=1 day)")
    ap.add_argument("--train-end", type=str, default=None,
                    help="UTC ISO cutoff for TRAIN (optional). If absent, use --train-ratio.")
    ap.add_argument("--train-ratio", type=float, default=0.7,
                    help="If no --train-end, first ratio of overlap used for TRAIN.")
    ap.add_argument("--dtw-results", default=str(ANALYSIS_DIR / "sample_results_dtw.csv"))
    ap.add_argument("--k-table", default=str(ANALYSIS_DIR / "dtw_train_lags.csv"))

    # LeadLag stage
    ap.add_argument("--z-entry", type=float, default=1.5)
    ap.add_argument("--z-follow", type=float, default=0.5)
    ap.add_argument("--z-close", type=float, default=0.3)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--z-lookback", type=int, default=720, help="z-score window in minutes (default 12h)")
    ap.add_argument("--beta-lookback", type=int, default=720, help="beta window in minutes (default 12h)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--plots", action="store_true", help="also save PNG equity plots")
    ap.add_argument("--coins", nargs="*", default=None, help="optional list like DOGEUSD ETHUSD ...")
    ap.add_argument("--plot-collective", action="store_true",
                help="produce an overlay plot of all coin equities and a portfolio curve")
    ap.add_argument("--collective-resample", type=str, default=None,
                help="optional resample rule for collective outputs, e.g. '5min','15min','1H','1D'")
    ap.add_argument("--no-per-coin-plots", action="store_true",
                help="do not save individual PNGs; only produce the collective overlay")

    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir  = PLOTS_DIR;          plots_dir.mkdir(parents=True, exist_ok=True)
    results_csv= Path(args.dtw_results)
    k_table_csv= Path(args.k_table)

    # 1) DTW TRAIN
    run_dtw_train(data_dir=data_dir,
                  ref_file=args.ref,
                  lookback=args.lookback,
                  train_end=args.train_end,
                  train_ratio=args.train_ratio,
                  out_results=results_csv,
                  out_k_table=k_table_csv)

    # 2) Load k-table
    ktab = pd.read_csv(k_table_csv)
    ktab["symbol"] = ktab["symbol"].astype(str)
    k_map = dict(zip(ktab["symbol"], ktab["k"]))
    tstart_map = dict(zip(ktab["symbol"], ktab["train_end"]))

    # 3) Backtest per coin (TEST window)
    leadlag = import_leadlag()
    ref_file = args.ref
    coins = list_coins(data_dir, ref_file, args.coins)

    stats_rows = []
    for coin_file in coins:
        sym = Path(coin_file).stem
        if sym not in k_map:
            print(f"[pipeline] skip {sym}: no k in {k_table_csv.name}")
            continue
        k = int(k_map[sym])
        test_start = pd.Timestamp(tstart_map[sym])
        if test_start.tzinfo is None:
            test_start = test_start.tz_localize("UTC")

        print(f"[pipeline] {sym}: k={k}, TEST from {test_start.isoformat()}")

        try:
            # FIX: wrap in try-except to catch individual coin failures
            eq, _, _, st = leadlag.backtest_leadlag_spread(
                coin_file=coin_file,
                ref_file=ref_file,
                k=k,
                z_entry=args.z_entry,
                z_follow=args.z_follow,
                z_close=args.z_close,
                fee_bps=args.fee_bps,
                test_start=test_start,
                test_end=None,
                z_lookback=args.z_lookback,
                beta_lookback=args.beta_lookback
            )
        except Exception as e:
            print(f"  [ERROR] backtest failed for {sym}: {e}")
            continue

        # save equity curve
        eq_path = out_dir / f"equity_{sym}.csv"
        eq.to_csv(eq_path, header=["equity"])
        print(f"  -> saved equity {eq_path}")
        all_eq.append(eq.rename(sym))

        # collect stats
        st["ref"] = Path(ref_file).stem
        st["k"] = k
        st["test_start"] = test_start.isoformat()
        stats_rows.append(st)

        # optional per-coin plot (can be suppressed with --no-per-coin-plots)
        if args.plots and not args.no_per_coin_plots:
            try:
                import matplotlib.pyplot as plt
                ax = eq.rename("equity").plot(figsize=(9,3))
                ax.set_title(f"{sym} lead–lag equity (k={k})")
                ax.set_ylabel("cum. return")
                ax.grid(True)
                fig_path = plots_dir / f"equity_{sym}.png"
                plt.tight_layout(); plt.savefig(fig_path, dpi=200); plt.close()
                print(f"  -> saved plot   {fig_path}")
            except Exception as e:
                print(f"  [warn] plotting failed for {sym}: {e}")

    # 4) Plot collective (single overlay of all coins)
    if args.plot_collective and all_eq:
        try:
            import matplotlib.pyplot as plt

            # Align all equities into one DataFrame
            eq_df = pd.concat(all_eq, axis=1).sort_index()

            # Optional resampling
            if args.collective_resample:
                def _resample(series: pd.Series, rule: str) -> pd.Series:
                    r = series.diff().fillna(0.0)
                    return r.resample(rule).sum().cumsum()
                eq_df = pd.concat([_resample(eq_df[c], args.collective_resample) 
                                   for c in eq_df.columns], axis=1)

            # Normalize so all lines start at 0
            norm_df = eq_df - eq_df.iloc[0]

            # Plot all coins
            fig, ax = plt.subplots(figsize=(12, 5))
            norm_df.plot(ax=ax, linewidth=1.2, alpha=0.7)

            # FIX: Add equal-weight portfolio line
            portfolio = norm_df.mean(axis=1)
            ax.plot(portfolio.index, portfolio.values, 
                   linewidth=2.5, color='black', label='Portfolio (equal-weight)', 
                   linestyle='--', alpha=0.9)

            ax.set_title("Lead–Lag cumulative equity – all coins (normalized start)")
            ax.set_ylabel("cumulative return (normalized)")
            ax.grid(True, linestyle=":", alpha=0.5)

            # Manage legend
            ncols = max(1, min(4, int(np.ceil(len(norm_df.columns) / 8))))
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), 
                     borderaxespad=0., ncol=ncols, fontsize=9)

            fig.tight_layout()
            fig_path = plots_dir / "equity_collective.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> saved plot   {fig_path}")

        except Exception as e:
            print(f"[warn] collective plotting failed: {e}")

    # 5) Write summary
    if stats_rows:
        summary = pd.DataFrame(stats_rows)
        summary_path = out_dir / "summary_stats.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n[pipeline] summary -> {summary_path}")
        print(summary.to_string(index=False))
        
        # FIX: Add aggregate statistics
        print(f"\n[AGGREGATE STATS]")
        print(f"Total coins tested: {len(summary)}")
        print(f"Mean Sharpe: {summary['sharpe'].mean():.2f}")
        print(f"Median Sharpe: {summary['sharpe'].median():.2f}")
        print(f"Profitable coins: {(summary['ret_total'] > 0).sum()} / {len(summary)}")
        print(f"Mean total return: {summary['ret_total'].mean():.2%}")
    else:
        print("\n[pipeline] no stats produced (check data and k-table).")

if __name__ == "__main__":
    main()