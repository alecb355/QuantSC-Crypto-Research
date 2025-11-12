#!/usr/bin/env python3
"""
Top-50 Crypto (by market cap) 1-minute OHLCV (last 365 days)
- Uses CoinGecko to get the Top-50 list (ranking snapshot).
- Uses Binance (or Binance.US) REST to download 1-minute klines for each mapped symbol.
- Outputs one CSV per symbol, matching the format:
    timestamp,open,high,low,close,volume
  where timestamp is UTC in '%Y-%m-%d %H:%M:00' (minute precision).

USAGE
-----
# 1) Install deps (Python 3.9+ recommended)
python -m pip install --upgrade pip
pip install requests pandas

# 2) Run (defaults to global Binance)
python top50_minute_pipeline.py

# Optional flags
python top50_minute_pipeline.py --exchange binance.us --quote USD --extra-quotes "USDT,USDC" --days 365 --outdir data_1m

NOTES
-----
- CoinGecko cannot provide 1-minute bars for a full year on public/pro plans.
  We therefore use CoinGecko only for the Top-50 ranking, and fetch OHLCV
  from an exchange (Binance or Binance.US) that provides minute klines.
- This script respects Binance row limits (max 1000 klines per request) by paging.
- It writes CSV incrementally in batches to avoid excess memory usage.
"""

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_BASES = {
    "binance": "https://api.binance.com",
    "binance.us": "https://api.binance.us",
}


def get_top_coins(n: int = 50, vs_currency: str = "usd") -> List[Dict]:
    """
    Fetch Top-N coins by market cap from CoinGecko (ranking snapshot).
    Returns a list of dicts containing 'id', 'symbol', 'name', etc.
    """
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "price_change_percentage": "24h",
    }
    r = requests.get(COINGECKO_MARKETS_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def load_exchange_symbols(exchange: str) -> Dict[str, Dict]:
    """
    Load exchange symbols map from Binance(US) /api/v3/exchangeInfo.

    Returns dict: symbol_str -> symbol_info
    Only includes symbols with status='TRADING' and spot 'isSpotTradingAllowed' if present.
    """
    base = BINANCE_BASES[exchange]
    url = f"{base}/api/v3/exchangeInfo"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    symbols = {}
    for s in data.get("symbols", []):
        status_ok = s.get("status") == "TRADING"
        spot_ok = s.get("isSpotTradingAllowed", True)
        if status_ok and spot_ok:
            symbols[s["symbol"]] = s
    return symbols


def map_coin_to_symbol(coin_symbol: str, symbols_map: Dict[str, Dict], quotes: List[str]) -> Optional[str]:
    """
    Try to map a CoinGecko coin symbol (e.g., 'btc') to an exchange symbol like 'BTCUSDT'.
    Tries each quote in quotes order and returns the first available match.
    """
    base = coin_symbol.upper()
    for q in quotes:
        sym = f"{base}{q}"
        if sym in symbols_map:
            return sym
    return None


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fmt_minute(ms_epoch: int) -> str:
    """
    Format milliseconds since epoch to '%Y-%m-%d %H:%M:00' in UTC.
    Kline open times are aligned to minute boundaries; we zero seconds.
    """
    dt = datetime.utcfromtimestamp(ms_epoch / 1000.0).replace(second=0, microsecond=0)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def fetch_klines_paged(
    exchange: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1m",
    limit: int = 1000,
    max_retries: int = 5,
    backoff_base: float = 0.5,
):
    """
    Generator yielding kline batches from Binance(US) REST /api/v3/klines.

    Advances by last open time + 1 ms to avoid duplicates.
    Each item is the raw JSON array for one kline, shape:
        [ openTime, open, high, low, close, volume, closeTime, ... ]

    Rate-limit & transient error handling:
    - Retries 429 and 5xx with exponential backoff (backoff_base * 2**attempt).
    """
    base = BINANCE_BASES[exchange]
    url = f"{base}/api/v3/klines"
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=60)
                # 429 rate limit handling
                if resp.status_code == 429:
                    sleep_s = backoff_base * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    return
                yield batch
                # advance
                last_open = batch[-1][0]
                cursor = last_open + 1
                break
            except requests.HTTPError:
                # Retry 5xx responses; anything else re-raise
                if resp is not None and 500 <= resp.status_code < 600:
                    sleep_s = backoff_base * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue
                raise
            except requests.RequestException:
                # network hiccup, retry
                sleep_s = backoff_base * (2 ** attempt)
                time.sleep(sleep_s)
                continue


def write_symbol_csv(
    outdir: Path,
    symbol: str,
    batches: List[List],
    append: bool = True
) -> int:
    """
    Write klines batches to CSV in the format:
    timestamp,open,high,low,close,volume

    Returns number of rows written.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{symbol}.csv"
    write_header = not append or not path.exists()
    rows_written = 0
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for batch in batches:
            for k in batch:
                # kline fields
                ot, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
                w.writerow([fmt_minute(ot), o, h, l, c, v])
                rows_written += 1
    return rows_written


def run(
    exchange: str,
    vs_currency: str,
    quote: str,
    extra_quotes: List[str],
    days: int,
    outdir: Path,
    include_stablecoins: bool = False,
    limit_symbols: Optional[int] = None,
):
    print(f"[INFO] Exchange: {exchange}  Quote: {quote}  Days: {days}")
    quotes = [quote]
    for q in extra_quotes:
        q_up = q.upper()
        if q_up not in quotes:
            quotes.append(q_up)

    # 1) Top-N list from CoinGecko
    top = get_top_coins(50, vs_currency=vs_currency)
    print(f"[INFO] Retrieved Top-{len(top)} list from CoinGecko")

    # Optional filter to skip stablecoins by symbol name heuristic
    if not include_stablecoins:
        stable_syms = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDD", "GUSD", "PYUSD"}
        before = len(top)
        top = [c for c in top if c.get("symbol", "").upper() not in stable_syms]
        after = len(top)
        if after < before:
            print(f"[INFO] Skipped likely stablecoins: {before - after} removed")

    # 2) Load exchange symbols
    symbols_map = load_exchange_symbols(exchange)
    print(f"[INFO] Loaded {len(symbols_map)} tradable symbols from {exchange}")

    # 3) Map coins to exchange symbols
    mapping = []
    for c in top:
        cg_id = c.get("id")
        cg_sym = (c.get("symbol") or "").strip()
        if not cg_sym:
            continue
        exch_sym = map_coin_to_symbol(cg_sym, symbols_map, quotes)
        mapping.append((cg_id, cg_sym.upper(), exch_sym))

    mapped = [(a, b, s) for (a, b, s) in mapping if s]
    skipped = [(a, b) for (a, b, s) in mapping if not s]

    if limit_symbols:
        mapped = mapped[:limit_symbols]

    print("[INFO] Symbol mapping (first 15):")
    for item in mapped[:15]:
        print("   ", item)
    if skipped:
        print(f"[WARN] {len(skipped)} top coins not found on {exchange} with quotes {quotes}. Example:", skipped[:10])

    # 4) Time window
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms, end_ms = ms(start), ms(now)
    print(f"[INFO] Time window UTC: {start.isoformat()} -> {now.isoformat()}")

    # 5) Fetch & write per symbol (streaming)
    for (_cg_id, base_sym, exch_sym) in mapped:
        print(f"[INFO] Downloading {exch_sym} 1m klines...")
        rows_written = 0
        batch_buffer: List[List] = []
        out_path = outdir / f"{exch_sym}.csv"
        if out_path.exists():
            out_path.unlink()  # start fresh per run

        for batch in fetch_klines_paged(exchange, exch_sym, start_ms, end_ms, interval="1m", limit=1000):
            batch_buffer.append(batch)
            # write in chunks to keep memory low
            buffered_rows = sum(len(b) for b in batch_buffer)
            if buffered_rows >= 10000:  # ~10k rows per flush
                rows_written += write_symbol_csv(outdir, exch_sym, batch_buffer, append=True)
                batch_buffer = []
                print(f"   [..] {exch_sym}: wrote {rows_written} rows so far")

        # final flush
        if batch_buffer:
            rows_written += write_symbol_csv(outdir, exch_sym, batch_buffer, append=True)

        print(f"[DONE] {exch_sym}: total rows written = {rows_written}  -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Top-50 market-cap 1-minute OHLCV downloader")
    p.add_argument("--exchange", choices=["binance", "binance.us"], default="binance",
                   help="Which exchange REST API to use (default: binance)")
    p.add_argument("--vs-currency", default="usd", help="CoinGecko ranking currency (default: usd)")
    p.add_argument("--quote", default="USDT",
                   help="Primary quote currency to target on the exchange (default: USDT)")
    p.add_argument("--extra-quotes", default="USD,USDC,BUSD",
                   help="Comma-separated fallback quotes to try if primary quote pair not listed")
    p.add_argument("--days", type=int, default=365,
                   help="How many trailing days of 1m data to fetch (default: 365)")
    p.add_argument("--outdir", default="data_1m",
                   help="Output directory for CSV files (default: data_1m)")
    p.add_argument("--include-stablecoins", action="store_true",
                   help="Include stablecoins (default: False)")
    p.add_argument("--limit-symbols", type=int, default=None,
                   help="Limit number of mapped symbols to download (debugging)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    extra_quotes = [q.strip() for q in args.extra_quotes.split(",") if q.strip()]
    run(
        exchange=args.exchange,
        vs_currency=args.vs_currency,
        quote=args.quote.strip().upper(),
        extra_quotes=[q.upper() for q in extra_quotes],
        days=args.days,
        outdir=outdir,
        include_stablecoins=args.include_stablecoins,
        limit_symbols=args.limit_symbols,
    )
