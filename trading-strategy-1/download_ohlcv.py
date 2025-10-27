import yfinance as yf
import pandas as pd

sym = "SOL-USD"                 
interval = "5m"                 
period = "60d"                  

df = yf.download(sym, interval=interval, period=period, auto_adjust=False, progress=False)

# Lowercase OHLCV column names
df = df.rename(columns=str.lower).reset_index()

# Rename the datetime index column to 'bar'
if "Datetime" in df.columns:      # intraday
    df = df.rename(columns={"Datetime": "bar"})
elif "Date" in df.columns:        # daily
    df = df.rename(columns={"Date": "bar"})
else:
    # Some environments name it 'index' after reset_index()
    df = df.rename(columns={"index": "bar"})

# Keep canonical columns and ensure UTC
df["bar"] = pd.to_datetime(df["bar"], utc=True)
df = df[["bar", "open", "high", "low", "close", "volume"]].sort_values("bar")

df.to_csv("SOLUSD_5m_yahoo.csv", index=False)
print(df.head())
print(df.tail())