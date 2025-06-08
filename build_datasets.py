import yfinance as yf
import pandas as pd

ticker = "AAPL"
df = yf.download([ticker], period="3mo", interval="1d", auto_adjust=True)

# Flatten MultiIndex columns
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
df.index.name = "Date"
df = df.reset_index()

# Print for sanity check
print("Final columns:", df.columns.tolist())

# 50-day Moving Average
df['50_MA'] = df['Close'].rolling(50).mean()

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Save to CSV
df.to_csv("aapl_dataset.csv", index=False)
print("✅ Saved clean aapl_dataset.csv")