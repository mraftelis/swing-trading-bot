import os
import time
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from notify import send_pushover_alert

# Load tickers
with open("tickers.txt") as f:
    TICKERS = [line.strip() for line in f if line.strip()]

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_model = pipeline("sentiment-analysis")

def get_finviz_headlines(ticker):
    try:
        url = f"https://finviz.com/rss.ashx?t={ticker}"
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:5]]
        if not headlines:
            print(f"⚠️ No Finviz news for {ticker}, using fallback.")
            headlines = [
                f"{ticker} trading sideways",
                f"Investors cautious on {ticker}",
                f"{ticker} under market watch"
            ]
        return headlines
    except Exception as e:
        print(f"❌ Error fetching Finviz news for {ticker}: {e}")
        return []

def build_dataset_if_missing(ticker):
    path = f"{ticker.lower()}_dataset.csv"
    if os.path.exists(path):
        return

    try:
        print(f"📥 Building dataset for {ticker}")
        df = yf.download([ticker], period="3mo", interval="1d", auto_adjust=True)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.index.name = "Date"
        df = df.reset_index()

        df['50_MA'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df.to_csv(path, index=False)
        print(f"✅ Created {path}")
    except Exception as e:
        print(f"❌ Failed to build dataset for {ticker}: {e}")

def process_ticker(ticker):
    try:
        build_dataset_if_missing(ticker)
        df = pd.read_csv(f"{ticker.lower()}_dataset.csv")

        headlines = get_finviz_headlines(ticker)
        sentiments = sentiment_model(headlines)
        score = sum(r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in sentiments)
        embedding = np.mean(embedding_model.encode(headlines), axis=0)

        sentiment_scores = [score] * len(df)
        embeddings = [embedding] * len(df)

        df['Sentiment_Score'] = sentiment_scores
        embed_df = pd.DataFrame(embeddings, columns=[f"Embed_{i}" for i in range(384)])
        df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)

        df['Buy_Signal'] = (
            (df['RSI'] < 45) &
            (df['Close'] > df['50_MA']) &
            (df['Sentiment_Score'] > 0)
        ).astype(int)

        signal_df = df[df['Buy_Signal'] == 1]
        if not signal_df.empty:
            alert_df = signal_df[['Date', 'Close', 'RSI', 'Sentiment_Score']].tail(1)
            body = alert_df.to_string(index=False)
            send_pushover_alert(f"📈 Swing Alert for {ticker}", body)
        else:
            send_pushover_alert(f"📉 No Signal for {ticker}", "No swing trades triggered today.")

        df.to_csv(f"{ticker.lower()}_signals.csv", index=False)
        print(f"✅ Processed {ticker} → {ticker.lower()}_signals.csv")
    except Exception as e:
        print(f"❌ Failed processing {ticker}: {e}")

def main():
    for ticker in TICKERS:
        process_ticker(ticker)

if __name__ == "__main__":
    main()