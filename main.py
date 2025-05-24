import os
import pandas as pd
import sqlite3
import time
from tqdm import tqdm
import yfinance as yf
import numpy as np
import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from notify import send_pushover_alert

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Load tickers from file
with open("tickers.txt") as f:
    TICKERS = [line.strip() for line in f if line.strip()]

# Sentiment and embedding models
sentiment_pipeline = pipeline("sentiment-analysis")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helper Functions ----------
def get_price_data(ticker):
    df = yf.download(ticker, period="3mo", interval="1d")
    df['Ticker'] = ticker
    return df

def add_indicators(df):
    df['50_MA'] = df['Close'].rolling(50).mean()
    df['200_MA'] = df['Close'].rolling(200).mean()
    df['MA_Gap'] = (df['50_MA'] - df['200_MA']) / df['200_MA'] * 100
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    return df

def add_premarket_gap(df):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close'] * 100
    return df

def add_return_labels(df, window=3, threshold=0.02):
    df['Future_Close'] = df['Close'].shift(-window)
    df['Return_3d'] = (df['Future_Close'] - df['Close']) / df['Close']
    df.drop(columns=['Future_Close'], inplace=True)
    return df

def add_buy_signal(df):
    df['Buy_Signal'] = (
        (df['Return_3d'] > 0.02) &
        (df['Sentiment_Score'] > 0) &
        (df['RSI'] < 45) &
        (df['Gap'] > 0)
    ).astype(int)
    return df

def get_news_sentiment_and_embedding(ticker, date):
    query_date = pd.to_datetime(date).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q={ticker}&from={query_date}&to={query_date}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    headlines = [a['title'] for a in articles][:5]
    if not headlines:
        return 0, np.zeros(384)

    sentiments = sentiment_pipeline(headlines)
    net_score = sum(r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in sentiments)
    embeddings = embedding_model.encode(headlines)
    avg_embedding = np.mean(embeddings, axis=0)
    return net_score, avg_embedding

# ---------- Main Script ----------
def main():
    conn = sqlite3.connect('swing_trading.db')
    results = []

    for ticker in tqdm(TICKERS):
        try:
            df = get_price_data(ticker)
            df = add_indicators(df)
            df = add_premarket_gap(df)
            df = df.dropna()

            sentiment_scores = []
            embeddings = []

            for date in df.index.strftime('%Y-%m-%d'):
                score, embed = get_news_sentiment_and_embedding(ticker, date)
                sentiment_scores.append(score)
                embeddings.append(embed)
                time.sleep(1)  # NewsAPI limit

            df['Sentiment_Score'] = sentiment_scores
            df['Sentiment_Score_3d'] = pd.Series(sentiment_scores).rolling(3).mean().values
            df = add_return_labels(df)
            df = add_buy_signal(df)

            embed_df = pd.DataFrame(embeddings, columns=[f'Embed_{i}' for i in range(384)])
            df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)

            df.to_sql("swing_data", conn, if_exists='append', index=False)

            latest_buy = df[df['Buy_Signal'] == 1].tail(1)
            if not latest_buy.empty:
                results.append(latest_buy)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    conn.close()

    if results:
        alert_df = pd.concat(results)[['Ticker', 'Close', 'RSI', 'Gap', 'Sentiment_Score']]
        body = alert_df.to_string(index=False)
        send_pushover_alert("📈 Daily Swing Trade Alert", body)
    else:
        send_pushover_alert("📉 No Buy Signals Today", "No swing trades triggered today.")

if __name__ == "__main__":
    main()