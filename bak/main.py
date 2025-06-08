import os
import pandas as pd
import time
import yfinance as yf
import numpy as np
import requests
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# from notify import send_pushover_alert  # Disabled for debugging

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Load tickers
with open("tickers.txt") as f:
    TICKERS = [line.strip() for line in f if line.strip()]

# Initialize sentiment and embedding models
sentiment_pipeline = pipeline("sentiment-analysis")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helper Functions ----------
def get_price_data(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    df['Ticker'] = ticker
    df['Date'] = df.index
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
    df['Gap'] = ((df['Open'].values - df['Prev_Close'].values) / df['Prev_Close'].values) * 100
    return df

def add_return_labels(df, window=3):
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

    try:
        response = requests.get(url)
        articles = response.json().get('articles', [])
        headlines = [str(a['title']) for a in articles if a.get('title')]
    except Exception as e:
        print(f"API error for {ticker} on {query_date}: {e}")
        return 0, np.zeros(384)

    if not headlines:
        print(f"No headlines for {ticker} on {query_date}")
        return 0, np.zeros(384)

    try:
        embeddings = embedding_model.encode(headlines)
        embeddings = np.array(embeddings)

        print(f"✅ Raw embeddings shape: {embeddings.shape}")

        if embeddings.ndim != 2 or embeddings.shape[1] != 384:
            print(f"❌ Invalid embedding shape BEFORE MEAN: {embeddings.shape}")
            return 0, np.zeros(384)

        avg_embedding = embeddings.mean(axis=0)

    except Exception as e:
        print(f"❌ Embedding exception for {ticker} on {query_date}: {e}")
        return 0, np.zeros(384)

    try:
        sentiments = sentiment_pipeline(headlines)
        net_score = sum(r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in sentiments)
    except Exception as e:
        print(f"Sentiment error for {ticker} on {query_date}: {e}")
        net_score = 0

    return net_score, avg_embedding

# ---------- Main Pipeline ----------
def main():
    results = []

    for ticker in tqdm(TICKERS):
        try:
            df = get_price_data(ticker)
            df = add_indicators(df)
            df = add_premarket_gap(df)
            df = df.dropna()

            sentiment_scores = []
            embeddings = []

            for date in df['Date']:
                score, avg_embedding = get_news_sentiment_and_embedding(ticker, date)

                avg_embedding = np.array(avg_embedding).flatten()
                if avg_embedding.shape != (384,):
                    print(f"\u274c BAD embedding shape for {ticker} on {date}: {avg_embedding.shape}")
                    avg_embedding = np.zeros(384)

                sentiment_scores.append(score)
                embeddings.append(avg_embedding)
                time.sleep(1)

            # Clean and validate sentiment scores
            cleaned_scores = []
            for i, s in enumerate(sentiment_scores):
                if isinstance(s, (int, float)):
                    cleaned_scores.append(float(s))
                else:
                    print(f"\u274c Invalid sentiment score at index {i}: {type(s)}, replacing with 0.0")
                    cleaned_scores.append(0.0)

            df['Sentiment_Score'] = cleaned_scores

            for i, score in enumerate(cleaned_scores):
                if isinstance(score, (list, np.ndarray)) and np.array(score).ndim != 0:
                    print(f"\u274c STILL nested score at {i}: shape {np.array(score).shape}")

            df['Sentiment_Score_3d'] = pd.Series(cleaned_scores).rolling(3).mean().values
            df = add_return_labels(df)
            df = add_buy_signal(df)

            # Create embedding DataFrame safely
            cleaned_embeddings = []
            for idx, emb in enumerate(embeddings):
                emb = np.array(emb).flatten()
                if emb.shape != (384,):
                    print(f"\u274c Skipping invalid embedding at index {idx}, shape {emb.shape}")
                    emb = np.zeros(384)
                cleaned_embeddings.append(emb)

            embed_df = pd.DataFrame(cleaned_embeddings, columns=[f'Embed_{i}' for i in range(384)])
            df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)

            latest_buy = df[df['Buy_Signal'] == 1].tail(1)
            if not latest_buy.empty:
                results.append(latest_buy)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if results:
        alert_df = pd.concat(results)[['Ticker', 'Close', 'RSI', 'Gap', 'Sentiment_Score']]
        print("Buy signals found:\n", alert_df.to_string(index=False))

        summary_df = pd.concat(results)[['Date', 'Ticker', 'Close', 'RSI', 'Gap', 'Sentiment_Score', 'Return_3d']]
        summary_df.sort_values(by="Sentiment_Score", ascending=False, inplace=True)
        summary_df.to_csv("swing_signals.csv", index=False)
    else:
        print("No buy signals found.")

if __name__ == "__main__":
    main()