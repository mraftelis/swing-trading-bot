"""
main_pipeline.py — Live Swing Trading Signal Pipeline
======================================================

For each ticker in tickers.txt:
  1. Downloads / refreshes OHLCV data (6 months, daily)
  2. Computes a full technical indicator suite (EMA, RSI, MACD, BB, ATR, ADX, Volume)
  3. Fetches Finviz headlines and scores sentiment
  4. Generates a confluence-based Buy / Sell signal with a numeric Signal_Score
  5. Saves *_signals.csv and sends a Pushover notification on active signals

Signal logic (see indicators.py for full detail):
  BUY   → EMA50>EMA200 AND RSI∈[32,52] AND ADX>20 AND DI+>DI− AND score≥3/7
  SELL  → RSI>72 OR MACD cross-down OR Close<EMA50
  Exit  → ATR-based stop (2×ATR) and target (3×ATR) are pre-computed
"""

import os
import time
import traceback

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from indicators import add_indicators, add_signals
from notify import send_pushover_alert

# Analyst recommendation mapping (yfinance scale: 1=Strong Buy, 5=Strong Sell)
_ANALYST_LABEL = {
    1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Underperform", 5: "Sell"
}

# ─── Config ──────────────────────────────────────────────────────────────────
with open("tickers.txt") as f:
    TICKERS = [line.strip() for line in f if line.strip()]

DATA_PERIOD = "6mo"   # 6 months gives solid EMA50 context and enough signal history

# ─── NLP Models (loaded once at startup) ─────────────────────────────────────
print("Loading NLP models…")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_sentiment_model = hf_pipeline("sentiment-analysis")
print("Models ready.\n")


# ─── Analyst Data ────────────────────────────────────────────────────────────

def get_analyst_data(ticker: str) -> dict:
    """
    Fetch analyst consensus and recent news from yfinance.

    Returns a dict with:
      analyst_score  : float  (-1 to +1, positive = bullish consensus)
      rec_mean       : float  (1=Strong Buy … 5=Sell; None if unavailable)
      rec_label      : str    (human-readable recommendation)
      num_analysts   : int
      price_target   : float  (consensus 12-month price target; None if unavailable)
      recent_upgrades: int    (upgrades in last 30 days)
      recent_downgrades: int
      news_headlines : list[str]   (up to 5 recent headlines from yfinance)
    """
    result = {
        "analyst_score":    0.0,
        "rec_mean":         None,
        "rec_label":        "N/A",
        "num_analysts":     0,
        "price_target":     None,
        "recent_upgrades":  0,
        "recent_downgrades": 0,
        "news_headlines":   [],
    }
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        # Consensus recommendation (1 = Strong Buy, 5 = Sell)
        rec_mean = info.get("recommendationMean")
        if rec_mean is not None:
            result["rec_mean"]     = round(rec_mean, 2)
            result["num_analysts"] = info.get("numberOfAnalystOpinions", 0)
            # Map 1–5 scale to -1 to +1 (linear inversion)
            result["analyst_score"] = round((3 - rec_mean) / 2, 3)   # 1→+1, 3→0, 5→-1
            result["rec_label"]    = _ANALYST_LABEL.get(round(rec_mean), "Hold")

        # Consensus price target
        tgt = info.get("targetMeanPrice")
        if tgt and info.get("currentPrice"):
            result["price_target"] = round(tgt, 2)

        # Recent upgrades / downgrades (last 30 days)
        try:
            upgrades_df = t.upgrades_downgrades
            if upgrades_df is not None and not upgrades_df.empty:
                upgrades_df.index = pd.to_datetime(upgrades_df.index, utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
                recent = upgrades_df[upgrades_df.index >= cutoff]
                if "Action" in recent.columns:
                    result["recent_upgrades"]   = int((recent["Action"].str.lower() == "up").sum())
                    result["recent_downgrades"] = int((recent["Action"].str.lower() == "down").sum())
        except Exception:
            pass

        # Recent news headlines from yfinance
        try:
            news = t.news or []
            result["news_headlines"] = [
                n.get("content", {}).get("title", "") or n.get("title", "")
                for n in news[:5]
                if n.get("content", {}).get("title") or n.get("title")
            ]
        except Exception:
            pass

    except Exception as exc:
        print(f"  ⚠  Analyst data error for {ticker}: {exc}")

    return result


# ─── Sentiment Helper ─────────────────────────────────────────────────────────

def get_sentiment(ticker: str) -> tuple:
    """
    Fetch up to 10 headlines from Finviz RSS, compute:
      - net_score  : float  (positive = bullish, negative = bearish)
      - embedding  : np.ndarray shape (384,)
    Returns (0.0, zeros) on error.
    """
    fallback = [
        f"{ticker} trading sideways",
        f"Investors cautious on {ticker}",
        f"{ticker} under market watch",
    ]
    try:
        feed = feedparser.parse(f"https://finviz.com/rss.ashx?t={ticker}")
        headlines = [e.title for e in feed.entries[:10]]
    except Exception as exc:
        print(f"  ⚠  RSS error for {ticker}: {exc}")
        headlines = fallback

    if not headlines:
        headlines = fallback

    # Sentiment score
    try:
        results = _sentiment_model(headlines[:5])
        score = sum(
            r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results
        )
    except Exception:
        score = 0.0

    # Embedding
    try:
        embedding = np.mean(_embed_model.encode(headlines[:5]), axis=0)
        if embedding.shape != (384,):
            embedding = np.zeros(384)
    except Exception:
        embedding = np.zeros(384)

    return float(score), embedding


# ─── Dataset Builder ─────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    """Download adjusted OHLCV data and return a clean DataFrame."""
    raw = yf.download(
        [ticker], period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw.index.name = "Date"
    df = raw.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_or_refresh_dataset(ticker: str) -> pd.DataFrame:
    """
    Returns a fresh DataFrame with indicators computed.
    Re-downloads if the cached CSV is missing or older than 1 day.
    """
    path = f"{ticker.lower()}_dataset.csv"
    stale = True

    if os.path.exists(path):
        age_seconds = time.time() - os.path.getmtime(path)
        stale = age_seconds > 86_400  # refresh after 24 hours

    if stale:
        print(f"  📥 Fetching {ticker} ({DATA_PERIOD})…")
        df = fetch_ohlcv(ticker)
        df = add_indicators(df)
        df.to_csv(path, index=False)
        print(f"  ✅ Saved {path}")
    else:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        # Re-compute indicators in case the schema changed
        df = add_indicators(df)

    return df


# ─── Pipeline Core ────────────────────────────────────────────────────────────

def process_ticker(ticker: str):
    try:
        # 1. OHLCV + indicators
        df = build_or_refresh_dataset(ticker)

        # 2. Sentiment (Finviz RSS)
        sentiment_score, embedding = get_sentiment(ticker)
        df["Sentiment_Score"] = sentiment_score

        # Append embedding columns
        embed_df = pd.DataFrame(
            [embedding] * len(df),
            columns=[f"Embed_{i}" for i in range(len(embedding))],
        )
        df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)

        # 3. Analyst consensus (yfinance)
        analyst = get_analyst_data(ticker)
        df["Analyst_Score"]      = analyst["analyst_score"]
        df["Analyst_Rec_Mean"]   = analyst["rec_mean"]
        df["Price_Target"]       = analyst["price_target"]
        df["Recent_Upgrades"]    = analyst["recent_upgrades"]
        df["Recent_Downgrades"]  = analyst["recent_downgrades"]

        # Combined sentiment = weighted average of news + analyst
        combined_sentiment = (
            0.6 * sentiment_score
            + 0.4 * analyst["analyst_score"] * 5   # scale analyst to same range as news
        )

        # 4. Signals — use combined sentiment
        df = add_signals(df, combined_sentiment)

        # 5. Analyst veto: if strong sell consensus (rec_mean > 3.8), suppress buy signals
        if analyst["rec_mean"] is not None and analyst["rec_mean"] > 3.8:
            df["Buy_Signal"] = 0
            print(f"  ⚠  Analyst consensus is {analyst['rec_label']} ({analyst['rec_mean']}) — buy signals suppressed.")

        # 6. Persist
        out_path = f"{ticker.lower()}_signals.csv"
        df.to_csv(out_path, index=False)
        print(f"  💾 {out_path} saved.")

        # 7. Alert on the latest bar's signal
        latest    = df.iloc[-1]
        last_date = pd.Timestamp(latest["Date"]).date()

        # Show analyst info in console regardless of signal
        print(
            f"  📊 Analyst: {analyst['rec_label']} (mean={analyst['rec_mean']}, "
            f"n={analyst['num_analysts']}, target=${analyst['price_target']})"
        )
        if analyst["recent_upgrades"] or analyst["recent_downgrades"]:
            print(f"     Last 30d: {analyst['recent_upgrades']} upgrades, {analyst['recent_downgrades']} downgrades")

        if int(latest["Buy_Signal"]) == 1:
            score = int(latest.get("Signal_Score", 0))
            # Price-to-target upside
            cur_price = float(latest["Close"])
            upside_str = ""
            if analyst["price_target"] and cur_price > 0:
                upside = (analyst["price_target"] - cur_price) / cur_price * 100
                upside_str = f"\nPT Upside: {upside:+.1f}%"

            body = (
                f"Date:      {last_date}\n"
                f"Close:     ${cur_price:.2f}\n"
                f"RSI:       {float(latest['RSI']):.1f}\n"
                f"ADX:       {float(latest['ADX']):.1f}\n"
                f"MACD:      {float(latest['MACD']):.3f}\n"
                f"Score:     {score}/8\n"
                f"Sentiment: {sentiment_score:+.2f}\n"
                f"Analyst:   {analyst['rec_label']} ({analyst['rec_mean']}){upside_str}\n"
                f"Stop:      ${float(latest['Suggested_Stop']):.2f}\n"
                f"Target:    ${float(latest['Suggested_Target']):.2f}"
            )
            send_pushover_alert(f"📈 BUY  {ticker}", body)
            print(f"  🟢 BUY signal on {ticker}  (score={score}/8)")

        elif int(latest["Sell_Signal"]) == 1:
            body = (
                f"Date:     {last_date}\n"
                f"Close:    ${float(latest['Close']):.2f}\n"
                f"RSI:      {float(latest['RSI']):.1f}\n"
                f"Analyst:  {analyst['rec_label']}"
            )
            send_pushover_alert(f"📉 SELL {ticker}", body)
            print(f"  🔴 SELL signal on {ticker}")

        else:
            print(f"  ⚪ No signal on {ticker}.")

    except Exception:
        print(f"  ❌ Error processing {ticker}:")
        traceback.print_exc()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SWING TRADING PIPELINE")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print("=" * 60)

    for ticker in TICKERS:
        print(f"\n[{ticker}]")
        process_ticker(ticker)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
