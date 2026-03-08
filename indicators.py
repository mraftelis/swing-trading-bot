"""
indicators.py — Shared technical indicator library for swing trading.

All indicators use Wilder's smoothing (EWM with com=period-1) where applicable,
matching the industry standard (RSI, ATR, ADX).

Functions
---------
add_indicators(df)  → adds all columns to the OHLCV dataframe
add_signals(df)     → appends Buy_Signal, Sell_Signal, Signal_Score columns
"""

import numpy as np
import pandas as pd


# ─── Core Indicator Functions ─────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's exponential smoothing (com = period - 1)."""
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()


def _adx(
    high: pd.Series, low: pd.Series, atr: pd.Series, period: int = 14
) -> tuple:
    """
    Returns (ADX, +DI, -DI) using Wilder's smoothing.

    +DM  = current_high − previous_high   (if > 0 and > |previous_low − current_low|)
    −DM  = previous_low − current_low     (if > 0 and > |current_high − previous_high|)
    """
    h_diff = high - high.shift(1)
    l_diff = low.shift(1) - low

    plus_dm = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
    minus_dm = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)

    smooth_plus = plus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    smooth_minus = minus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    safe_atr = atr.replace(0, 1e-10)
    di_plus = 100 * smooth_plus / safe_atr
    di_minus = 100 * smooth_minus / safe_atr

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    adx = dx.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    return adx, di_plus, di_minus


def _macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple:
    """Returns (MACD line, signal line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple:
    """Returns (upper band, lower band, middle band, %B, bandwidth)."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower + 1e-10)   # 0=lower, 1=upper
    bandwidth = (upper - lower) / mid.replace(0, 1e-10)
    return upper, lower, mid, pct_b, bandwidth


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


# ─── Master Indicator Builder ─────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV dataframe.

    Required columns: Date, Open, High, Low, Close, Volume
    All columns are added in-place and the dataframe is returned.
    """
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)

    # ── Exponential Moving Averages ───────────────────────────────────────────
    for span in [9, 21, 50, 200]:
        df[f"EMA{span}"] = c.ewm(span=span, adjust=False).mean()

    # ── Simple Moving Averages (50/200 for Golden-cross reference) ────────────
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    df["RSI"] = _rsi(c, 14)

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    df["MACD"], df["MACD_Sig"], df["MACD_Hist"] = _macd(c)

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────────
    df["BB_Upper"], df["BB_Lower"], df["BB_Mid"], df["BB_Pct"], df["BB_Width"] = (
        _bollinger(c, 20, 2.0)
    )

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    df["ATR"] = _atr(h, l, c, 14)

    # ── ADX + Directional Indicators (14) ────────────────────────────────────
    df["ADX"], df["DI_Plus"], df["DI_Minus"] = _adx(h, l, df["ATR"], 14)

    # ── Volume Analysis ───────────────────────────────────────────────────────
    df["Vol_MA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_MA20"].replace(0, 1e-10)

    # OBV normalised (z-score over 20 bars)
    obv_raw = _obv(c, v)
    df["OBV_Slope"] = obv_raw.diff(5)   # 5-bar OBV momentum

    # ── Premarket Gap ─────────────────────────────────────────────────────────
    df["Prev_Close"] = c.shift(1)
    df["Gap_Pct"] = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"].replace(0, 1e-10) * 100

    # ── ATR Stop / Target levels for the most recent close ───────────────────
    df["Suggested_Stop"] = (c - 2.0 * df["ATR"]).round(2)
    df["Suggested_Target"] = (c + 3.0 * df["ATR"]).round(2)

    return df


# ─── Signal Generator ─────────────────────────────────────────────────────────

def add_signals(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Generate buy and sell signals using a confluence-based scoring system.

    BUY logic (must-have AND scored conditions):
        Must-have  → EMA50 > EMA200  AND  RSI in [30, 55]  AND  ADX > 18
        Scored (+1 each, threshold ≥ 2 of 7):
            • EMA9 > EMA21            (short-term momentum aligned)
            • MACD histogram rising   (momentum turning up)
            • MACD line > 0           (positive momentum)
            • Volume ratio > 0.8      (decent participation)
            • %B < 0.70               (price not overextended)
            • OBV_Slope > 0           (institutional accumulation)
            • DI+ > DI-               (directional bias bullish)
            • Positive sentiment      (score > 0.5)
        Signal fires when must-have conditions met AND Score ≥ 2

    SELL logic — only clear momentum reversals:
        • RSI > 72                  (overbought extreme)
        • MACD bearish crossover    (MACD line crosses below signal line)
        • RSI > 68 AND BB%B > 0.92  (overbought with extended price)
        NOTE: "Close < EMA50" is intentionally excluded to prevent premature
              exits when buying into a pullback that touches EMA50 support.
              The ATR-based stop loss handles downside protection.
    """

    c = df["Close"].astype(float)

    # ──────────────────────────────────────────────────────────────────────────
    # BUY CONDITIONS
    # ──────────────────────────────────────────────────────────────────────────

    # Must-have (non-negotiable filters)
    uptrend      = df["EMA50"] > df["EMA200"]    # macro uptrend intact
    rsi_pull     = df["RSI"].between(30, 55)      # wider pullback zone
    trend_active = df["ADX"] > 18                 # market is trending (not sideways)

    must_have = uptrend & rsi_pull & trend_active

    # Scored conditions (+1 each)
    scored = pd.DataFrame(
        {
            "short_momentum": (df["EMA9"] > df["EMA21"]).astype(int),
            "macd_hist_up":   (df["MACD_Hist"] > df["MACD_Hist"].shift(1)).astype(int),
            "macd_positive":  (df["MACD"] > 0).astype(int),
            "volume_ok":      (df["Vol_Ratio"] > 0.8).astype(int),
            "not_extended":   (df["BB_Pct"] < 0.70).astype(int),
            "obv_rising":     (df["OBV_Slope"] > 0).astype(int),
            "di_bullish":     (df["DI_Plus"] > df["DI_Minus"]).astype(int),
            "sentiment":      int(sentiment_score > 0.5),
        }
    )
    df["Signal_Score"] = scored.sum(axis=1)

    df["Buy_Signal"] = (must_have & (df["Signal_Score"] >= 2)).astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # SELL CONDITIONS — momentum reversal only (NOT EMA50 break)
    # ATR stop loss handles the downside; sell signals capture topping action.
    # ──────────────────────────────────────────────────────────────────────────
    rsi_ob = df["RSI"] > 72
    macd_cross_down = (df["MACD"] < df["MACD_Sig"]) & (
        df["MACD"].shift(1) >= df["MACD_Sig"].shift(1)
    )
    overbought_extended = (df["RSI"] > 68) & (df["BB_Pct"] > 0.92)

    df["Sell_Signal"] = (rsi_ob | macd_cross_down | overbought_extended).astype(int)

    return df
