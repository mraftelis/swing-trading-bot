"""
backtest.py — Swing Trading Backtest Engine
============================================

Tests the multi-indicator swing trading strategy across 12 symbols
using 2 years of daily OHLCV data (pure technical analysis, no sentiment).

Strategy summary
----------------
ENTRY  (signal fires at bar close, executes at next bar's open)
  Must-have:
    • EMA50 > EMA200          → macro uptrend
    • RSI(14) ∈ [32, 52]     → pullback zone in uptrend
    • ADX(14) > 20            → market is trending, not choppy
    • DI+ > DI-               → bullish directional bias
  Scored (+1 each, threshold ≥ 3/6):
    • EMA9 > EMA21            → short-term momentum aligned
    • MACD Histogram rising   → momentum turning up
    • MACD line > 0           → positive territory
    • Volume ≥ 90% of 20-day avg  → decent participation
    • BB%B < 0.65             → price not yet extended
    • OBV_Slope > 0           → institutional accumulation

EXIT (checked each bar in priority order)
  1. Stop Loss    — bar's Low  ≤ entry − ATR_STOP_MULT × ATR(at entry)
  2. Take Profit  — bar's High ≥ entry + ATR_TP_MULT  × ATR(at entry)
  3. Sell Signal  — RSI > 72 OR MACD cross-down OR Close < EMA50
  4. Time Stop    — MAX_HOLD_DAYS bars elapsed

POSITION SIZING
  risk_amount = capital × RISK_PER_TRADE (2 %)
  shares      = risk_amount / (entry − stop_loss)
  capped at 95 % of remaining capital

METRICS REPORTED
  Trades, Win%, Avg Win%, Avg Loss%, Profit Factor,
  Total Return%, Max Drawdown%, Avg Hold Days, Final Capital
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import add_indicators, add_signals

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS: List[str] = [
    "SPY",    # S&P 500 ETF
    "AAPL",   # Apple
    "GOOG",   # Alphabet (Class C)
    "AMZN",   # Amazon
    "C",      # Citigroup
    "T",      # AT&T
    "MSFT",   # Microsoft
    "NFLX",   # Netflix
    "SBUX",   # Starbucks
    "TSLA",   # Tesla
    "QQQ",    # Nasdaq-100 ETF
    "F",      # Ford Motor
    "INTC",   # Intel
    "AAL",    # American Airlines
    "NVDA",   # Nvidia
    "AMD",    # Advanced Micro Devices
    "LYFT",   # Lyft
    "GE",     # GE Aerospace
    "ARM",    # ARM Holdings
    "TMUS",   # T-Mobile US
]

PORTFOLIO_CAPITAL:  float = 120_000.0  # shared pool across all tickers
RISK_PER_TRADE:     float = 0.015       # risk 1.5% of *portfolio* per trade
ATR_STOP_MULT:      float = 2.0         # initial stop = entry − 2.0 × ATR
ATR_TP_MULT:        float = 4.0         # hard target  = entry + 4.0 × ATR (let winners run)
ATR_TRAIL_MULT:     float = 2.0         # Chandelier trailing stop = highest_close − 2.0 × ATR
ATR_BREAKEVEN_MULT: float = 1.0         # move stop to breakeven once trade gains this many ATRs
ATR_PCT_MAX:        float = 3.0         # skip if ATR > 3.0% of price (too volatile/wide stops)
MAX_HOLD_DAYS:      int   = 20          # time-based exit
MIN_HOLD_DAYS:      int   = 3           # min bars before sell-signal exit (avoids whipsaw)
DATA_PERIOD:        str   = "5y"        # 5 years of daily bars


# ─── Data Fetcher ─────────────────────────────────────────────────────────────

def _fetch(ticker: str) -> Optional[pd.DataFrame]:
    """Download, clean and enrich an OHLCV dataframe. Returns None on failure."""
    raw = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty or len(raw) < 220:
        print(f"  ⚠  {ticker}: insufficient data ({len(raw)} bars), skipping.")
        return None

    # Flatten MultiIndex columns (yfinance returns these for some calls)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw.reset_index()
    df.columns = [str(c) for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])

    df = add_indicators(df)
    df = add_signals(df)   # sentiment_score=0.0 by default (pure technicals)

    # Drop rows where core indicators haven't warmed up yet
    df = df.dropna(subset=["ATR", "RSI", "EMA200", "ADX"]).reset_index(drop=True)

    if len(df) < 50:
        print(f"  ⚠  {ticker}: too few clean rows ({len(df)}), skipping.")
        return None

    return df


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def _run_backtest(df: pd.DataFrame, ticker: str, portfolio_capital: float) -> Tuple[list, float]:
    """
    Simulate trades for a single ticker using shared portfolio capital.

    Key improvements vs. v1
    -----------------------
    • Chandelier trailing stop (highest_close − ATR_TRAIL_MULT × entry_ATR)
    • Breakeven protection   (stop rises to entry once gain ≥ ATR_BREAKEVEN_MULT × ATR)
    • ATR% filter           (skips entries where ATR > ATR_PCT_MAX% of price — too volatile)
    • Shared capital pool   (risk based on portfolio size, not per-ticker bucket)

    Returns
    -------
    trades  : list of trade-dicts
    capital : float — portfolio capital after all trades in this ticker
    """
    capital   = portfolio_capital
    trades    = []
    in_trade  = False

    # Active trade state
    entry_price  = 0.0
    entry_atr    = 0.0   # ATR locked in at entry (used for trailing calculations)
    initial_stop = 0.0
    stop         = 0.0
    target       = 0.0
    highest_close = 0.0
    shares       = 0
    entry_idx    = 0
    entry_date   = None

    rows = df.to_dict("records")

    for i in range(1, len(rows)):
        cur  = rows[i]
        prev = rows[i - 1]

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if not in_trade and int(prev.get("Buy_Signal", 0)) == 1:
            ep  = float(cur["Open"])
            atr = float(prev["ATR"])

            # Reject if price or ATR is invalid
            if not (ep > 0 and atr > 0):
                continue

            # ── ATR% volatility filter ────────────────────────────────────────
            # Skip trades where the daily range is too wide (hard to set tight stops)
            if (atr / ep * 100) > ATR_PCT_MAX:
                continue

            sl = ep - ATR_STOP_MULT * atr
            tp = ep + ATR_TP_MULT   * atr
            risk_per_share = ep - sl

            if risk_per_share <= 0:
                continue

            # Position size: risk RISK_PER_TRADE of total portfolio
            sh = int(
                min(
                    capital * RISK_PER_TRADE / risk_per_share,
                    capital * 0.20 / ep,   # single position cap: 20% of portfolio
                )
            )
            if sh < 1:
                continue

            in_trade     = True
            entry_price  = ep
            entry_atr    = atr
            initial_stop = sl
            stop         = sl
            target       = tp
            highest_close = ep
            shares       = sh
            entry_idx    = i
            entry_date   = cur["Date"]

        # ── EXIT ──────────────────────────────────────────────────────────────
        elif in_trade:
            xp     = None
            reason = None
            held   = i - entry_idx

            lo  = float(cur["Low"])
            hi  = float(cur["High"])
            op  = float(cur["Open"])
            cl  = float(cur["Close"])

            # ── Update Chandelier trailing stop (computed from PREVIOUS highest close)
            # This prevents same-bar look-ahead: we set the stop before seeing today's moves.
            trail_stop = highest_close - ATR_TRAIL_MULT * entry_atr

            # Breakeven protection: once trade is up ATR_BREAKEVEN_MULT × ATR, floor at entry
            if highest_close >= entry_price + ATR_BREAKEVEN_MULT * entry_atr:
                stop = max(initial_stop, entry_price, trail_stop)
            else:
                # Before breakeven: only trail upward, never below initial stop
                stop = max(initial_stop, trail_stop)

            # ── Check exits (priority order) ──────────────────────────────────
            if lo <= stop:
                xp, reason = stop, "Stop Loss"
            elif hi >= target:
                xp, reason = target, "Take Profit"
            elif int(prev.get("Sell_Signal", 0)) == 1 and held >= MIN_HOLD_DAYS:
                xp, reason = op, "Sell Signal"
            elif held >= MAX_HOLD_DAYS:
                xp, reason = cl, "Time Exit"

            if xp is not None:
                pnl     = (xp - entry_price) * shares
                pnl_pct = (xp - entry_price) / entry_price * 100
                capital += pnl

                trades.append(
                    {
                        "Ticker":      ticker,
                        "Entry_Date":  entry_date,
                        "Exit_Date":   cur["Date"],
                        "Days_Held":   held,
                        "Entry_$":     round(entry_price, 2),
                        "Exit_$":      round(xp, 2),
                        "Stop_$":      round(stop, 2),
                        "Target_$":    round(target, 2),
                        "Shares":      shares,
                        "PnL_$":       round(pnl, 2),
                        "PnL_%":       round(pnl_pct, 2),
                        "Exit_Reason": reason,
                        "Capital":     round(capital, 2),
                    }
                )
                in_trade = False
            else:
                # Update highest close for trailing stop on next bar
                highest_close = max(highest_close, cl)

    return trades, capital


# ─── Performance Metrics ──────────────────────────────────────────────────────

def _calc_metrics(trades: list, ticker: str, capital_before: float) -> dict:
    """Compute per-ticker performance statistics relative to portfolio capital at entry."""
    base = {
        "Ticker": ticker, "Trades": 0, "Win_%": 0.0,
        "Avg_Win_%": 0.0, "Avg_Loss_%": 0.0, "PF": 0.0,
        "Net_PnL_$": 0.0, "Avg_Hold": 0.0,
        "Exits": {},
    }
    if not trades:
        return base

    t      = pd.DataFrame(trades)
    wins   = t[t["PnL_$"] > 0]
    losses = t[t["PnL_$"] <= 0]

    gw = float(wins["PnL_$"].sum())          if len(wins)   else 0.0
    gl = float(losses["PnL_$"].abs().sum()) if len(losses) else 0.0

    return {
        "Ticker":     ticker,
        "Trades":     len(t),
        "Win_%":      round(len(wins) / len(t) * 100, 1),
        "Avg_Win_%":  round(float(wins["PnL_%"].mean())   if len(wins)   else 0.0, 2),
        "Avg_Loss_%": round(float(losses["PnL_%"].mean()) if len(losses) else 0.0, 2),
        "PF":         round(gw / max(gl, 1e-10), 2),
        "Net_PnL_$":  round(t["PnL_$"].sum(), 2),
        "Avg_Hold":   round(float(t["Days_Held"].mean()), 1),
        "Exits":      t["Exit_Reason"].value_counts().to_dict(),
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def _print_header():
    width = 74
    print("\n" + "=" * width)
    print("  SWING TRADING BACKTEST  ─  Multi-Indicator + Trailing Stop")
    print(f"  Run date    : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Period      : {DATA_PERIOD}  |  Portfolio   : ${PORTFOLIO_CAPITAL:,.0f}")
    print(f"  Stop        : {ATR_STOP_MULT}×ATR initial  |  Trail : {ATR_TRAIL_MULT}×ATR Chandelier")
    print(f"  Target      : {ATR_TP_MULT}×ATR hard cap  |  Breakeven at +{ATR_BREAKEVEN_MULT}×ATR")
    print(f"  Risk/trade  : {RISK_PER_TRADE*100:.1f}%  |  ATR% max : {ATR_PCT_MAX}%  |  Max hold : {MAX_HOLD_DAYS}d")
    print("=" * width)


def _print_summary(all_metrics: list, all_trades: list):
    width = 74
    print("\n" + "=" * width)
    print("  RESULTS SUMMARY")
    print("=" * width)

    if not all_metrics:
        print("  No results.\n")
        return

    mdf = pd.DataFrame(all_metrics)
    cols = ["Ticker", "Trades", "Win_%", "Avg_Win_%", "Avg_Loss_%", "PF", "Net_PnL_$", "Avg_Hold"]
    print(mdf[cols].to_string(index=False))

    # Portfolio-level aggregates
    total_trades = sum(m["Trades"] for m in all_metrics)
    wins_all     = sum(1 for t in all_trades if t["PnL_$"] > 0)
    wr_all       = wins_all / max(total_trades, 1) * 100
    gross_win    = sum(t["PnL_$"] for t in all_trades if t["PnL_$"] > 0)
    gross_loss   = abs(sum(t["PnL_$"] for t in all_trades if t["PnL_$"] <= 0))
    pf_all       = gross_win / max(gross_loss, 1e-10)
    net_pnl      = sum(m["Net_PnL_$"] for m in all_metrics)
    final_cap    = PORTFOLIO_CAPITAL + net_pnl
    total_ret    = net_pnl / PORTFOLIO_CAPITAL * 100

    all_exit_reasons: Dict[str, int] = {}
    for m in all_metrics:
        for reason, count in m.get("Exits", {}).items():
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count

    print("\n" + "-" * width)
    print(f"  PORTFOLIO AGGREGATE")
    print(f"  {'Starting Capital':<26} ${PORTFOLIO_CAPITAL:>11,.0f}")
    print(f"  {'Final Portfolio Value':<26} ${final_cap:>11,.2f}")
    print(f"  {'Total Return':<26} {total_ret:>+10.2f}%")
    print(f"  {'Total Trades':<26} {total_trades:>11d}")
    print(f"  {'Overall Win Rate':<26} {wr_all:>10.1f}%")
    print(f"  {'Profit Factor':<26} {pf_all:>11.2f}")

    if all_exit_reasons:
        print(f"\n  Exit Reason Breakdown:")
        for reason, count in sorted(all_exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / max(total_trades, 1) * 100
            print(f"    {reason:<18} {count:>4}  ({pct:.1f}%)")
    print("-" * width + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _print_header()

    all_trades:  List[Dict] = []
    all_metrics: List[Dict] = []
    portfolio_capital = PORTFOLIO_CAPITAL  # shared, flows across all tickers sequentially

    for ticker in TICKERS:
        print(f"\n  {ticker:<6} — downloading {DATA_PERIOD} of data…")

        try:
            df = _fetch(ticker)
            if df is None:
                continue

            cap_before = portfolio_capital
            trades, portfolio_capital = _run_backtest(df, ticker, portfolio_capital)
            m = _calc_metrics(trades, ticker, cap_before)

            all_trades.extend(trades)
            all_metrics.append(m)

            if m["Trades"] == 0:
                print(f"         ⚪ No trades generated.")
            else:
                pnl = m["Net_PnL_$"]
                icon = "✅" if pnl >= 0 else "🔴"
                print(
                    f"         {icon}  "
                    f"Trades:{m['Trades']:3d}  "
                    f"Win:{m['Win_%']:5.1f}%  "
                    f"Net PnL:${pnl:>+8,.0f}  "
                    f"PF:{m['PF']:.2f}  "
                    f"AvgHold:{m['Avg_Hold']:.1f}d"
                )

        except Exception as exc:
            print(f"         ❌ {exc}")
            import traceback; traceback.print_exc()

    _print_summary(all_metrics, all_trades)

    # ── CSV exports ───────────────────────────────────────────────────────────
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"  📄 Trade log saved  → backtest_trades.csv  ({len(trades_df)} trades)")

    if all_metrics:
        display_cols = ["Ticker", "Trades", "Win_%", "Avg_Win_%", "Avg_Loss_%", "PF", "Net_PnL_$", "Avg_Hold"]
        pd.DataFrame(all_metrics)[display_cols].to_csv("backtest_metrics.csv", index=False)
        print(f"  📄 Metrics saved    → backtest_metrics.csv")

    print()


if __name__ == "__main__":
    main()
