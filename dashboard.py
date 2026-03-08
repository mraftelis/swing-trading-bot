"""
dashboard.py — Swing Trade Signal Dashboard
============================================
Visualises live signal files (*_signals.csv) and optional backtest results
(backtest_trades.csv, backtest_metrics.csv).

Run with:  streamlit run dashboard.py
"""

import glob
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Swing Trade Dashboard")
st.title("📈 Swing Trade Signal Dashboard")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter")
view = st.sidebar.radio("View", ["Live Signals", "Backtest Results"], index=0)

# ─── Live Signals View ───────────────────────────────────────────────────────
if view == "Live Signals":
    signal_files = sorted(glob.glob("*_signals.csv"))
    if not signal_files:
        st.warning("No signal files found. Run `python main_pipeline.py` first.")
        st.stop()

    all_signals = []
    for fpath in signal_files:
        try:
            df = pd.read_csv(fpath)
            ticker = os.path.basename(fpath).replace("_signals.csv", "").upper()
            df["Ticker"] = ticker
            all_signals.append(df)
        except Exception as exc:
            st.error(f"Failed to load {fpath}: {exc}")

    if not all_signals:
        st.error("Could not load any signal files.")
        st.stop()

    signals_df = pd.concat(all_signals, ignore_index=True)
    signals_df["Date"] = pd.to_datetime(signals_df["Date"])
    signals_df.sort_values("Date", ascending=False, inplace=True)

    tickers  = sorted(signals_df["Ticker"].unique())
    selected = st.sidebar.multiselect("Select Tickers", tickers, default=tickers)
    filtered = signals_df[signals_df["Ticker"].isin(selected)]

    # ── Latest Signal Status ──────────────────────────────────────────────────
    st.markdown("### 📡 Latest Signal per Ticker")
    latest_rows = []
    for t in selected:
        sub = filtered[filtered["Ticker"] == t]
        if sub.empty:
            continue
        row = sub.iloc[0]
        buy  = int(row.get("Buy_Signal", 0))
        sell = int(row.get("Sell_Signal", 0))
        status = "🟢 BUY" if buy else ("🔴 SELL" if sell else "⚪ NEUTRAL")
        latest_rows.append(
            {
                "Ticker":   t,
                "Date":     row["Date"].date(),
                "Close":    round(float(row["Close"]), 2),
                "RSI":      round(float(row.get("RSI", 0)), 1),
                "ADX":      round(float(row.get("ADX", 0)), 1),
                "MACD":     round(float(row.get("MACD", 0)), 3),
                "Score":    row.get("Signal_Score", "-"),
                "Signal":   status,
                "Stop":     row.get("Suggested_Stop", "-"),
                "Target":   row.get("Suggested_Target", "-"),
            }
        )
    if latest_rows:
        st.dataframe(pd.DataFrame(latest_rows), use_container_width=True)

    # ── Historical Buy Signals Table ──────────────────────────────────────────
    st.markdown("### 🧾 Historical Buy Signals")
    buy_df = filtered[filtered.get("Buy_Signal", 0) == 1].copy() if "Buy_Signal" in filtered else pd.DataFrame()
    try:
        buy_df = filtered[filtered["Buy_Signal"] == 1].copy()
    except KeyError:
        buy_df = pd.DataFrame()

    if not buy_df.empty:
        show_cols = [c for c in ["Date", "Ticker", "Close", "RSI", "ADX",
                                  "Signal_Score", "Sentiment_Score"] if c in buy_df.columns]
        st.dataframe(buy_df[show_cols].head(50), use_container_width=True)
    else:
        st.info("No buy signals found in selected tickers.")

    st.markdown("---")

    # ── Per-Ticker Charts ─────────────────────────────────────────────────────
    for ticker in selected:
        df = filtered[filtered["Ticker"] == ticker].sort_values("Date")
        if df.empty:
            continue

        st.subheader(f"📊 {ticker}")
        col1, col2 = st.columns(2)

        # Price + EMAs + buy/sell markers
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="blue")))
            if "EMA50" in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50", line=dict(color="orange", dash="dash")))
            if "EMA200" in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA200"], name="EMA200", line=dict(color="red", dash="dot")))

            # Buy markers
            if "Buy_Signal" in df.columns:
                buys = df[df["Buy_Signal"] == 1]
                fig.add_trace(go.Scatter(
                    x=buys["Date"], y=buys["Close"],
                    mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"),
                    name="Buy",
                ))
            # Sell markers
            if "Sell_Signal" in df.columns:
                sells = df[df["Sell_Signal"] == 1]
                fig.add_trace(go.Scatter(
                    x=sells["Date"], y=sells["Close"],
                    mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
                    name="Sell",
                ))

            fig.update_layout(title="Price, EMA50/200 & Signals", xaxis_title="", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True, key=f"{ticker}_price")

        # RSI
        with col2:
            if "RSI" in df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")))
                fig2.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought 70")
                fig2.add_hline(y=55, line_dash="dot",  line_color="orange", annotation_text="Upper Zone 55")
                fig2.add_hline(y=30, line_dash="dash", line_color="green",  annotation_text="Oversold 30")
                fig2.update_layout(title="RSI (14)", yaxis_range=[0, 100])
                st.plotly_chart(fig2, use_container_width=True, key=f"{ticker}_rsi")

        col3, col4 = st.columns(2)

        # MACD
        with col3:
            if all(c in df.columns for c in ["MACD", "MACD_Sig", "MACD_Hist"]):
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],     name="MACD",   line=dict(color="blue")))
                fig3.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Sig"], name="Signal", line=dict(color="orange")))
                colors = ["green" if v >= 0 else "red" for v in df["MACD_Hist"]]
                fig3.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color=colors))
                fig3.update_layout(title="MACD (12/26/9)")
                st.plotly_chart(fig3, use_container_width=True, key=f"{ticker}_macd")

        # ADX
        with col4:
            if "ADX" in df.columns:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df["Date"], y=df["ADX"],      name="ADX",  line=dict(color="black")))
                if "DI_Plus" in df.columns:
                    fig4.add_trace(go.Scatter(x=df["Date"], y=df["DI_Plus"],  name="+DI",  line=dict(color="green", dash="dash")))
                if "DI_Minus" in df.columns:
                    fig4.add_trace(go.Scatter(x=df["Date"], y=df["DI_Minus"], name="-DI",  line=dict(color="red",   dash="dash")))
                fig4.add_hline(y=20, line_dash="dot", line_color="gray", annotation_text="ADX 20 (trend threshold)")
                fig4.update_layout(title="ADX (14) + Directional Indicators")
                st.plotly_chart(fig4, use_container_width=True, key=f"{ticker}_adx")

        # Signal Score
        if "Signal_Score" in df.columns:
            fig5 = px.bar(df, x="Date", y="Signal_Score", color="Signal_Score",
                          color_continuous_scale="RdYlGn", range_color=[0, 8],
                          title="Signal Score (0–8, green=strong)")
            st.plotly_chart(fig5, use_container_width=True, key=f"{ticker}_score")

        st.markdown("---")

# ─── Backtest Results View ────────────────────────────────────────────────────
else:
    metrics_file = "backtest_metrics.csv"
    trades_file  = "backtest_trades.csv"

    if not os.path.exists(metrics_file):
        st.warning("No backtest results found. Run `python backtest.py` first.")
        st.stop()

    m_df = pd.read_csv(metrics_file)
    t_df = pd.read_csv(trades_file) if os.path.exists(trades_file) else pd.DataFrame()

    st.markdown("### 📊 Per-Ticker Performance")
    st.dataframe(m_df, use_container_width=True)

    # Summary bar chart
    if not m_df.empty:
        fig = px.bar(
            m_df, x="Ticker", y="Return_%",
            color="Return_%", color_continuous_scale="RdYlGn",
            text="Return_%", title="Return % per Ticker",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True, key="bt_returns")

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.bar(m_df, x="Ticker", y="Win_%",  color="Win_%",
                          color_continuous_scale="Blues", title="Win Rate %")
            st.plotly_chart(fig2, use_container_width=True, key="bt_winrate")
        with col2:
            fig3 = px.bar(m_df, x="Ticker", y="Max_DD_%", color="Max_DD_%",
                          color_continuous_scale="Reds_r", title="Max Drawdown %")
            st.plotly_chart(fig3, use_container_width=True, key="bt_dd")

    # Trade log
    if not t_df.empty:
        st.markdown("### 🗂️ Trade Log")
        tickers = st.sidebar.multiselect("Filter Tickers", sorted(t_df["Ticker"].unique()),
                                          default=sorted(t_df["Ticker"].unique()))
        t_display = t_df[t_df["Ticker"].isin(tickers)]
        st.dataframe(t_display, use_container_width=True)

        # P&L distribution
        fig4 = px.histogram(t_display, x="PnL_%", color="Exit_Reason",
                             nbins=30, title="P&L % Distribution by Exit Reason",
                             barmode="overlay", opacity=0.7)
        fig4.add_vline(x=0, line_color="black", line_dash="dash")
        st.plotly_chart(fig4, use_container_width=True, key="bt_pnl_dist")

        # Equity curves
        st.markdown("### 📈 Equity Curves")
        for ticker in tickers:
            td = t_df[t_df["Ticker"] == ticker].copy()
            if td.empty:
                continue
            td["Exit_Date"] = pd.to_datetime(td["Exit_Date"])
            td = td.sort_values("Exit_Date")
            td["Cumulative_PnL"] = td["PnL_$"].cumsum()
            fig5 = px.line(td, x="Exit_Date", y="Capital",
                           title=f"{ticker} — Capital Over Time",
                           markers=True)
            st.plotly_chart(fig5, use_container_width=True, key=f"bt_equity_{ticker}")
