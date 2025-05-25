import os
import glob
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Swing Trade Dashboard")
st.title("📈 Swing Trade Signal Dashboard")

# Find all *_signals.csv files
signal_files = sorted(glob.glob("*_signals.csv"))
if not signal_files:
    st.warning("No signal files found. Run the pipeline first.")
    st.stop()

# Load all signal files
all_signals = []
for file in signal_files:
    try:
        df = pd.read_csv(file)
        ticker = file.replace("_signals.csv", "").upper()
        df['Ticker'] = ticker
        all_signals.append(df)
    except Exception as e:
        st.error(f"Failed to load {file}: {e}")

signals_df = pd.concat(all_signals, ignore_index=True)
signals_df['Date'] = pd.to_datetime(signals_df['Date'])
signals_df.sort_values(by="Date", ascending=False, inplace=True)

# Sidebar filter
st.sidebar.header("🔍 Filter")
tickers = sorted(signals_df['Ticker'].unique())
selected = st.sidebar.multiselect("Select Tickers", tickers, default=tickers)
filtered = signals_df[signals_df['Ticker'].isin(selected)]

# Show recent buy signals
st.markdown("### 🧾 Recent Buy Signals")
buy_df = filtered[filtered['Buy_Signal'] == 1].copy()
if not buy_df.empty:
    st.dataframe(buy_df[['Date', 'Ticker', 'Close', 'RSI', 'Sentiment_Score']], use_container_width=True)
else:
    st.info("No buy signals in the selected tickers.")

st.markdown("---")

# Charts per ticker
for ticker in selected:
    df = filtered[filtered['Ticker'] == ticker].sort_values(by="Date")
    if df.empty:
        continue

    st.subheader(f"📊 {ticker} Signals")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(df, x="Date", y=["Close", "50_MA"], title="Price & 50-Day MA")
        st.plotly_chart(fig1, use_container_width=True, key=f"{ticker}_chart_price_ma")

    with col2:
        fig2 = px.line(df, x="Date", y="RSI", title="RSI")
        st.plotly_chart(fig2, use_container_width=True, key=f"{ticker}_chart_rsi")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.line(df, x="Date", y="Sentiment_Score", title="Sentiment Score")
        st.plotly_chart(fig3, use_container_width=True, key=f"{ticker}_chart_sentiment")

    with col4:
        fig4 = px.scatter(
            df[df['Buy_Signal'] == 1],
            x="Date", y="Close",
            color_discrete_sequence=["green"],
            title="Buy Signals"
        )
        st.plotly_chart(fig4, use_container_width=True, key=f"{ticker}_chart_buy_signals")