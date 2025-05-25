import streamlit as st
import pandas as pd

st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")
st.title("📈 Daily Swing Trade Dashboard")
st.markdown("Filter swing trading signals based on RSI, sentiment, and gap")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("swing_signals.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("🔎 Filter Options")

# Date selector
available_dates = sorted(df["Date"].dt.date.unique(), reverse=True)
selected_date = st.sidebar.selectbox("Select Date", available_dates)

# RSI range
rsi_range = st.sidebar.slider("RSI Range", 0, 100, (0, 45))

# Minimum sentiment score
min_sentiment = st.sidebar.slider("Minimum Sentiment Score", -1.0, 1.0, 0.0)

# Ticker search
ticker_search = st.sidebar.text_input("Search Ticker", "")

# ---------- Apply Filters ----------
filtered = df.copy()
filtered = filtered[df["Date"].dt.date == selected_date]
filtered = filtered[(filtered["RSI"] >= rsi_range[0]) & (filtered["RSI"] <= rsi_range[1])]
filtered = filtered[filtered["Sentiment_Score"] >= min_sentiment]

if ticker_search:
    filtered = filtered[filtered["Ticker"].str.contains(ticker_search.upper(), na=False)]

# ---------- Display ----------
st.subheader(f"📊 Signals for {selected_date}")
if not filtered.empty:
    st.dataframe(
        filtered[["Ticker", "Close", "RSI", "Gap", "Sentiment_Score", "Return_3d"]]
        .sort_values(by="Sentiment_Score", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
else:
    st.info("No signals found for the selected filters and date.")

# ---------- Full Data Toggle ----------
with st.expander("🔍 Preview All Signals"):
    st.dataframe(df.sort_values(by="Date", ascending=False).reset_index(drop=True).head(100))