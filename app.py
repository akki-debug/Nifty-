import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Page Config
st.set_page_config(page_title="NIFTY 50 Stock Dashboard", page_icon="📈", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Custom styles for a cleaner look */
        .stApp {
            background-color: #f4f4f4;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #1f77b4;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.5rem;
            color: #ff5733;
            font-weight: bold;
        }
        .stat-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-title">📈 NIFTY 50 Stock Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time Stock Market Analysis</p>', unsafe_allow_html=True)

# Sidebar for selections
st.sidebar.header("⚙️ Settings")

# Stock Selection
nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]
selected_stock = st.sidebar.selectbox("🔍 Select a Stock", nifty50_stocks)

# Date Selection
start_date = st.sidebar.date_input("📅 Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 End Date", value=pd.to_datetime("2023-12-31"))

# Convert dates
start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Fetch stock data
try:
    stock_data = yf.Ticker(selected_stock)
    hist = stock_data.history(start=start_date_str, end=end_date_str)

    # Display stock info
    st.markdown(f"### 📊 Basic Information - {selected_stock}")
    with st.expander("🔍 View Stock Details"):
        st.write(stock_data.info)

    # Price & Volume Charts
    st.markdown("### 📈 Stock Performance")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price Chart
    sns.lineplot(x=hist.index, y=hist["Close"], ax=ax1, color="blue", linewidth=2.5)
    ax1.set_title("Stock Closing Price", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (₹)")
    ax1.grid(True)

    # Volume Chart
    sns.barplot(x=hist.index, y=hist["Volume"], ax=ax2, color="orange", alpha=0.7)
    ax2.set_title("Trading Volume", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    st.pyplot(fig)

    # Stock Statistics
    st.markdown("### 📌 Key Statistics")
    stats = {
        "📍 Current Price": round(hist["Close"][-1], 2),
        "📊 Daily Return (%)": round(hist["Close"].pct_change().dropna().mean() * 100, 2),
        "📉 Volatility (%)": round(hist["Close"].pct_change().dropna().std() * np.sqrt(252) * 100, 2),
        "📈 52-Week High": round(hist["High"].max(), 2),
        "📉 52-Week Low": round(hist["Low"].min(), 2),
    }

    # Display in columns
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    for col, (stat, value) in zip([col1, col2, col3, col4, col5], stats.items()):
        col.markdown(f'<div class="stat-card"><b>{stat}</b>: {value}</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"🚨 Error fetching data: {str(e)}")
