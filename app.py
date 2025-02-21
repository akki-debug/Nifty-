import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Title and header
st.title("NIFTY 50 Stock Dashboard")
st.header("Real-time Stock Information")

# Create a selectbox for stock selection
nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "WIPRO.NS", "BAJFINANCE.NS",
    "M&M.NS", "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "BRITANNIA.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS", "VEDL.NS", "TATAMOTORS.NS",
    "MARUTI.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BOSCHLTD.NS", "MRF.NS",
    "NATIONALUM.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "AARTIIND.NS", "ABB.NS",
    "ABBOTINDIA.NS", "3MINDIA.NS", "AIAENG.NS", "APLLTD.NS", "BALKRISIND.NS",
    "BANDHANBNK.NS", "BATAINDIA.NS", "BERGEPAINT.NS", "BHARATFORG.NS"
]

# Create stock selector
selected_stock = st.selectbox("Select Stock", nifty50_stocks)

# Date range selector
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
date_range = st.slider("Date Range", 
                      min_value=start_date, 
                      max_value=end_date, 
                      value=[start_date, end_date])

# Fetch data
try:
    stock_data = yf.Ticker(selected_stock)
    hist = stock_data.history(start=date_range[0], end=date_range[1])
    
    # Display basic information
    st.subheader(f"Basic Information - {selected_stock}")
    info = stock_data.info
    st.write(info)
    
    # Display charts
    st.subheader("Price Charts")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price chart
    ax1.plot(hist.index, hist['Close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.grid(True)
    
    # Volume chart
    ax2.bar(hist.index, hist['Volume'], label='Volume')
    ax2.set_title('Trading Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # Display statistics
    st.subheader("Key Statistics")
    stats = {
        'Current Price': hist['Close'][-1],
        'Daily Return': hist['Close'].pct_change().dropna().mean() * 100,
        'Volatility': hist['Close'].pct_change().dropna().std() * np.sqrt(252) * 100,
        '52-Week High': hist['High'].max(),
        '52-Week Low': hist['Low'].min()
    }
    st.write(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
    
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
