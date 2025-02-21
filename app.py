import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

# Page Configuration
st.set_page_config(page_title="Black-Litterman Portfolio Optimizer", page_icon="📈", layout="wide")

# Title
st.title("📊 Black-Litterman Portfolio Optimizer")
st.write("Optimize your portfolio using the Black-Litterman model.")

# Sidebar Inputs
st.sidebar.header("⚙️ Model Inputs")

# NIFTY 50 Stock List
nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]

# Portfolio Stock Selection
selected_assets = st.sidebar.multiselect("📌 Select Stocks for Portfolio", nifty50_stocks, nifty50_stocks[:5])

# Market Expected Return & Risk-Free Rate
market_return = st.sidebar.slider("📈 Expected Market Return (%)", 5, 20, 10) / 100
risk_free_rate = st.sidebar.slider("🏦 Risk-Free Rate (%)", 2, 10, 5) / 100
confidence = st.sidebar.slider("💡 Investor Confidence in Views (%)", 0, 100, 50) / 100

# User Views on Expected Returns
st.sidebar.subheader("🔮 Your Views on Returns (%)")
user_views = {}
for stock in selected_assets:
    view = st.sidebar.number_input(f"{stock}", -10.0, 20.0, 5.0) / 100
    user_views[stock] = view

# Date Range
start_date = st.sidebar.date_input("📅 Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 End Date", value=pd.to_datetime("2023-12-31"))
start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Compute Black-Litterman Model
if st.sidebar.button("🔍 Optimize Portfolio"):
    try:
        # Fetch historical price data
        prices = yf.download(selected_assets, start=start_date_str, end=end_date_str)["Adj Close"]

        # Compute returns & covariance
        returns = prices.pct_change().dropna()
        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(returns).ledoit_wolf()

        # Market Caps (Approximated)
        market_caps = {stock: prices[stock].iloc[-1] * 1e6 for stock in selected_assets}

        # Convert user views into expected returns
        Q = np.array([user_views[stock] for stock in selected_assets])
        P = np.identity(len(selected_assets))

        # Black-Litterman Model
        bl = BlackLittermanModel(S, pi=mu, market_caps=market_caps, absolute_views=Q, P=P, tau=confidence)
        bl_mu = bl.bl_returns  # Adjusted expected returns

        # Optimize Portfolio
        ef = EfficientFrontier(bl_mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()

        # Display Results
        st.subheader("📊 Optimal Portfolio Allocation")
        allocation_df = pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["Weight"])
        st.write(allocation_df)

        # Plot Allocation
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=allocation_df.index, y=allocation_df["Weight"], palette="coolwarm", ax=ax)
        ax.set_title("Optimal Portfolio Allocation")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Stock")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Error computing Black-Litterman model: {str(e)}")
