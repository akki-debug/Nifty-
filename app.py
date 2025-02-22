import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
import scipy.optimize as sco
from pypfopt import risk_models, expected_returns

# ---------------- Streamlit UI ----------------
st.title("📈 Black-Litterman Portfolio Optimization for NIFTY 50")
st.sidebar.header("Portfolio Inputs")

# ---------------- Step 1: Stock Selection ----------------
nifty50_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
selected_stocks = st.sidebar.multiselect("Select NIFTY 50 Stocks", nifty50_tickers, default=nifty50_tickers[:3])

# ---------------- Step 2: Market Inputs ----------------
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0) / 100
expected_market_return = st.sidebar.slider("Expected Market Return (%)", 0.0, 20.0, 12.0) / 100

# ---------------- Step 3: Fetch Data from Yahoo Finance ----------------
start_date = "2019-01-01"
end_date = date.today().strftime("%Y-%m-%d")

@st.cache_data
def fetch_data(tickers):
    data = yf.download(tickers, start=start_date, end=end_date)
    if data.empty:
        st.error("Failed to fetch stock data. Check tickers or internet connection.")
        return None
    
    if 'Adj Close' not in data.columns:
        st.error("Error: 'Adj Close' column not found. Yahoo Finance might be down.")
        return None
    
    return data['Adj Close']

if selected_stocks:
    stock_data = fetch_data(selected_stocks)
    if stock_data is None:
        st.warning("Stock data could not be loaded. Try different tickers or check your connection.")
    else:
        st.subheader("Stock Price Data")
        st.line_chart(stock_data)

        returns = stock_data.pct_change().dropna()
        
        # Compute Market Implied Returns (CAPM)
        cov_matrix = returns.cov() * 252
        market_caps = np.array([200, 180, 150, 140, 120][:len(selected_stocks)])  # Dummy Market Caps
        weights_market = market_caps / np.sum(market_caps)
        risk_aversion = 2.5  # Standard value
        pi = risk_aversion * np.dot(cov_matrix, weights_market)

        # ---------------- Step 4: Define Views ----------------
        st.sidebar.subheader("Subjective Views on Expected Returns")
        views = []
        confidence_levels = []
        
        for stock in selected_stocks:
            view = st.sidebar.slider(f"{stock} Expected Return (%)", -10.0, 10.0, 0.0) / 100
            confidence = st.sidebar.slider(f"{stock} Confidence (%)", 0.0, 100.0, 50.0) / 100
            views.append(view)
            confidence_levels.append(confidence)
        
        P = np.eye(len(selected_stocks))
        Q = np.array(views)
        omega = np.diag(confidence_levels) * cov_matrix.values.diagonal()

        # ---------------- Step 5: Compute Black-Litterman Posterior Returns ----------------
        tau = 0.025
        inv_cov = np.linalg.inv(tau * cov_matrix)
        M1 = np.linalg.inv(inv_cov + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
        M2 = np.dot(inv_cov, pi) + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)
        bl_returns = np.dot(M1, M2)
        bl_cov = cov_matrix + np.linalg.inv(inv_cov + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))

        # ---------------- Step 6: Portfolio Optimization ----------------
        def neg_sharpe(weights, cov_matrix, returns):
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_vol  # Maximize Sharpe Ratio

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(len(selected_stocks)))  # No short selling
        opt_result = sco.minimize(neg_sharpe, weights_market, args=(bl_cov, bl_returns),
                                  method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = opt_result.x
        optimal_portfolio = dict(zip(selected_stocks, optimal_weights))

        # ---------------- Step 7: Performance Metrics ----------------
        bl_portfolio_returns = (returns * optimal_weights).sum(axis=1)
        cumulative_returns = (1 + bl_portfolio_returns).cumprod()

        sharpe_ratio = (bl_portfolio_returns.mean() * 252) / (bl_portfolio_returns.std() * np.sqrt(252))
        max_drawdown = np.min(cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1)
        portfolio_volatility = bl_portfolio_returns.std() * np.sqrt(252)
        expected_return = bl_portfolio_returns.mean() * 252

        # ---------------- Step 8: Visualization ----------------
        st.subheader("Portfolio Allocation")
        st.bar_chart(pd.DataFrame(optimal_portfolio.values(), index=optimal_portfolio.keys(), columns=["Weight"]))

        st.subheader("Performance Metrics")
        st.write(f"✅ **Sharpe Ratio:** {sharpe_ratio:.2f}")
        st.write(f"📉 **Max Drawdown:** {max_drawdown:.2%}")
        st.write(f"📊 **Portfolio Volatility:** {portfolio_volatility:.2%}")
        st.write(f"📈 **Expected Return:** {expected_return:.2%}")

        st.subheader("Cumulative Portfolio Returns")
        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_returns, label="Black-Litterman Portfolio", linewidth=2)
        plt.legend()
        st.pyplot(plt)
else:
    st.warning("Please select at least one stock to proceed.")
