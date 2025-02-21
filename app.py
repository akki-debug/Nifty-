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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_app():
    """Configure the Streamlit app"""
    st.set_page_config(
        page_title="Black-Litterman Portfolio Optimizer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📊 Black-Litterman Portfolio Optimizer")
    st.write("Optimize your portfolio using the Black-Litterman model with market views.")
    
    return st.sidebar.header("⚙️ Model Inputs")

def validate_inputs(selected_assets, start_date, end_date, market_return, risk_free_rate, confidence):
    """Validate all user inputs"""
    errors = []
    
    if len(selected_assets) < 2:
        errors.append("Please select at least 2 stocks for diversification")
        
    if (end_date - start_date).days < 90:
        errors.append("Time period should be at least 90 days for reliable analysis")
        
    if market_return <= risk_free_rate:
        errors.append("Market return should be higher than risk-free rate")
        
    return errors

def fetch_stock_data(stocks, start_date, end_date):
    """Fetch stock data with error handling"""
    try:
        logger.info(f"Fetching data for {len(stocks)} stocks...")
        prices = yf.download(stocks, start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))["Adj Close"]
        
        # Validate data quality
        if prices.empty:
            raise ValueError("No price data available for selected period")
            
        if prices.isnull().values.any():
            raise ValueError("Missing data points detected")
            
        return prices
        
    except Exception as e:
        logger.error(f"Failed to fetch stock data: {str(e)}")
        raise

def compute_portfolio_metrics(prices, market_return, risk_free_rate, confidence, user_views):
    """Compute portfolio metrics with error handling"""
    try:
        # Compute returns & covariance
        returns = prices.pct_change().dropna()
        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(returns).ledoit_wolf()
        
        # Market Caps (Approximated)
        market_caps = {stock: prices[stock].iloc[-1] * 1e6 for stock in prices.columns}
        
        # Convert user views into expected returns
        Q = np.array([user_views[stock] for stock in prices.columns])
        P = np.identity(len(prices.columns))
        
        # Black-Litterman Model
        bl = BlackLittermanModel(S, pi=mu, market_caps=market_caps, 
                                absolute_views=Q, P=P, tau=confidence)
        bl_mu = bl.bl_returns
        
        # Optimize Portfolio
        ef = EfficientFrontier(bl_mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        return cleaned_weights, ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
    except Exception as e:
        logger.error(f"Failed to compute portfolio metrics: {str(e)}")
        raise

def main():
    """Main application function"""
    setup_app()
    
    # NIFTY 50 Stock List
    nifty50_stocks = [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
        "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
    ]

    # Get user inputs
    selected_assets = st.multiselect(
        "📌 Select Stocks for Portfolio", 
        nifty50_stocks, 
        nifty50_stocks[:5],
        key="selected_assets"
    )

    # Market Expected Return & Risk-Free Rate
    col1, col2 = st.columns(2)
    market_return = col1.slider(
        "📈 Expected Market Return (%)", 
        5, 20, 10, 
        key="market_return"
    ) / 100
    risk_free_rate = col2.slider(
        "🏦 Risk-Free Rate (%)", 
        2, 10, 5, 
        key="risk_free_rate"
    ) / 100
    confidence = st.slider(
        "💡 Investor Confidence in Views (%)", 
        0, 100, 50, 
        key="confidence"
    ) / 100

    # User Views on Expected Returns
    st.subheader("🔮 Your Views on Returns (%)")
    user_views = {}
    for stock in selected_assets:
        view = st.number_input(
            f"{stock}", 
            min_value=-10.0, 
            max_value=20.0, 
            value=5.0, 
            key=f"user_view_{stock}"
        ) / 100
        user_views[stock] = view

    # Date Range
    col3, col4 = st.columns(2)
    start_date = col3.date_input(
        "📅 Start Date", 
        value=pd.to_datetime("2023-01-01"),
        key="start_date"
    )
    end_date = col4.date_input(
        "📅 End Date", 
        value=pd.to_datetime("2023-12-31"),
        key="end_date"
    )

    if st.button("🔍 Optimize Portfolio"):
        try:
            # Validate inputs
            errors = validate_inputs(
                selected_assets, 
                start_date, 
                end_date, 
                market_return, 
                risk_free_rate, 
                confidence
            )
            
            if errors:
                for error in errors:
                    st.error(error)
                return
            
            # Fetch and process data
            prices = fetch_stock_data(selected_assets, start_date, end_date)
            
            # Compute portfolio metrics
            cleaned_weights, (expected_return, volatility, sharpe_ratio) = compute_portfolio_metrics(
                prices, 
                market_return, 
                risk_free_rate, 
                confidence, 
                user_views
            )

            # Display Results
            st.subheader("📊 Portfolio Performance Metrics")
            col5, col6, col7 = st.columns(3)
            col5.metric(label="Expected Return", value=f"{expected_return:.2%}")
            col6.metric(label="Volatility", value=f"{volatility:.2%}")
            col7.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")

            # Display Portfolio Allocation
            st.subheader("📌 Optimal Portfolio Allocation")
            allocation_df = pd.DataFrame.from_dict(cleaned_weights, 
                                                 orient="index", 
                                                 columns=["Weight"])
            st.write(allocation_df)

            # Plot Allocation
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=allocation_df.index, 
                       y=allocation_weights["Weight"], 
                       palette="coolwarm", 
                       ax=ax)
            ax.set_title("Optimal Portfolio Allocation")
            ax.set_ylabel("Weight")
            ax.set_xlabel("Stock")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Download Portfolio Weights
            csv = allocation_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "📥 Download Portfolio Weights (CSV)",
                data=csv,
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )

        except Exception as e:
            logger.error(f"Error processing portfolio optimization: {str(e)}")
            st.error("🚨 An error occurred during portfolio optimization. Please check your inputs.")

if __name__ == "__main__":
    main()

