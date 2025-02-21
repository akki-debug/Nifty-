import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from black_litterman import BlackLittermanModel

class NiftyPortfolioOptimizer:
    def __init__(self):
        self.nifty_stocks = self._get_nifty_constituents()
        
    def _get_nifty_constituents(self):
        # List of current NIFTY 50 stocks
        return ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 
                'INFY', 'TCS', 'BHARTIARTL', 'ASIANPAINT']  # Add all 50 stocks
        
    def get_market_caps(self):
        caps = {}
        for stock in self.nifty_stocks:
            ticker = yf.Ticker(f"{stock}.NS")
            info = ticker.info
            if 'marketCap' in info:
                caps[stock] = info['marketCap']
        return caps
    
    def calculate_prior_returns(self):
        returns = {}
        for stock in self.nifty_stocks:
            ticker = yf.Ticker(f"{stock}.NS")
            hist = ticker.history(period='1y')
            if len(hist) > 0:
                returns[stock] = hist['Close'].pct_change().mean() * 252
        return returns

def main():
    st.title('NIFTY 50 Black-Litterman Portfolio Optimizer')
    st.sidebar.header('Model Parameters')
    
    # Get market cap weights and prior returns
    optimizer = NiftyPortfolioOptimizer()
    market_caps = optimizer.get_market_caps()
    prior_returns = optimizer.calculate_prior_returns()
    
    # Calculate initial market cap weights
    total_cap = sum(market_caps.values())
    market_weights = {stock: cap/total_cap for stock, cap in market_caps.items()}
    
    # Display current market weights
    st.subheader('Current Market Cap Weights')
    fig = go.Figure(data=[go.Pie(labels=list(market_weights.keys()),
                                values=list(market_weights.values()))])
    st.plotly_chart(fig, use_container_width=True)
    
    # Views Input Section
    st.sidebar.subheader('Add Your Views')
    num_views = st.sidebar.number_input('Number of Views', min_value=0, max_value=len(market_caps))
    
    views = {}
    confidences = {}
    
    for i in range(num_views):
        st.sidebar.subheader(f'View {i+1}')
        stock = st.sidebar.selectbox(f'Select Stock {i+1}', list(market_caps.keys()))
        view_return = st.sidebar.number_input(f'Return View ({stock})', min_value=-100.0, max_value=100.0)
        confidence = st.sidebar.slider(f'Confidence Level ({stock})', min_value=0.0, max_value=1.0, value=0.5)
        
        views[stock] = view_return
        confidences[stock] = confidence
    
    # Run Black-Litterman Model
    if len(views) > 0:
        bl_model = BlackLittermanModel(
            prior_returns=prior_returns,
            prior_confidence=0.5,
            views=views,
            confidence=confidences
        )
        
        posterior_returns = bl_model.bl_returns()
        optimal_weights = bl_model.market_clearing_weights()
        
        # Display Results
        st.subheader('Optimal Portfolio Weights')
        fig = go.Figure(data=[go.Pie(labels=list(optimal_weights.index),
                                    values=list(optimal_weights))])
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics
        st.subheader('Portfolio Statistics')
        portfolio_stats = pd.DataFrame({
            'Prior Returns (%)': [prior_returns.get(stock, 0) * 100 for stock in market_caps],
            'Posterior Returns (%)': [posterior_returns.get(stock, 0) * 100 for stock in market_caps],
            'Optimal Weight (%)': [optimal_weights.get(stock, 0) * 100 for stock in market_caps]
        })
        st.table(portfolio_stats.round(2))

if __name__ == '__main__':
    main()

