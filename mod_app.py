import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from model import get_model, run_prophet
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt


# Set the default ticker symbol
DEFAULT_TICKER = 'AAPL'

# Define the Streamlit app
st.title('Stock Price Prediction')

# Allow the user to input a ticker symbol
ticker_input = st.text_input('Enter Ticker Symbol', DEFAULT_TICKER)

# Validate the ticker symbol and get Prophet model for selected ticker
try:
    yf.Ticker(ticker_input).info  # Check if ticker is valid
    model = get_model(ticker_input)
except:
    st.error('Invalid Ticker Symbol')
    st.stop()

# Allow the user to select a start and end date for the prediction
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Run Prophet for the selected date range and store the results
results = run_prophet(model, start_date, end_date)

# Display the predicted closing prices for the selected ticker and date range
st.write(results)

# Plot the predicted closing prices for the selected ticker and date range
fig = plot_plotly(model, results)
st.plotly_chart(fig)
