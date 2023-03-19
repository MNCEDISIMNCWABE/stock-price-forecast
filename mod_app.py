#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from model import get_model, run_prophet
import matplotlib.pyplot as plt

# Set the default ticker symbol
#DEFAULT_TICKER = 'AAPL'

# Define the Streamlit app
st.title('Stock Price Prediction with Prophet')

# Allow the user to select a ticker symbol
ticker = st.selectbox('Select Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA'], index=0)

# Get Prophet model for selected ticker
model = get_model(ticker)

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

