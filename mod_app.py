import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from model import get_model, run_prophet


# Set the default ticker symbol
DEFAULT_TICKER = 'WHL.JO'

# Define the Streamlit app
st.title('Closing Stock Price Forecasting App')

# Allow the user to input a ticker symbol
ticker_input = st.text_input('Enter Ticker/Company Symbol', DEFAULT_TICKER)

# Get Prophet model for selected ticker
model = get_model(ticker_input)

# Allow the user to select a start and end date for the prediction
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Run Prophet for the selected date range and store the results
results = run_prophet(model, start_date, end_date)

# Plot the predicted closing prices for the selected ticker and date range
st.write("Prediction Plot:")
fig = plot_plotly(model, results)

# Create a trace for the actual closing prices
trace_actual = go.Scatter(
    x=results['ds'],
    y=results['y'],
    mode='lines',
    name='Actual Closing Price'
)

# Add the actual closing prices trace to the plot
fig.add_trace(trace_actual)

# Add a legend to the plot
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig)

# Rename the columns of the results dataframe
results = results.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Predicted Lower Bound', 'yhat_upper': 'Predicted Upper Bound', 'Actual Price': 'Actual Closing Price'})

# Display the predicted and actual closing prices for the selected ticker and date range
st.write('Predicted and Actual Closing Stock Prices:')
st.write(results)
