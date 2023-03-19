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
DEFAULT_TICKER = 'WHL.JO'

# Define the Streamlit app
st.title('Closing Stock Price Predictions')

# Add disclaimer to the sidebar
st.sidebar.markdown("""
**Disclaimer:**  
Please note that the author of this app is not a financial advisor and does not provide any investment advice. 
The purpose of this app is only to demonstrate machine learning lifecycle using techniques & tools like GitHub, Python, and Streamlit.  

Author: [Mncedisi Mncwabe](https://www.linkedin.com/in/mncedisi-mncwabe-a1b087171/).
""")


# Allow the user to input a ticker symbol
ticker_input = st.text_input('Enter Company Symbol', DEFAULT_TICKER)

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
st.plotly_chart(fig)

# Rename the columns of the results dataframe
results = results.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Predicted Lower Bound', 'yhat_upper': 'Predicted Upper Bound'})

# Display the predicted closing prices for the selected ticker and date range
st.write('Predicted Closing Stock Prices:')
st.write(results)
