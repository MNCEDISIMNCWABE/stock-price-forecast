import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
st.title('Closing Stock Price Predictions App')

# Add disclaimer to the sidebar
st.sidebar.markdown("""
Welcome to the stock price prediction app! The app allows users to easily type the company ticker symbol as listed on yahoo finance, 
the date range and then the app will predict the closing stock price for the selected dates. 
The model is trained and tested on the historical stock price data from Yahoo Finance.
When you type the company ticker symbol, the app is able to take live data from the Yahoo Finance website and run it through the model to provide a prediction.
The results are presented in the form of a table and graph. The results also contain a 95% confidence interval (lower and upper bound) of the predicted stock price.

**Disclaimer:**  
Please note that the author of this app is not a financial advisor and does not provide any investment advice. 
The purpose of this app is only to demonstrate machine learning lifecycle using techniques & tools like GitHub, Python, and Streamlit.  

Author: [Mncedisi Mncwabe](https://www.linkedin.com/in/mncedisi-mncwabe-a1b087171/).
""")


# Allow the user to input a ticker symbol
ticker_input = st.text_input('Enter Ticker/Company Symbol:', DEFAULT_TICKER)

# Get Prophet model for selected ticker
model = get_model(ticker_input)

# Set the default start and end dates
start_date = datetime.today().date()
end_date = start_date + timedelta(days=1095)

# Allow the user to select a start and end date for the prediction
start_date = st.date_input('Select Prediction Start Date:', start_date)
end_date = st.date_input('Select Prediction End Date:', end_date)

# Run Prophet for the selected date range and store the results
results = run_prophet(model, start_date, end_date)

# Plot the predicted closing prices for the selected ticker and date range
st.write("***Predictions Plot:***")
fig = plot_plotly(model, results)
st.plotly_chart(fig)

# Rename the columns of the results dataframe
results = results.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Predicted Lower Bound', 'yhat_upper': 'Predicted Upper Bound'})

# Display the predicted closing prices for the selected ticker and date range
st.write('***Predicted Closing Stock Prices:***')
st.write(results)
