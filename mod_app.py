#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from fbprophet import Prophet
import model

# Set page title
st.set_page_config(page_title="Stock Price Prediction App")

# Set page title and subtitle
st.title("Stock Price Prediction App")
st.write("This app predicts the future stock prices of the selected company.")

# Set up user input
tickers = st.text_input("Enter the tickers separated by commas (e.g., AAPL,GOOG,IBM)", value='CLS.JO')
start_date = st.date_input("Start date:")
end_date = st.date_input("End date:")

# Train models for each ticker
models = {}
for ticker in tickers.split(','):
    models[ticker] = model.train_model(ticker)

# Make predictions
predictions = {}
for ticker in tickers.split(','):
    # Create future dataframe
    future = models[ticker].make_future_dataframe(periods=(end_date - start_date).days)

    # Make prediction
    forecast = models[ticker].predict(future)

    # Filter predictions for the specified date range
    mask = (forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))
    predictions[ticker] = forecast.loc[mask]

# Display predictions
for ticker in tickers.split(','):
    st.write(f"Predictions for {ticker}:")
    st.write(predictions[ticker])

    # Plot predictions
    st.line_chart(predictions[ticker][['ds', 'yhat']])

