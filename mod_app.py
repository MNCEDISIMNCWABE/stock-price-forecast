#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model

# Set start and end dates for data download
start_date = '2014-01-02'
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")   

# Set page title
st.set_page_config(page_title='Stock Price Prediction', layout='wide')

# Define function to plot stock price predictions
def plot_predictions(predictions):
    fig = px.line(predictions, x=predictions.index, y='yhat', title='Stock Price Predictions')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Stock Price',
        legend_title=None,
        width=1200,
        height=600
    )
    st.plotly_chart(fig)

# Define function to get user inputs
def get_inputs():
    tickers = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL,MSFT)").upper().split(',')
    start_date = st.date_input("Enter start date", value=pd.to_datetime("today") - pd.Timedelta(days=365))
    end_date = st.date_input("Enter end date", value=pd.to_datetime("today"))
    return tickers, start_date, end_date

# Main function to run the app
def main():
    # Get user inputs
    tickers, start_date, end_date = get_inputs()

    # Download data and train models for each ticker
    models = {}
    predictions = pd.DataFrame(columns=['ds'])
    for ticker in tickers:
        model, forecast = train_model(ticker, start_date, end_date)
        models[ticker] = model
        forecast.columns = ['yhat', 'yhat_upper', 'yhat_lower']
        forecast['ticker'] = ticker
        predictions = pd.concat([predictions, forecast], axis=0)

    # Plot stock price predictions
    if not predictions.empty:
        predictions.reset_index(inplace=True)
        predictions = predictions.pivot(index='ds', columns='ticker', values='yhat')
        st.write(predictions)
        plot_predictions(predictions)

if __name__ == '__main__':
    main()

