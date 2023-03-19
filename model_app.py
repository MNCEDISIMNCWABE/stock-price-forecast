#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

def predict_stock_prices(start_date, end_date, tickers):
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)[['Close']]
    data.columns = data.columns.droplevel()
    data = data/100
    # Release Date from the index
    data = data.reset_index()

    df = pd.melt(data, id_vars='Date', value_vars=tickers)
    df.columns = ['ds', 'ticker', 'y']

    # Orders & Drivers data
    data_orders = df[['ds','ticker','y']]

    ##>>>>>>>>>>>>>>>> STAGE 1: Hourly Order Prediction
    df_grouped_orders = data_orders.groupby('ticker').filter(lambda x: len(x) >= 2)
    final_forecast_orders = pd.DataFrame(columns=['ticker','ds','yhat'])
    grouped_orders = df_grouped_orders.groupby('ticker')
    for branch in grouped_orders.groups:
        group_orders = grouped_orders.get_group(branch)
        m_orders = Prophet(interval_width=0.95)
        m_orders.fit(group_orders)
        future_orders = m_orders.make_future_dataframe(periods=465, freq='H')
        forecast_orders = m_orders.predict(future_orders)
        forecast_orders['ticker'] = branch
        final_forecast_orders = pd.concat([final_forecast_orders, forecast_orders], ignore_index=True)
        for_loop_forecast = final_forecast_orders[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]

    # give the tickers clear names
    ticker_names = []
    for ticker in tickers:
        if ticker=='CLS.JO':
            ticker_names.append('Clicks')
        elif ticker=='GLN.JO':
            ticker_names.append('Glencore')
        elif ticker=='PPH.JO':
            ticker_names.append('Pepkorh')
        elif ticker=='WHL.JO':
            ticker_names.append('Woolies')
        elif ticker=='APN.JO':
            ticker_names.append('Aspen')
        elif ticker=='RBP.JO':
            ticker_names.append('Royal-Bafokeng')
        elif ticker=='PIK.JO':
            ticker_names.append('PnP')
        elif ticker=='HIL.JO':
            ticker_names.append('HomeChoice')
        elif ticker=='SOL.JO':
            ticker_names.append('Sasol')
        elif ticker=='EXX.JO':
            ticker_names.append('Exxaro')
        elif ticker=='MCG.JO':
            ticker_names.append('MultiChoice')
        elif ticker=='AIL.JO':
            ticker_names.append('African-Rainbow')
        elif ticker=='TGA.JO':
            ticker_names.append('Thungela')
        elif ticker=='SSW.JO':
            ticker_names.append('Sibanye-Stillwater')
        else:
            ticker_names.append(ticker)
    for_loop_forecast['ticker_name'] = ticker_names

    # Output predicted prices and line plots
    for ticker in tickers:
        plot_data = for_loop_forecast[for_loop_forecast['ticker']==ticker]
        plot_data = plot_data.rename(columns={'ds':'Date', 'yhat':'Predicted Closing Price'})
        plot_data = plot_data.set_index('Date')
        st.write('## Predicted Closing Prices for ' + ticker)
        st.write(plot_data)

