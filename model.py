#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

def get_model(ticker):
    # Download historical data from Yahoo Finance
    data = yf.download(ticker, start='2014-01-02', end=pd.to_datetime("today").strftime("%Y-%m-%d"))[['Close']]
    data.columns = ['y']
    data['ds'] = data.index
    
    # Fit a Prophet model to the data
    model = Prophet()
    model.fit(data)
    
    return model

def run_prophet(model, start_date, end_date):
    # Make future dataframe for prediction date range
    future = model.make_future_dataframe(periods=(end_date - start_date).days + 1, include_history=False)

    # Make prediction
    forecast = model.predict(future)

    # Filter prediction for selected date range
    forecast = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]

    return forecast[['ds', 'yhat']]


# In[ ]:




