#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet

def train_model(ticker, start_date, end_date):
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    data.columns = ['y']
    data['ds'] = data.index
    
    # Train model
    model = Prophet(interval_width=0.95)
    model.fit(data)

    # Make predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    forecast = forecast.set_index('ds')

    return model, forecast

