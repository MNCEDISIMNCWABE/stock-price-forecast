#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from prophet import Prophet

def train_model(ticker):
    # Download data from 2014-01-02 to current date
    df = pd.read_csv(f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1388552400&period2=1616197200&interval=1d&events=history&includeAdjustedClose=true")

    # Prepare data for Prophet model
    df = df[['Date', 'Close']]
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Train model
    model = Prophet(interval_width=0.95)
    model.fit(df)

    return model

