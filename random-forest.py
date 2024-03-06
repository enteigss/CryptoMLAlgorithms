#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 03:37:54 2024

@author: jordangreen
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Features: prices last 30 days

# Load dataset

data = pd.read_csv('/Users/jordangreen/Desktop/CS365-Project/Datasets/crypto-values/BTC.csv')

# Preprocess data
window_size = 30
for i in range(window_size):
    data[f'Price_{i}'] = data['close'].shift(i)
    
data['Target'] = data['close'].shift(-1)
data = data.dropna().reset_index(drop=True)
    
X = data[[f'Price_{i}' for i in range(window_size)]]
y = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Iterative forecast

def iterative_forecasting(model, initial_input, steps):
    
    current_input = initial_input.copy()
    predictions = []
    
    for _ in range(steps):
        
        # Predict next step
        next_step_prediction = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(next_step_prediction)
        
        # Update the current input for the next prediction
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

