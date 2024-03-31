import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

# Features: prices last 30 days

# Load dataset

data = yf.download("BTC-USD", start="2022-01-01", end="2024-03-05", interval="1d")

# Preprocess data
window_size = 30
shifted_data_frames = []
features =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for feature in features:
    for i in range(window_size): 
        column_name = f'{feature}_{i}'
        shifted_df = data[feature].shift(i).rename(column_name)
        shifted_data_frames.append(shifted_df)

shifted_data = pd.concat(shifted_data_frames, axis =1)
data = pd.concat([data, shifted_data], axis=1)

data['Target'] = data['Close'].shift(-1)
data = data.dropna().reset_index(drop=True)
    
X = data.loc[:, data.columns.str.contains('_\d+$')]
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

print(f"The Mean Absolute Error: {mae}")
print(f"The Mean Squared Error: {mse}")
print(f"The R2 Score: {r2}\n")
# Iterative forecast

def iterative_forecasting(model, initial_input, steps):
    #features in initial_input must be the same as those used during model training
    #initial_input should have the features in the same order as the model was trained on

    current_input = initial_input.copy()
    predictions = []
    
    for _ in range(steps):
        
        # Predict next step
        next_step_prediction = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(next_step_prediction)
        
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_step_prediction
    
    return predictions

#predicts the closing price for the next day 
feature_names = X_train.columns.tolist()
latest_input = data.loc[:, data.columns.str.contains('_\d+$')].iloc[-1].values.reshape(1, -1)
latest_input_df = pd.DataFrame(latest_input, columns=feature_names)
predicted_close_price = model.predict(latest_input_df)[0]
print(f"Predicted closing price for BTC on 2024-03-06: {predicted_close_price}")

#val = yf.download("BTC-USD", start="2024-03-06", end="2024-03-07", interval="1d")['Close'].iloc[0]
#print(f"Actual closing price for BTC on 2024-03-06: {val}")
