import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

BTC_data = yf.download("BTC-USD", period="max", interval="1d")
print(BTC_data.head(3))
print(BTC_data.tail(3))
BTC_data.index = pd.to_datetime(BTC_data.index)
# Function utilizing Random Forest for yf data and outputs MAE, MSE, and R2
def RandomForest(d): 
    # Preprocess data
    window_size = 30
    shifted_data_frames = []
    features =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for feature in features:
        for i in range(window_size): 
            column_name = f'{feature}_{i}'
            shifted_df = d[feature].shift(i).rename(column_name)
            shifted_data_frames.append(shifted_df)

    shifted_data = pd.concat(shifted_data_frames, axis =1)
    d = pd.concat([d, shifted_data], axis=1)

    d['Target'] = d['Close'].shift(-1)
    d = d.dropna().reset_index(drop=True)
    
    X = d.loc[:, d.columns.str.contains('_\d+$')]
    y = d['Target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=42)

    # Initialize and train the random forest model
    model = RandomForestRegressor(n_estimators=200, random_state=42)

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

    # Assuming you have already created this DataFrame with necessary preprocessing for model input
    feature_names = X_train.columns.tolist()
    latest_input = d.loc[:, d.columns.str.contains('_\d+$')].iloc[-1].values.reshape(1, -1)
    latest_input_df = pd.DataFrame(latest_input, columns=feature_names)

    # Load your trained model (not shown how it's loaded here, assuming 'model' is already available)
    predicted_close_price = model.predict(latest_input_df)[0]
    print(f"Predicted closing price for BTC on the next day: {predicted_close_price}")
    """
    # Calculating the next day
    latest_date = d.index.max()  # Get the latest date from your DataFrame
    next_day = latest_date + pd.Timedelta(days=1)  # Calculate the next day

    print(f"The date for the next day is: {next_day}")
    """


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

RandomForest(BTC_data)
"""
latest_date = BTC_data.index.max()
next_day = latest_date + pd.Timedelta(days=1)
print(latest_date[:])

#predicts the closing price for the next day 
feature_names = X_train.columns.tolist()
latest_input = data.loc[:, data.columns.str.contains('_\d+$')].iloc[-1].values.reshape(1, -1)
latest_input_df = pd.DataFrame(latest_input, columns=feature_names)
predicted_close_price = model.predict(latest_input_df)[0]
print(f"Predicted closing price for BTC on next day: {predicted_close_price}")

#val = yf.download("BTC-USD", start="2024-03-06", end="2024-03-07", interval="1d")['Close'].iloc[0]
#print(f"Actual closing price for BTC on 2024-03-06: {val}")
"""