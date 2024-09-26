# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

# Function to create dataset with sliding window
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# LSTM model creation
def build_and_train_lstm(X, y):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=64, verbose=0)
    return model

# Pipeline to detect drift based on prediction errors
def run_comparison_pipeline(window_size):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create LSTM datasets
    X, y = create_dataset(scaled_data, window_size)

    # Train and predict using LSTM
    model = build_and_train_lstm(X, y)
    predicted_stock_price = model.predict(X)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    actual_prices = close_prices[window_size:].flatten()
    predicted_prices = predicted_stock_price.flatten()

    # Split data into reference and current sets
    split_index = len(predicted_prices) // 2
    ref_actual = actual_prices[:split_index]
    ref_predicted = predicted_prices[:split_index]
    curr_actual = actual_prices[split_index:]
    curr_predicted = predicted_prices[split_index:]

    # Calculate performance metrics
    ref_mae = mean_absolute_error(ref_actual, ref_predicted)
    curr_mae = mean_absolute_error(curr_actual, curr_predicted)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Detect drift points based on prediction errors
    error_threshold = ref_mae * 1.5  # Define a threshold for drift detection
    drift_mask = np.abs(curr_actual - curr_predicted) > error_threshold

    # Highlight drift points
    dates = data.index[window_size:]
    drift_dates = dates[split_index:][drift_mask]
    drift_values = curr_predicted[drift_mask]

    # Plotting Drift Points
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Stock Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Stock Price', color='orange')

    if len(drift_dates) > 0:
        plt.scatter(drift_dates, drift_values, color='red', label='Detected Drift', s=50)
    
    plt.title('Drift Detection Based on Prediction Errors')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    window_size = [60,90,120,180]
    for i in window_size:
        run_comparison_pipeline(i)