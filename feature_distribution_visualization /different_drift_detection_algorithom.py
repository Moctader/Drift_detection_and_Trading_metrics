# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error
from river.drift import PageHinkley, ADWIN, KSWIN

# Function to create dataset with sliding window
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# LSTM model creation
def build_and_train_lstm(X, y, validation_data=None):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    model.fit(X, y, epochs=100, batch_size=64, validation_data=validation_data, 
              callbacks=[early_stop, reduce_lr], verbose=1)
    
    return model

# Pipeline to compare drift detection using various detectors
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

    # Initialize Drift Detectors
    detectors = {
        "Page-Hinkley": PageHinkley(),
        "ADWIN": ADWIN(),
        "KSWIN": KSWIN(alpha=0.05)  # KS Test with a confidence level of 95%
    }

    # Detect drift points using each drift detector
    drift_results = {detector: {'dates': [], 'values': []} for detector in detectors}
    dates = data.index[window_size:]
    
    for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
        error = abs(actual - predicted)

        # Update each detector with the error and log drift points
        for detector_name, detector in detectors.items():
            detector.update(error)
            if detector.drift_detected:
                drift_results[detector_name]['dates'].append(dates[split_index + idx])
                drift_results[detector_name]['values'].append(predicted)

    # Plotting Drift Points
    ref_dates = dates[:split_index]
    curr_dates = dates[split_index:]

    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, ref_actual, label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, ref_predicted, label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, curr_actual, label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, curr_predicted, label='Predicted Stock Price (Current)', color='red')

    # Plot drift points for each detector
    colors = ['purple', 'magenta', 'cyan', 'black']
    for i, (detector_name, result) in enumerate(drift_results.items()):
        if len(result['dates']) > 0:
            plt.scatter(result['dates'], result['values'], color=colors[i], label=f'{detector_name} Drift', s=50)

    plt.title('Multiple Drift Detection Algorithms')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    window_size = 60
    run_comparison_pipeline(window_size)
