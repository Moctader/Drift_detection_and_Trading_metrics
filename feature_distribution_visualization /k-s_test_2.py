import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

# Function to create dataset with sliding window
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# LSTM model creation
def build_and_train_lstm(X, y, epochs, validation_data=None):
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
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    model.fit(X, y, epochs=epochs, batch_size=64, validation_data=validation_data, 
              callbacks=[early_stop, reduce_lr], verbose=1)
    
    return model

# Pipeline to compare drift detection using K-S test
def run_comparison_pipeline(window_size, epochs):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Split data into reference (70%) and current (30%) sets
    split_index = int(len(scaled_data) * 0.7)
    ref_data = scaled_data[:split_index]
    curr_data = scaled_data[split_index - window_size:]  # Include window_size overlap

    # Create LSTM datasets
    X_ref, y_ref = create_dataset(ref_data, window_size)
    X_curr, y_curr = create_dataset(curr_data, window_size)

    # Train and predict using LSTM
    model = build_and_train_lstm(X_ref, y_ref, epochs)
    predicted_ref = model.predict(X_ref)
    predicted_curr = model.predict(X_curr)
    
    predicted_ref = scaler.inverse_transform(predicted_ref)
    predicted_curr = scaler.inverse_transform(predicted_curr)
    
    actual_ref = scaler.inverse_transform(y_ref.reshape(-1, 1)).flatten()
    actual_curr = scaler.inverse_transform(y_curr.reshape(-1, 1)).flatten()
    predicted_ref = predicted_ref.flatten()
    predicted_curr = predicted_curr.flatten()

    # Calculate performance metrics
    ref_mae = mean_absolute_error(actual_ref, predicted_ref)
    curr_mae = mean_absolute_error(actual_curr, predicted_curr)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Calculate maximum deviation
    ref_max_deviation = np.max(np.abs(actual_ref - predicted_ref))
    print(f"Reference Max Deviation: {ref_max_deviation}")

    # Drift detection using K-S Test
    drift_results = {'K-S Test': {'dates': [], 'values': []}}
    dates = data.index[window_size:]  # Adjust the dates to account for the window size
    ref_dates = dates[:len(actual_ref)]  # Reference dates must match the length of actual_ref
    curr_dates = dates[len(actual_ref):]  # Current dates must match the length of actual_curr

    for idx in range(window_size, len(actual_curr)):
        ref_segment = actual_ref[idx-window_size:idx]
        curr_segment = actual_curr[idx-window_size:idx]
        
        ks_stat, p_value = ks_2samp(ref_segment, curr_segment)
        
        if p_value < 0.05:  # Drift detected
            drift_results['K-S Test']['dates'].append(curr_dates[idx])
            drift_results['K-S Test']['values'].append(predicted_curr[idx])

    # Plotting Drift Points
    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, actual_ref, label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, predicted_ref, label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, actual_curr, label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, predicted_curr, label='Predicted Stock Price (Current)', color='red')

    # Plot drift points for K-S Test
    if len(drift_results['K-S Test']['dates']) > 0:
        plt.scatter(drift_results['K-S Test']['dates'], drift_results['K-S Test']['values'], color='purple', label='K-S Test Drift', s=50)

    plt.title(f'Drift Detection using K-S Test (Epochs: {epochs})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    window_size = 60
    for epochs in [100, 150, 400]:
        run_comparison_pipeline(window_size, epochs)