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
from evidently.metrics import ColumnDriftMetric, RegressionQualityMetric
from evidently.report import Report

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

# Function to evaluate drift using Evidently AI
def evaluate_drift_evidently(ref: pd.DataFrame, curr: pd.DataFrame, ref_actual: pd.DataFrame, curr_actual: pd.DataFrame):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="prediction", stattest="ks", stattest_threshold=0.05),  # Prediction Drift
        ColumnDriftMetric(column_name="target", stattest="ks", stattest_threshold=0.05),  # Target Drift
        RegressionQualityMetric()  # Model Performance
    ])
    ref_data = pd.concat([ref.rename(columns={"value": "prediction"}), ref_actual.rename(columns={"value": "target"})], axis=1)
    curr_data = pd.concat([curr.rename(columns={"value": "prediction"}), curr_actual.rename(columns={"value": "target"})], axis=1)
    report.run(reference_data=ref_data, current_data=curr_data)
    return report

# Pipeline to compare drift detection using various detectors
def run_comparison_pipeline(window_size, epochs):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Split data into reference (60%) and current (40%) sets
    split_index = int(len(scaled_data) * 0.6)
    ref_data = scaled_data[:split_index]
    curr_data = scaled_data[split_index - window_size:]  # Include window_size overlap

    # Create LSTM datasets
    X_ref, y_ref = create_dataset(ref_data, window_size)
    X_curr, y_curr = create_dataset(curr_data, window_size)
    print(len(X_ref), len(y_ref))  
    print(len(X_curr), len(y_curr))  

    # Train and predict using LSTM
    model = build_and_train_lstm(X_ref, y_ref, epochs)
    predicted_ref = model.predict(X_ref)
    predicted_curr = model.predict(X_curr)

    # Inverse transform predictions and actual values
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

    # Calculate predicted differences for current and reference sets separately
    predicted_ref_diff = np.diff(predicted_ref)  # 1D after np.diff
    predicted_curr_diff = np.diff(predicted_curr)  # 1D after np.diff

    # Align lengths by adding a 0 at the beginning (since np.diff reduces length by 1)
    predicted_ref_diff = np.insert(predicted_ref_diff, 0, 0)  # Add 0 at the beginning
    predicted_curr_diff = np.insert(predicted_curr_diff, 0, 0)  # Add 0 at the beginning

    actual_ref_diff = np.diff(actual_ref)  # 1D after np.diff   
    actual_curr_diff = np.diff(actual_curr)  # 1D after np.diff

    actual_curr_diff = np.insert(actual_curr_diff, 0, 0)  # Add 0 at the beginning
    actual_ref_diff = np.insert(actual_ref_diff, 0, 0)  # Add 0 at the beginning

    # Ensure we are using the correct lengths for the reference and current sets
    ref_actual = pd.DataFrame({'value': actual_ref_diff[:len(predicted_ref_diff)]})  # Reference set
    ref_predicted = pd.DataFrame({'value': predicted_ref_diff[:len(ref_actual)]})  # Predicted reference set

    curr_actual = pd.DataFrame({'value': actual_curr_diff[:len(predicted_curr_diff)]})  # Current set
    curr_predicted = pd.DataFrame({'value': predicted_curr_diff[:len(curr_actual)]})  # Predicted current set
    
    # Debugging information
    print(f"Reference Actual Length: {len(ref_actual)}, Current Actual Length: {len(curr_actual)}")
    print(f"Reference Predicted Length: {len(ref_predicted)}, Current Predicted Length: {len(curr_predicted)}")

    # Evidently AI drift detection
    report = evaluate_drift_evidently(ref_predicted, curr_predicted, ref_actual, curr_actual)
    report.save_html("evidently_report.html")

    print("\n--- Evidently AI Drift Report ---")
    report.show()

    # Prepare dates for plotting
    dates = data.index[window_size:]  # Adjust the dates to account for the window size
    ref_dates = dates[:len(ref_actual)]  # Reference dates must match the length of ref_actual

    # Ensure that curr_dates and actual_curr lengths are aligned
    curr_dates = dates[len(ref_dates):len(ref_dates) + len(curr_actual)]  # Start from split_index and match length
    print(f"Length of ref_dates: {len(ref_dates)}, Length of curr_dates: {len(curr_dates)}")
    print(f"Length of actual_ref: {len(actual_ref)}, Length of actual_curr: {len(actual_curr)}")

    # Plotting Drift Points
    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, actual_ref[:len(ref_dates)], label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, predicted_ref[:len(ref_dates)], label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, actual_curr[:len(curr_dates)], label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, predicted_curr[:len(curr_dates)], label='Predicted Stock Price (Current)', color='red')

    plt.title(f'Multiple Drift Detection Algorithms (Epochs: {epochs})')
    plt.legend()
    plt.savefig(f'drift_detection_epochs_{epochs}.png')
    plt.show()

if __name__ == "__main__":
    window_size = 60
    for epochs in [2]:  # You can change this value to test with different epochs
        run_comparison_pipeline(window_size, epochs)