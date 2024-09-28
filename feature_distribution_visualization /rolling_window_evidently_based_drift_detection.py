# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    model.fit(X, y, epochs=70, batch_size=64, validation_data=validation_data, 
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

# Pipeline to compare Evidently AI drift detection
def run_comparison_pipeline(window_size, epochs):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create LSTM datasets
    X, y = create_dataset(scaled_data, window_size)
    model = build_and_train_lstm(X, y)

    # Predict prices using the trained model
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Calculate predicted differences
    predicted_diff = np.diff(predicted_prices, axis=0)
    predicted_diff = np.vstack([np.zeros((1, 1)), predicted_diff]).flatten()  # Align lengths

    # Calculate actual differences
    actual_diff = np.diff(close_prices, axis=0)
    actual_diff = np.vstack([np.zeros((1, 1)), actual_diff]).flatten()

    # Split data into reference and current sets
    split_index = len(predicted_prices) // 2

    # Ensure we are using the same length for the reference and current sets
    ref_actual = pd.DataFrame({'value': actual_diff[:split_index]})
    ref_predicted = pd.DataFrame({'value': predicted_diff[:split_index]})
    curr_actual = pd.DataFrame({'value': actual_diff[split_index:split_index + len(predicted_diff[split_index:])]})
    curr_predicted = pd.DataFrame({'value': predicted_diff[split_index:split_index + len(predicted_diff[split_index:])]})

    # Print statements for debug
    print(ref_actual.head())
    print(ref_predicted.head())
    print(curr_actual.head())
    print(curr_predicted.head())

    # Evidently AI drift detection
    report = evaluate_drift_evidently(ref_predicted, curr_predicted, ref_actual, curr_actual)
    report.save_html("evidently_report.html")
    print("\n--- Evidently AI Drift Report ---")
    report.show()


    dates = data.index[window_size:]  # Ensure the correct dates for plotting
    ref_dates = dates[:split_index]
    curr_dates = dates[split_index:]

    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, ref_actual['value'], label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, ref_predicted['value'], label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, curr_actual['value'], label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, curr_predicted['value'], label='Predicted Stock Price (Current)', color='red')

    plt.title('Evidently AI Drift Detection')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    window_size = 60
    run_comparison_pipeline(window_size, epochs=70)
