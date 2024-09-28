# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
def build_and_train_lstm(X, y):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=64, verbose=0)
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
def run_comparison_pipeline(window_size):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Calculate rolling window differences
    diff_data = np.diff(scaled_data, axis=0)
    diff_data = np.vstack([np.zeros((1, 1)), diff_data])  # Add a zero at the beginning to match the original length

    # Create LSTM datasets
    X, y = create_dataset(diff_data, window_size)

    # Train and predict using LSTM
    model = build_and_train_lstm(X, y)
    predicted_diff = model.predict(X)
    predicted_diff = scaler.inverse_transform(predicted_diff)
    actual_diff = np.diff(close_prices, axis=0)
    actual_diff = np.vstack([np.zeros((1, 1)), actual_diff]).flatten()
    predicted_diff = predicted_diff.flatten()

    # Reconstruct the predicted prices from differences
    predicted_prices = np.cumsum(predicted_diff) + close_prices[0]

    # Split data into reference and current sets
    split_index = len(predicted_prices) // 2
    ref_actual = pd.DataFrame({'value': actual_diff[:split_index]})
    ref_predicted = pd.DataFrame({'value': predicted_diff[:split_index]})
    curr_actual = pd.DataFrame({'value': actual_diff[split_index:]})
    curr_predicted = pd.DataFrame({'value': predicted_diff[split_index:]})

    # Evidently AI drift detection
    report = evaluate_drift_evidently(ref_predicted, curr_predicted, ref_actual, curr_actual)
    report.save_html("evidently_report.html")
    print("\n--- Evidently AI Drift Report ---")
    report.show()

    # Plotting Drift Points
    dates = data.index[window_size:]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, close_prices[window_size:], label='Actual Stock Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Stock Price', color='orange')
    
    plt.title('Evidently AI Drift Detection')
    plt.legend()
    #plt.show()

if __name__ == "__main__":
    window_size = 60
    run_comparison_pipeline(window_size)