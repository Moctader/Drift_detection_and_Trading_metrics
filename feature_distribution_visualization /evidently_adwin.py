# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from river.drift import ADWIN
from evidently.metrics import ColumnDriftMetric
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
    return model.predict(X)

# Function to evaluate drift using Evidently AI
def evaluate_drift_evidently(ref: pd.DataFrame, curr: pd.DataFrame):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="kl_div", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="wasserstein", stattest_threshold=0.1)
    ])
    report.run(reference_data=ref, current_data=curr)
    results = report.as_dict()
    drift_report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, metric in enumerate(results['metrics']):
        stat_test_name = metric['result'].get('stattest_name', 'Unknown')
        drift_report.loc[i, 'stat_test'] = stat_test_name
        drift_report.loc[i, 'drift_score'] = metric['result']['drift_score']
        drift_report.loc[i, 'is_drifted'] = metric['result']['drift_detected']
    return drift_report

# Create windows function for Evidently AI drift evaluation
def create_windows(data, column, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[column].iloc[i:i+window_size].values)
    return pd.DataFrame(windows, columns=[f'{column}_{i}' for i in range(window_size)]).stack().reset_index(drop=True).to_frame(name='value')

# Pipeline to compare ADWIN and Evidently AI drift detection
def run_comparison_pipeline(window_size=60):
    # 1. Fetch Apple stock data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Close prices
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create LSTM datasets
    X, y = create_dataset(scaled_data, window_size)

    # Train and predict using LSTM
    predicted_stock_price = build_and_train_lstm(X, y)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    actual_prices = close_prices[window_size:].flatten()
    predicted_prices = predicted_stock_price.flatten()

    # Initialize ADWIN for real-time drift detection
    adwin = ADWIN()

    # Visualize ADWIN Drift detection
    dates = data.index[window_size:]
    drift_dates_adwin = []
    drift_values_adwin = []

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Stock Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Stock Price', color='orange')

    # ADWIN drift detection over the predicted data
    for idx, (actual, predicted) in enumerate(zip(actual_prices, predicted_prices)):
        adwin.update(abs(actual - predicted))
        if adwin.drift_detected:
            drift_dates_adwin.append(dates[idx])
            drift_values_adwin.append(predicted)

    # Highlight ADWIN drift points
    if drift_dates_adwin:
        plt.scatter(drift_dates_adwin, drift_values_adwin, color='red', label='ADWIN Drift', s=50)
    
    plt.title('ADWIN Drift Detection')
    plt.legend()
    plt.show()

    # Evidently AI drift detection over sliding windows
    drift_reports_evidently = []
    drift_dates_evidently = []

    for start in range(0, len(data) - 2 * window_size, window_size):
        ref_window = data.iloc[start:start + window_size]
        curr_window = data.iloc[start + window_size:start + 2 * window_size]
        
        ref_windows = create_windows(ref_window, 'Close', window_size)
        curr_windows = create_windows(curr_window, 'Close', window_size)

        # Drift evaluation using Evidently
        drift_report = evaluate_drift_evidently(ref_windows, curr_windows)
        drift_reports_evidently.append(drift_report)

        # Check if drift was detected in the current window
        if drift_report['is_drifted'].any():
            drift_dates_evidently.append(dates[start + window_size])

    # Plot Evidently drift detection results
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Stock Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Stock Price', color='orange')

    if drift_dates_evidently:
        plt.scatter(drift_dates_evidently, 
                    [predicted_prices[dates.get_loc(d)] for d in drift_dates_evidently], 
                    color='green', label='Evidently Drift', s=50)

    plt.title('Evidently Drift Detection')
    plt.legend()
    plt.show()

    # Comparison Summary: ADWIN vs Evidently Drift Detection
    print("\n--- Drift Detection Comparison Summary ---")
    print(f"ADWIN detected drift at: {drift_dates_adwin}")
    print(f"Evidently detected drift at: {drift_dates_evidently}")
    
    # Optionally, you can print out detailed Evidently drift reports
    for i, report in enumerate(drift_reports_evidently):
        print(f"\nEvidently Drift Report for Window {i+1}:\n", report)

# Main function to run the comparison
if __name__ == "__main__":
    window_size = 60
    run_comparison_pipeline(window_size)
