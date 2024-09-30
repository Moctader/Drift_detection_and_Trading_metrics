# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Pipeline to compare Evidently AI drift detection
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
    predicted_stock_price = build_and_train_lstm(X, y)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    actual_prices = close_prices[window_size:].flatten()
    predicted_prices = predicted_stock_price.flatten()

    # Create reference and current data for Evidently AI drift analysis
    ref_data = pd.DataFrame({'value': predicted_prices[:len(predicted_prices)//2]})
    curr_data = pd.DataFrame({'value': predicted_prices[len(predicted_prices)//2:]})

    # Evidently AI drift detection
    drift_report = evaluate_drift_evidently(ref_data[['value']], curr_data[['value']])
    print("\n--- Evidently AI Drift Report ---")
    print(drift_report)

    # Create a boolean mask for drift detection
    drift_mask_evidently = np.zeros(len(curr_data), dtype=bool)
    drift_mask_evidently[:len(drift_report)] = drift_report['is_drifted'].values

    # Highlight Evidently AI Drift Points
    dates = data.index[window_size:]
    drift_dates_evidently = dates[len(dates)//2:][drift_mask_evidently]
    drift_values_evidently = predicted_prices[len(predicted_prices)//2:][drift_mask_evidently]

    # Plotting Evidently AI Drift Points
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Stock Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Stock Price', color='orange')

    if len(drift_dates_evidently) > 0:
        plt.scatter(drift_dates_evidently, drift_values_evidently, color='purple', label='Evidently AI Drift', s=50)
    
    plt.title('Evidently AI Drift Detection')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    window_size = 60
    run_comparison_pipeline(window_size)