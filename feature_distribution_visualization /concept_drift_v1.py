import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from river.drift import ADWIN
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

# Step 1: Download Apple stock data (reference and current data)
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Preprocess the data (close price as the target)
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

window_size = 60

# Function to create dataset with a sliding window
def create_dataset(data, time_step=window_size):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# LSTM model creation
def build_and_train_lstm(X, y):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=64, verbose=0)
    return model

# Step 3: Create Reference and Current Data (Split into training and testing)
split_ratio = 0.9  # 90% Reference, 10% Current
split_index = int(len(scaled_data) * split_ratio)

# Reference Data (Historical data)
reference_data = scaled_data[:split_index]
X_ref, y_ref = create_dataset(reference_data)

# Current Data (New data)
current_data = scaled_data[split_index - window_size:]  # Include the last window_size data points from reference data
X_curr, y_curr = create_dataset(current_data)

# Step 4: Build and train the LSTM model on reference data
lstm_model = build_and_train_lstm(X_ref, y_ref)

# Step 5: Predict on both Reference and Current Data
predicted_ref = lstm_model.predict(X_ref)
predicted_curr = lstm_model.predict(X_curr)

# Inverse transform the predictions to original scale
predicted_ref = scaler.inverse_transform(predicted_ref)
predicted_curr = scaler.inverse_transform(predicted_curr)

actual_ref = scaler.inverse_transform(y_ref.reshape(-1, 1))
actual_curr = scaler.inverse_transform(y_curr.reshape(-1, 1))

# Step 6: Compare performance on Reference and Current Data
def print_metrics(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    print(f"\n--- {label} Metrics ---")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Metrics for Reference Data
print_metrics(actual_ref, predicted_ref, 'Reference')

# Metrics for Current Data
print_metrics(actual_curr, predicted_curr, 'Current')

# Step 7: Concept Drift Detection Using ADWIN (Real-time drift detection)
adwin = ADWIN()
drift_dates_adwin = []
drift_values_adwin = []

# Dates corresponding to current data (after the split index)
dates_curr = data.index[split_index:]  # Adjust for the sliding window

# Update ADWIN with the prediction error and check for drift
for idx, (actual, predicted) in enumerate(zip(actual_curr, predicted_curr)):
    adwin.update(abs(actual - predicted))  # Feed the absolute error to ADWIN
    if adwin.drift_detected:  # Check for drift using ADWIN
        drift_dates_adwin.append(dates_curr[idx])  # Log the date of drift
        drift_values_adwin.append(predicted)  # Log the predicted value at the drift point

# Plotting ADWIN Drift Points
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data.index[window_size:], close_prices[window_size:], label='Actual Stock Price', color='blue', linewidth=1)
ax.plot(data.index[window_size:split_index], predicted_ref, label='Predicted Reference', color='green', linestyle='--', linewidth=1)
ax.plot(dates_curr, predicted_curr, label='Predicted Current', color='orange', linestyle='--', linewidth=1)

# Highlight ADWIN Drift Points
ax.plot(drift_dates_adwin, drift_values_adwin, 'o', color='red', label='ADWIN Drift', markersize=5)

# Add title and legend
ax.set_title('Stock Price Prediction with Drift Detection (ADWIN)')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price (USD)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Step 8: Evidently AI for Drift Detection (Statistical tests on feature distributions)
def evaluate_drift_evidently(ref: pd.DataFrame, curr: pd.DataFrame):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1)
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

# Create reference and current data for Evidently AI drift analysis
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return pd.DataFrame(windows, columns=[f'window_{i}' for i in range(window_size)]).stack().reset_index(drop=True).to_frame(name='value')

# Sliding window for Evidently AI drift detection
ref_window = create_windows(reference_data.flatten(), window_size=window_size)
curr_window = create_windows(current_data.flatten(), window_size=window_size)

# Evidently AI drift detection
drift_report = evaluate_drift_evidently(ref_window, curr_window)
print("\n--- Evidently AI Drift Report ---")
print(drift_report)

# Create a boolean mask for drift detection
drift_mask_evidently = np.zeros(len(dates_curr), dtype=bool)
drift_mask_evidently[:len(drift_report)] = drift_report['is_drifted'].values

# Highlight Evidently AI Drift Points
drift_dates_evidently = dates_curr[drift_mask_evidently]
drift_values_evidently = predicted_curr[drift_mask_evidently]

# Plotting Evidently AI Drift Points
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data.index[window_size:], close_prices[window_size:], label='Actual Stock Price', color='blue', linewidth=1)
ax.plot(data.index[window_size:split_index], predicted_ref, label='Predicted Reference', color='green', linestyle='--', linewidth=1)
ax.plot(dates_curr, predicted_curr, label='Predicted Current', color='orange', linestyle='--', linewidth=1)

# Highlight Evidently AI Drift Points
ax.plot(drift_dates_evidently, drift_values_evidently, 'o', color='purple', label='Evidently AI Drift', markersize=5)

# Add title and legend
ax.set_title('Stock Price Prediction with Drift Detection (Evidently AI)')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price (USD)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Final Comparison of ADWIN and Evidently AI Drift Results
print(f"\nADWIN detected drift at {len(drift_dates_adwin)} points.")
print(f"Evidently AI Drift Detected: {drift_report['is_drifted'].sum()} occurrences.")
print("\nADWIN Drift Points:")
for date, value in zip(drift_dates_adwin, drift_values_adwin):
    print(f"Date: {date}, Predicted Value: {value[0]}")

print("\nEvidently AI Drift Points:")
for date, value in zip(drift_dates_evidently, drift_values_evidently):
    print(f"Date: {date}, Predicted Value: {value[0]}")