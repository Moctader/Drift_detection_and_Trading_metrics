# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from evidently.metrics import ColumnDriftMetric, RegressionQualityMetric
from evidently.report import Report

# Function to create synthetic time series data
def create_synthetic_data(length=1000, freq=0.1):
    t = np.arange(length)
    data = np.sin(2 * np.pi * freq * t)
    return data

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

# Fetch AAPL stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Preprocess Close prices
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Split data into reference (60%) and current (40%) sets
split_index = int(len(scaled_data) * 0.6)
ref_data = scaled_data[:split_index]
curr_data = scaled_data[split_index:]

# Calculate differences
ref_data_diff = np.diff(ref_data, axis=0).flatten()
curr_data_diff = np.diff(curr_data, axis=0).flatten()

# Introduce slight drift in the current data
slight_drift = np.linspace(0, -0.9, len(curr_data_diff))
curr_data_diff_with_slight_drift = curr_data_diff + slight_drift

# Generate synthetic data that mimics the stock data
synthetic_data = create_synthetic_data(length=len(scaled_data))
synthetic_ref_data = synthetic_data[:split_index]
synthetic_curr_data = synthetic_data[split_index:]

# Introduce drift in the synthetic prediction data by adding a linear trend
drift = np.linspace(0, 1, len(synthetic_curr_data))
synthetic_curr_prediction_with_drift = synthetic_curr_data + drift

# Convert to DataFrames
ref_data_df = pd.DataFrame({'value': ref_data_diff})
curr_data_df = pd.DataFrame({'value': curr_data_diff})
synthetic_ref_data_df = pd.DataFrame({'value': synthetic_ref_data})
synthetic_curr_data_df = pd.DataFrame({'value': synthetic_curr_data})
synthetic_curr_prediction_with_drift_df = pd.DataFrame({'value': synthetic_curr_prediction_with_drift})
synthetic_ref_prediction_with_drift_df = pd.DataFrame({'value': synthetic_ref_data})

# Trend Decomposition: Separate the trend from the data
trend = np.linspace(0, 1, len(scaled_data))
ref_data_with_trend = ref_data.flatten() + trend[:split_index]
curr_data_with_trend = curr_data.flatten() + trend[split_index:]

# Reverse the trend for drift comparison
reversed_trend = -trend
ref_data_with_reversed_trend = ref_data.flatten() + reversed_trend[:split_index]
curr_data_with_reversed_trend = curr_data.flatten() + reversed_trend[split_index:]

# Add the trend back to the differenced data
def add_trend(diff_data, initial_value, trend):
    return np.cumsum(np.insert(diff_data, 0, initial_value)) + trend

ref_data_with_trend_added_back = add_trend(ref_data_diff, ref_data[0], trend[:split_index])
curr_data_with_trend_added_back = add_trend(curr_data_diff, curr_data[0], trend[split_index:])

# Plot the original stock data, processed data, synthetic data, and drift detection
fig, axs = plt.subplots(5, 1, figsize=(12, 22))

# Original Stock Data
axs[0].plot(data['Close'], label='Original Stock Data', color='blue')
axs[0].set_title('Original AAPL Stock Data')
axs[0].legend()

# Processed Data
axs[1].plot(ref_data_diff, label='Reference Stock Data (Processed)', color='blue')
axs[1].plot(range(len(ref_data_diff), len(ref_data_diff) + len(curr_data_diff)), curr_data_diff, label='Current Stock Data (Processed)', color='green')
axs[1].set_title('Processed AAPL Stock Data')
axs[1].legend()

# Synthetic Data
axs[2].plot(synthetic_ref_data, label='Synthetic Reference Data', color='blue')
axs[2].plot(range(len(synthetic_ref_data), len(synthetic_ref_data) + len(synthetic_curr_data)), synthetic_curr_data, label='Synthetic Current Data', color='green')
axs[2].set_title('Synthetic Data')
axs[2].legend()

# Drift Detection
axs[3].plot(ref_data_diff, label='Reference Stock Data (Processed)', color='blue')
axs[3].plot(range(len(ref_data_diff), len(ref_data_diff) + len(curr_data_diff)), curr_data_diff_with_slight_drift, label='Current Stock Data (Slight Drift)', color='green')
axs[3].plot(range(len(ref_data_diff), len(ref_data_diff) + len(curr_data_diff)), synthetic_curr_prediction_with_drift[:len(curr_data_diff)], label='Synthetic Prediction Data (With Drift)', color='red')
axs[3].plot(range(len(ref_data_diff)), synthetic_ref_prediction_with_drift_df[:len(ref_data_diff)], label='Synthetic Reference Data (With Drift)', color='orange')
axs[3].set_title('Drift Detection in Synthetic Data')
axs[3].legend()

# Trend Analysis
axs[4].plot(ref_data_with_trend, label='Reference Data with Trend', color='blue')
axs[4].plot(range(len(ref_data_with_trend), len(ref_data_with_trend) + len(curr_data_with_trend)), curr_data_with_trend, label='Current Data with Trend', color='green')
axs[4].plot(ref_data_with_reversed_trend, label='Reference Data with Reversed Trend', color='red')
axs[4].plot(range(len(ref_data_with_reversed_trend), len(ref_data_with_reversed_trend) + len(curr_data_with_reversed_trend)), curr_data_with_reversed_trend, label='Current Data with Reversed Trend', color='orange')
axs[4].plot(ref_data_with_trend_added_back, label='Reference Data with Trend Added Back', color='purple')
axs[4].plot(range(len(ref_data_with_trend_added_back), len(ref_data_with_trend_added_back) + len(curr_data_with_trend_added_back)), curr_data_with_trend_added_back, label='Current Data with Trend Added Back', color='brown')
axs[4].set_title('Trend Analysis')
axs[4].legend()

plt.tight_layout()
plt.show()

# Convert synthetic_ref_data and synthetic_curr_data to DataFrames for drift evaluation
synthetic_ref_data_df_full = pd.DataFrame({'value': synthetic_ref_data})
synthetic_curr_data_df_full = pd.DataFrame({'value': synthetic_curr_data})

# Evaluate drift using Evidently AI for data with drift in synthetic prediction but not in stock target
report_with_drift = evaluate_drift_evidently(synthetic_ref_prediction_with_drift_df[:len(ref_data_diff)], synthetic_curr_prediction_with_drift_df[:len(curr_data_diff)], synthetic_ref_data_df_full, synthetic_curr_data_df_full)
report_with_drift.save_html("evidently_report_with_drift.html")

print("\n--- Evidently AI Drift Report (With Drift) ---")
report_with_drift.show()