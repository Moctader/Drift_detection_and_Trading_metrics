# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Generate synthetic data
data = create_synthetic_data()

# Split data into reference (60%) and current (40%) sets
split_index = int(len(data) * 0.6)
ref_data = data[:split_index]
curr_data = data[split_index:]

# Introduce drift in the current prediction data by adding a linear trend
drift = np.linspace(0, 1, len(curr_data))
curr_prediction_with_drift = curr_data + drift

# Convert to DataFrames
ref_data_df = pd.DataFrame({'value': ref_data})
curr_data_df = pd.DataFrame({'value': curr_data})
curr_prediction_with_drift_df = pd.DataFrame({'value': curr_prediction_with_drift})

# Plot the synthetic data
plt.figure(figsize=(12, 6))
plt.plot(ref_data, label='Reference Data', color='blue')
plt.plot(range(split_index, len(data)), curr_data, label='Current Data (No Drift)', color='green')
plt.plot(range(split_index, len(data)), curr_prediction_with_drift, label='Current Prediction Data (With Drift)', color='red')
plt.title('Synthetic Time Series Data with and without Drift')
plt.legend()
plt.show()

# Evaluate drift using Evidently AI for data with drift in prediction but not in target
report_with_drift = evaluate_drift_evidently(ref_data_df, curr_prediction_with_drift_df, ref_data_df, curr_data_df)
report_with_drift.save_html("evidently_report_with_drift.html")

print("\n--- Evidently AI Drift Report (With Drift) ---")
report_with_drift.show()