import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from river.drift import PageHinkley, ADWIN, KSWIN
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch Apple Stock Data Using yfinance
ticker = 'AAPL'
data = yf.download(ticker, start="2015-01-01", end="2023-01-01", interval="1d")
data = data[['Close']].dropna()
data.reset_index(inplace=True)

# Step 2: Difference the Data to Make it Stationary
data['Differenced'] = data['Close'].diff().dropna()

# Step 3: Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Differenced'].dropna().values.reshape(-1, 1))

sequence_length = 60  # We use the last 60 days to predict the next day
x_train, y_train = [], []
for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

# Step 4: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10)

# Step 5: Make Predictions
predicted_prices = model.predict(x_train)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Rescale to original prices

# Align the data with the predictions
aligned_data = data.iloc[sequence_length + 1:]  # Adjust for differencing and sequence length

# Add the differenced component back to the predicted prices to get the final predictions
final_predicted_prices = predicted_prices.flatten() + data['Close'].shift(1).iloc[sequence_length + 1:].values

# Ensure the lengths of aligned_data['Date'] and final_predicted_prices match
aligned_data = aligned_data.iloc[-len(final_predicted_prices):]

# Plot the predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], final_predicted_prices, label='Predicted Stock Price', color='red')
plt.title('LSTM Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 6: Prepare data for drift detection
aligned_dates = aligned_data['Date'].values[-len(final_predicted_prices):]
aligned_original = aligned_data['Close'].values[-len(final_predicted_prices):]

drift_data = pd.DataFrame({
    'Date': aligned_dates,
    'Original': aligned_original,
    'Prediction': final_predicted_prices
}).dropna()

# Split the data into reference and current datasets
split_index = len(drift_data) // 2
reference_data = drift_data[:split_index]
current_data = drift_data[split_index:]

# Plot reference and current data
plt.figure(figsize=(10, 6))
plt.plot(reference_data['Date'], reference_data['Original'], label='Reference Actual', color='blue')
plt.plot(reference_data['Date'], reference_data['Prediction'], label='Reference Predicted', color='orange')
plt.plot(current_data['Date'], current_data['Original'], label='Current Actual', color='green')
plt.plot(current_data['Date'], current_data['Prediction'], label='Current Predicted', color='red')
plt.title('Reference and Current Data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 7: Define performance metrics
def calculate_metrics(detected_drifts, actual_drifts, total_points):
    detection_rate = len(detected_drifts & actual_drifts) / len(actual_drifts)
    false_negative_rate = len(actual_drifts - detected_drifts) / len(actual_drifts)
    false_positive_rate = len(detected_drifts - actual_drifts) / (total_points - len(actual_drifts))
    detection_delay = np.mean([min(abs(d - a) for a in actual_drifts) for d in detected_drifts])
    return detection_rate, false_negative_rate, false_positive_rate, detection_delay

# Step 8: Evaluate algorithms with different configurations
def evaluate_drift_detection(ref_actual, ref_predicted, curr_actual, curr_predicted, data, window_size, split_index):
    detectors = {
        "Page-Hinkley": [PageHinkley()],
        "ADWIN": [ADWIN(delta=d) for d in [0.001, 0.01, 0.1]],
        "KSWIN": [KSWIN(alpha=a) for a in [0.01, 0.05, 0.1]]
    }

    best_configurations = {}
    total_points = len(curr_actual)

    for detector_name, configs in detectors.items():
        best_metrics = None
        best_config = None

        for config in configs:
            detected_drifts = set()
            for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
                error = abs(actual - predicted)
                config.update(error)
                if config.drift_detected:
                    detected_drifts.add(split_index + idx)

            metrics = calculate_metrics(detected_drifts, detected_drifts, total_points)
            if best_metrics is None or metrics < best_metrics:
                best_metrics = metrics
                best_config = config

        best_configurations[detector_name] = (best_config, best_metrics)

    return best_configurations
# Step 9: Plot drift points for each detector (updated)
def plot_drift_points(drift_results, data, window_size, split_index, title):
    ref_dates = data['Date'][:split_index]
    curr_dates = data['Date'][split_index:]

    plt.figure(figsize=(12, 6))
    
    # Plot reference data
    plt.plot(ref_dates, data['Original'][:split_index], label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, data['Prediction'][:split_index], label='Predicted Stock Price (Reference)', color='orange')
    
    # Plot current data
    plt.plot(curr_dates, data['Original'][split_index:], label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, data['Prediction'][split_index:], label='Predicted Stock Price (Current)', color='red')

    colors = ['purple', 'magenta', 'cyan']
    for i, (detector_name, result) in enumerate(drift_results.items()):
        if len(result['dates']) > 0:
            plt.scatter(result['dates'], result['values'], color=colors[i], label=f'{detector_name} Drift', s=30)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Example usage
ref_actual = reference_data['Original'].values
ref_predicted = reference_data['Prediction'].values
curr_actual = current_data['Original'].values
curr_predicted = current_data['Prediction'].values

# Detect drift points using each drift detector
detectors = {
    "Page-Hinkley": PageHinkley(),
    "ADWIN": ADWIN(delta=0.01),
    "KSWIN": KSWIN(alpha=0.05)
}

drift_results = {detector: {'dates': [], 'values': []} for detector in detectors}
dates = current_data['Date'].values

for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
    error = abs(actual - predicted)

    # Update each detector with the error and log drift points
    for detector_name, detector in detectors.items():
        detector.update(error)
        if detector.drift_detected:
            if split_index + idx < len(dates):
                drift_results[detector_name]['dates'].append(dates[idx])
                drift_results[detector_name]['values'].append(predicted)

# Evaluate drift detection
best_configurations = evaluate_drift_detection(ref_actual, ref_predicted, curr_actual, curr_predicted, drift_data, window_size=60, split_index=split_index)

# Plot drift points for the best configuration
best_drift_results = {detector_name: {'dates': [], 'values': []} for detector_name in best_configurations.keys()}
for detector_name, (config, _) in best_configurations.items():
    for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
        error = abs(actual - predicted)
        config.update(error)
        if config.drift_detected:
            best_drift_results[detector_name]['dates'].append(current_data['Date'].iloc[idx])
            best_drift_results[detector_name]['values'].append(predicted)

plot_drift_points(best_drift_results, drift_data, window_size=60, split_index=split_index, title='Drift Detection in Stock Price Predictions (Best Configuration)')






