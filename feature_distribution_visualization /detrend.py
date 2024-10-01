import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from river.drift import PageHinkley, ADWIN, KSWIN
from sklearn.metrics import mean_absolute_error

# Step 1: Fetch Apple Stock Data Using yfinance
ticker = 'AAPL'
data = yf.download(ticker, start="2015-01-01", end="2023-01-01", interval="1d")
data = data[['Close']].dropna()
data.reset_index(inplace=True)

# Step 2: Decompose Time Series to Identify Trends and Seasonality
result = seasonal_decompose(data['Close'], model='multiplicative', period=365)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Detrend the data by removing the trend component
detrended_data = data['Close'] - trend

# Step 3: Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(detrended_data.dropna().values.reshape(-1, 1))

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

# Ensure the lengths of the original and predicted prices are the same
aligned_data = data.iloc[sequence_length + len(trend.dropna()) - len(predicted_prices):]  # Align the data with the predictions

# Add the trend component back to the predicted prices to get the final predictions
aligned_trend = trend.dropna().iloc[sequence_length + len(trend.dropna()) - len(predicted_prices):].values
final_predicted_prices = predicted_prices.flatten()[:len(aligned_trend)] + aligned_trend

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

# Step 6: Prepare data for Evidently
# Ensure the lengths of the arrays are the same
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

# Plot Detrended Reference Data
plt.figure(figsize=(14, 7))
plt.plot(reference_data['Date'], reference_data['Original'], label='Detrended Reference Original', color='blue')
plt.plot(reference_data['Date'], reference_data['Prediction'], label='Detrended Reference Prediction', color='orange')
plt.title('Detrended Reference Data: Original vs Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Plot Detrended Current Data
plt.figure(figsize=(14, 7))
plt.plot(current_data['Date'], current_data['Original'], label='Detrended Current Original', color='green')
plt.plot(current_data['Date'], current_data['Prediction'], label='Detrended Current Prediction', color='red')
plt.title('Detrended Current Data: Original vs Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 7: Set up the Evidently Report for target and prediction drift detection
report = Report(metrics=[
    ColumnDriftMetric(column_name="Original", stattest="ks", stattest_threshold=0.05),
    ColumnDriftMetric(column_name="Prediction", stattest="ks", stattest_threshold=0.05)
])

# Generate data drift report
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("concept_drift_report.html")

# Step 8: Detect and Visualize Drift
def detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, data, window_size, split_index):
    """
    Detect drift points using various drift detectors and visualize the results.

    Parameters:
    ref_actual (np.ndarray): Actual values for the reference period.
    ref_predicted (np.ndarray): Predicted values for the reference period.
    curr_actual (np.ndarray): Actual values for the current period.
    curr_predicted (np.ndarray): Predicted values for the current period.
    data (pd.DataFrame): Original dataset containing the dates.
    window_size (int): Size of the window used for predictions.
    split_index (int): Index to split the reference and current periods.

    Returns:
    None
    """
    # Calculate performance metrics
    ref_mae = mean_absolute_error(ref_actual, ref_predicted)
    curr_mae = mean_absolute_error(curr_actual, curr_predicted)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Calculate maximum deviation
    ref_max_deviation = np.max(np.abs(ref_actual - ref_predicted))
    print(f"Reference Max Deviation: {ref_max_deviation}")

    # Initialize Drift Detectors
    detectors = {
        "Page-Hinkley": PageHinkley(),
        "ADWIN": ADWIN(),
        "KSWIN": KSWIN(alpha=0.05)  # KS Test with a confidence level of 95%
    }

    # Detect drift points using each drift detector
    drift_results = {detector: {'dates': [], 'values': []} for detector in detectors}
    dates = data.index[window_size:]
    
    # Store points where current deviation exceeds reference max deviation
    exceed_deviation_dates = []
    exceed_deviation_values = []

    for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
        error = abs(actual - predicted)

        # Check if current deviation exceeds reference max deviation
        if error > ref_max_deviation:
            if split_index + idx < len(dates):
                exceed_deviation_dates.append(dates[split_index + idx])
                exceed_deviation_values.append(predicted)

        # Update each detector with the error and log drift points
        for detector_name, detector in detectors.items():
            detector.update(error)
            if detector.drift_detected:
                if split_index + idx < len(dates):
                    drift_results[detector_name]['dates'].append(dates[split_index + idx])
                    drift_results[detector_name]['values'].append(predicted)

    # Ensure the lengths of dates and actual/predicted values match
    ref_dates = dates[:split_index]
    curr_dates = dates[split_index:split_index + len(curr_actual)]

    # Adjust curr_actual and curr_predicted to match the length of curr_dates
    curr_actual = curr_actual[:len(curr_dates)]
    curr_predicted = curr_predicted[:len(curr_dates)]

    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, ref_actual, label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, ref_predicted, label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, curr_actual, label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, curr_predicted, label='Predicted Stock Price (Current)', color='red')

    # Plot points where current deviation exceeds reference max deviation
    if len(exceed_deviation_dates) > 0:
        plt.scatter(exceed_deviation_dates, exceed_deviation_values, color='black', label='Exceed Deviation', s=30)

    # Plot drift points for each detector
    colors = ['purple', 'magenta', 'cyan']
    for i, (detector_name, result) in enumerate(drift_results.items()):
        if len(result['dates']) > 0:
            plt.scatter(result['dates'], result['values'], color=colors[i], label=f'{detector_name} Drift', s=30)

    plt.title('Drift Detection in Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Example usage
ref_actual = reference_data['Original'].values
ref_predicted = reference_data['Prediction'].values
curr_actual = current_data['Original'].values
curr_predicted = current_data['Prediction'].values

detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, drift_data, window_size=60, split_index=split_index)