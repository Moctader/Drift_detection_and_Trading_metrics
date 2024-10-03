import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from river.drift import PageHinkley  # Import Page-Hinkley drift detector

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

# Step 4: Split data into reference (70%) and current (30%) datasets
split_index = int(len(scaled_data) * 0.7)  # 70% for reference, 30% for current
ref_data = scaled_data[:split_index]
curr_data = scaled_data[split_index:]

# Prepare training data for the reference (70%) data
x_train, y_train = [], []
for i in range(sequence_length, len(ref_data)):
    x_train.append(ref_data[i-sequence_length:i, 0])
    y_train.append(ref_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

# Step 5: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10)

# Step 6: Make Predictions on the Reference and Current Data
# Reference (70%) Predictions
ref_pred = model.predict(x_train)
ref_pred = scaler.inverse_transform(ref_pred)  # Rescale to original prices

# Prepare test data for the current (30%) data
x_test, y_test = [], []
for i in range(sequence_length, len(curr_data)):
    x_test.append(curr_data[i-sequence_length:i, 0])
    y_test.append(curr_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM

# Current (30%) Predictions
curr_pred = model.predict(x_test)
curr_pred = scaler.inverse_transform(curr_pred)  # Rescale to original prices

# Step 7: Align the data with predictions for both Reference and Current
aligned_data_ref = data.iloc[sequence_length + 1:split_index + 1]  # Align reference data
aligned_data_curr = data.iloc[split_index + sequence_length + 1:]  # Align current data

# Add the differenced component back to the predicted prices for reference and current data
final_ref_predicted_prices = ref_pred.flatten() + data['Close'].shift(1).iloc[sequence_length + 1:split_index + 1].values
final_curr_predicted_prices = curr_pred.flatten() + data['Close'].shift(1).iloc[split_index + sequence_length + 1:].values

# Step 8: Page-Hinkley Drift Detection on Current Data
page_hinkley_detector = PageHinkley()

drift_results = {'Page-Hinkley': {'dates': [], 'values': []}}
dates = aligned_data_curr['Date'].values

# Detect drift in the current data
for idx, (actual, predicted) in enumerate(zip(aligned_data_curr['Close'].values, final_curr_predicted_prices)):
    error = abs(actual - predicted)
    
    # Update the Page-Hinkley detector with the prediction error
    page_hinkley_detector.update(error)
    
    # If drift is detected, store the date and predicted value
    if page_hinkley_detector.drift_detected:
        drift_results['Page-Hinkley']['dates'].append(dates[idx])
        drift_results['Page-Hinkley']['values'].append(predicted)

# Step 9: Combined Visualization for Reference and Current Data with Drift Detection

plt.figure(figsize=(14, 10))

# Plot the actual vs predicted prices for reference data (70%)
plt.subplot(2, 1, 1)
plt.plot(aligned_data_ref['Date'].values, aligned_data_ref['Close'].values, label='Actual Stock Price (Reference)', color='blue')
plt.plot(aligned_data_ref['Date'].values, final_ref_predicted_prices, label='Predicted Stock Price (Reference)', color='orange')
plt.title('Stock Price Predictions (Reference Data - 70%)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Plot the actual vs predicted prices for current data (30%) with drift detection
plt.subplot(2, 1, 2)
plt.plot(dates, aligned_data_curr['Close'].values, label='Actual Stock Price (Current)', color='green')
plt.plot(dates, final_curr_predicted_prices, label='Predicted Stock Price (Current)', color='red')

# Highlight the drift points detected by Page-Hinkley
if len(drift_results['Page-Hinkley']['dates']) > 0:
    for drift_date in drift_results['Page-Hinkley']['dates']:
        plt.axvline(x=drift_date, color='purple', linestyle='--', label='Page-Hinkley Drift')

plt.title('Stock Price Predictions with Page-Hinkley Drift Detection (Current Data - 30%)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Performance Evaluation
ref_mae = mean_absolute_error(aligned_data_ref['Close'].values, final_ref_predicted_prices)
curr_mae = mean_absolute_error(aligned_data_curr['Close'].values, final_curr_predicted_prices)
print(f"Reference MAE: {ref_mae}")
print(f"Current MAE: {curr_mae}")
