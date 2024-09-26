import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import yfinance as yf

# Load stock price data from Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
data = data[['Close']]

# Plot stock price
plt.figure(figsize=(10, 5))
plt.plot(data['Close'])
plt.title(f'{ticker} Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Function to create sequences of data for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Prepare the training and testing datasets
time_step = 60  # use the past 60 days to predict the next
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_step:]  # Include the last window from the training set

# Create train and test datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform([y_train])
y_test_actual = scaler.inverse_transform([y_test])

# Calculate and print the error
train_rmse = np.sqrt(mean_squared_error(y_train_actual[0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test_actual[0], test_predict[:, 0]))
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))

# Plot actual stock prices (Training set)
plt.plot(data.index[:train_size], scaler.inverse_transform(train_data), label='Actual Prices (Train)')

# Plot actual stock prices (Testing set)
plt.plot(data.index[train_size:], scaler.inverse_transform(test_data[time_step:]), label='Actual Prices (Test)')

# Plot predicted stock prices (Training set)
plt.plot(data.index[time_step:train_size], train_predict, label='Predicted Prices (Train)', color='green')

# Plot predicted stock prices (Testing set)
plt.plot(data.index[train_size:], test_predict, label='Predicted Prices (Test)', color='orange')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

### Concept Drift Detection using KS Test ###
# Define a sliding window for detecting drift
window_size = 200  # Use a window of last 200 days to compare
drift_threshold = 0.05  # Threshold for drift detection based on p-value

# Function to detect drift using KS test
def detect_concept_drift(old_data, new_data):
    # Kolmogorov-Smirnov test comparing distributions
    ks_stat, p_value = ks_2samp(old_data, new_data)
    return p_value < drift_threshold, p_value

# Sliding window to monitor concept drift over time
drift_detected_points = []
for i in range(window_size, len(scaled_data) - time_step):
    old_window = scaled_data[i - window_size:i, 0]
    new_window = scaled_data[i:i + window_size, 0]
    
    drift_detected, p_value = detect_concept_drift(old_window, new_window)
    
    if drift_detected:
        print(f"Concept drift detected at index {i}, p-value: {p_value}")
        drift_detected_points.append(i)

# Mark drift points on the graph
if drift_detected_points:
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Stock Prices')
    
    for point in drift_detected_points:
        plt.axvline(x=data.index[point], color='red', linestyle='--', label='Drift Point' if point == drift_detected_points[0] else "")
    
    plt.title(f'{ticker} Stock Prices with Detected Concept Drift')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
else:
    print("No concept drift detected.")

# Retrain the model (example trigger for retraining)
if drift_detected_points:
    print("Retraining the model on detected drift...")
    # You can retrain the model here using updated data after drift detection.
