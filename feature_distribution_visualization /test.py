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

# Align the lengths of the original and predicted prices
# Start from sequence_length to align with the model's input
predicted_prices_length = len(predicted_prices)
actual_length = len(data) - sequence_length

# Ensure the lengths match by slicing the original data
aligned_data = data.iloc[sequence_length:].reset_index(drop=True)

# Ensure trend and predictions are properly aligned
trend_aligned = trend.iloc[sequence_length:actual_length + sequence_length].dropna().values.flatten()

# Ensure that both predicted_prices and trend_aligned have the same length
if len(predicted_prices) == len(trend_aligned):
    final_predicted_prices = predicted_prices.flatten() + trend_aligned
else:
    raise ValueError(f"Length mismatch: predicted_prices {len(predicted_prices)} vs trend_aligned {len(trend_aligned)}")

# Plot the predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'].values, label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], final_predicted_prices, label='Predicted Stock Price', color='red')
plt.title('LSTM Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 6: Reverse the Trend for Drift Comparison
reversed_trend = -trend + 2 * np.mean(trend)

# Ensure seasonal and residual are aligned
seasonal_aligned = seasonal.iloc[sequence_length:actual_length + sequence_length].values
residual_aligned = residual.iloc[sequence_length:actual_length + sequence_length].values

# Reconstruct the stock price with reversed trend
reversed_stock_price = reversed_trend.iloc[sequence_length:actual_length + sequence_length].values * seasonal_aligned * residual_aligned

# Plot the reversed trend stock price vs actual stock price
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], reversed_stock_price, label='Reversed Trend Stock Price', color='red')
plt.title('Apple Stock Price with Reversed Trend')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 7: Reconstruct the Original Stock Price
reconstructed_stock_price = trend * seasonal * residual

# Plot the reconstructed stock price vs actual stock price
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], reconstructed_stock_price.iloc[sequence_length:actual_length + sequence_length].values, label='Reconstructed Stock Price', color='green')
plt.title('Apple Stock Price Reconstruction (Restoring Trend)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 8: Concept Drift Detection using Evidently
# Prepare data for Evidently
drift_data = pd.DataFrame({
    'Date': aligned_data['Date'],
    'Original': aligned_data['Close'].values,
    'Prediction': final_predicted_prices
}).dropna()

# Split the data into reference and current datasets
split_index = len(drift_data) // 2
reference_data = drift_data[:split_index]
current_data = drift_data[split_index:]

# Set up the Evidently Report for target and prediction drift detection
report = Report(metrics=[
    ColumnDriftMetric(column_name="Original", stattest="ks", stattest_threshold=0.05),
    ColumnDriftMetric(column_name="Prediction", stattest="ks", stattest_threshold=0.05)
])

# Generate data drift report
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("concept_drift_report.html")

# Optionally, display the report in a Jupyter Notebook (if applicable)
# report.show()
