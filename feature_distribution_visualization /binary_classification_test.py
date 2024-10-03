import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from river.drift import PageHinkley
import matplotlib.pyplot as plt

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

# Step 6: Make Predictions on the Current (30%) Data
x_test, y_test = [], []
for i in range(sequence_length, len(curr_data)):
    x_test.append(curr_data[i-sequence_length:i, 0])
    y_test.append(curr_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM

# Predict on the current data
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Rescale to original prices

# Step 7: Align the data with predictions
aligned_data = data.iloc[split_index + sequence_length + 1:]  # Adjust for differencing and sequence length

# Add the differenced component back to the predicted prices to get the final predictions
final_predicted_prices = predicted_prices.flatten() + data['Close'].shift(1).iloc[split_index + sequence_length + 1:split_index + sequence_length + 1 + len(predicted_prices)].values

# Ensure the lengths of aligned_data['Date'] and final_predicted_prices match
aligned_data = aligned_data.iloc[-len(final_predicted_prices):]
aligned_dates = aligned_data['Date'].values[-len(final_predicted_prices):]
aligned_original = aligned_data['Close'].values[-len(final_predicted_prices):]

# Create a dataframe with the original and predicted values
drift_data = pd.DataFrame({
    'Date': aligned_dates,
    'Original': aligned_original,
    'Prediction': final_predicted_prices
}).dropna()

# Step 8: Performance Evaluation
mae = mean_absolute_error(aligned_original, final_predicted_prices)
print(f"Mean Absolute Error on the current data: {mae}")

# Optionally, plot the results
plt.figure(figsize=(12, 6))

# Plot reference data
aligned_train_data = data.iloc[sequence_length + 1:split_index + sequence_length + 1]  # Adjust for differencing and sequence length
final_train_predictions = model.predict(x_train)
final_train_predictions = scaler.inverse_transform(final_train_predictions).flatten() + data['Close'].shift(1).iloc[sequence_length + 1:sequence_length + 1 + len(final_train_predictions)].values

plt.plot(aligned_train_data['Date'], aligned_train_data['Close'], label='Actual Stock Price (Reference)', color='blue')
plt.plot(aligned_train_data['Date'], final_train_predictions, label='Predicted Stock Price (Reference)', color='orange')

# Plot current data
plt.plot(aligned_dates, aligned_original, label='Actual Stock Price (Current)', color='green')
plt.plot(aligned_dates, final_predicted_prices, label='Predicted Stock Price (Current)', color='red')

plt.title('LSTM Predictions on Reference and Current Data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 9: Detect and Visualize Drift using Page-Hinkley
def detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, data, window_size, split_index):
    # Calculate performance metrics
    ref_mae = mean_absolute_error(ref_actual, ref_predicted)
    curr_mae = mean_absolute_error(curr_actual, curr_predicted)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Calculate maximum deviation
    ref_max_deviation = np.max(np.abs(ref_actual - ref_predicted))
    print(f"Reference Max Deviation: {ref_max_deviation}")

    # Initialize Page-Hinkley Drift Detector
    page_hinkley_detector = PageHinkley()
    drift_results = {'Page-Hinkley': {'dates': [], 'values': []}}
    dates = data.index[window_size:]
    
    for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
        error = abs(actual - predicted)

        # Update the detector with the error and log drift points
        page_hinkley_detector.update(error)
        if page_hinkley_detector.drift_detected:
            if split_index + idx < len(dates):
                drift_results['Page-Hinkley']['dates'].append(dates[split_index + idx])
                drift_results['Page-Hinkley']['values'].append(predicted)

    # Ensure the lengths of dates and actual/predicted values match
    ref_dates = dates[:split_index]
    curr_dates = dates[split_index:split_index + len(curr_actual)]

    # Adjust curr_actual and curr_predicted to match the length of curr_dates
    curr_actual = curr_actual[:len(curr_dates)]
    curr_predicted = curr_predicted[:len(curr_dates)]

    # Ensure lengths match for plotting
    min_length = min(len(ref_dates), len(ref_actual), len(ref_predicted))
    ref_dates = ref_dates[:min_length]
    ref_actual = ref_actual[:min_length]
    ref_predicted = ref_predicted[:min_length]

    min_length = min(len(curr_dates), len(curr_actual), len(curr_predicted))
    curr_dates = curr_dates[:min_length]
    curr_actual = curr_actual[:min_length]
    curr_predicted = curr_predicted[:min_length]

    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates, ref_actual, label='Actual Stock Price (Reference)', color='blue')
    plt.plot(ref_dates, ref_predicted, label='Predicted Stock Price (Reference)', color='orange')
    plt.plot(curr_dates, curr_actual, label='Actual Stock Price (Current)', color='green')
    plt.plot(curr_dates, curr_predicted, label='Predicted Stock Price (Current)', color='red')

    # Plot drift points for Page-Hinkley
    if len(drift_results['Page-Hinkley']['dates']) > 0:
        for drift_date in drift_results['Page-Hinkley']['dates']:
            plt.axvline(x=drift_date, color='purple', linestyle='--', label='Page-Hinkley Drift')

    plt.title('Drift Detection in Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Calculate binary classification metrics
    threshold = 0  # Define a threshold for binary classification
    ref_actual_binary = (np.diff(ref_actual) > threshold).astype(int)
    ref_predicted_binary = (np.diff(ref_predicted) > threshold).astype(int)
    curr_actual_binary = (np.diff(curr_actual) > threshold).astype(int)
    curr_predicted_binary = (np.diff(curr_predicted) > threshold).astype(int)

    ref_accuracy = accuracy_score(ref_actual_binary, ref_predicted_binary)
    ref_precision = precision_score(ref_actual_binary, ref_predicted_binary, zero_division=0)
    ref_recall = recall_score(ref_actual_binary, ref_predicted_binary, zero_division=0)
    ref_f1 = f1_score(ref_actual_binary, ref_predicted_binary, zero_division=0)

    curr_accuracy = accuracy_score(curr_actual_binary, curr_predicted_binary)
    curr_precision = precision_score(curr_actual_binary, curr_predicted_binary, zero_division=0)
    curr_recall = recall_score(curr_actual_binary, curr_predicted_binary, zero_division=0)
    curr_f1 = f1_score(curr_actual_binary, curr_predicted_binary, zero_division=0)

    print(f"Reference Accuracy: {ref_accuracy}, Precision: {ref_precision}, Recall: {ref_recall}, F1-Score: {ref_f1}")
    print(f"Current Accuracy: {curr_accuracy}, Precision: {curr_precision}, Recall: {curr_recall}, F1-Score: {curr_f1}")

    # Calculate confusion matrix for reference data
    ref_cm = confusion_matrix(ref_actual_binary, ref_predicted_binary)
    ref_tp = ref_cm[1, 1]
    ref_fp = ref_cm[0, 1]
    ref_tn = ref_cm[0, 0]
    ref_fn = ref_cm[1, 0]

    print(f"Reference Confusion Matrix:\n{ref_cm}")
    print(f"Reference TP: {ref_tp}, FP: {ref_fp}, TN: {ref_tn}, FN: {ref_fn}")

    # Calculate confusion matrix for current data
    curr_cm = confusion_matrix(curr_actual_binary, curr_predicted_binary)
    if curr_cm.shape == (2, 2):
        curr_tp = curr_cm[1, 1]
        curr_fp = curr_cm[0, 1]
        curr_tn = curr_cm[0, 0]
        curr_fn = curr_cm[1, 0]
    else:
        curr_tp = curr_fp = curr_tn = curr_fn = 0

    print(f"Current Confusion Matrix:\n{curr_cm}")
    print(f"Current TP: {curr_tp}, FP: {curr_fp}, TN: {curr_tn}, FN: {curr_fn}")

ref_actual = aligned_train_data['Close'].values
ref_predicted = final_train_predictions
curr_actual = aligned_original
curr_predicted = final_predicted_prices

detect_and_visualize_drift(ref_actual, ref_predicted, curr_actual, curr_predicted, drift_data, window_size=60, split_index=len(ref_actual))