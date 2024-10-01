import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from river import drift
from scipy.stats import ks_2samp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Step 1: Generate synthetic stock price data
np.random.seed(42)

# Parameters
initial_length = 100  # Length of the initial segment
abrupt_length = 50     # Length of the abrupt drift segment
gradual_length = 100   # Length of the gradual drift segment
total_length = initial_length + abrupt_length + gradual_length

# Generate initial stock prices (normal market behavior)
initial_prices = np.random.normal(loc=100, scale=5, size=initial_length)

# Generate abrupt drift (sudden price increase)
abrupt_drift = np.random.normal(loc=150, scale=5, size=abrupt_length)

# Generate gradual drift (gradually increasing prices)
gradual_drift = np.linspace(150, 200, gradual_length) + np.random.normal(loc=0, scale=2, size=gradual_length)

# Combine all segments to form the synthetic data
synthetic_prices = np.concatenate([initial_prices, abrupt_drift, gradual_drift])

# Create a DataFrame
df = pd.DataFrame(synthetic_prices, columns=['Close'])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Prepare the data for LSTM
def create_dataset(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(scaled_data, look_back)

# Split the data into initial training and streaming data
train_size = int(len(X) * 0.2)
X_train, X_stream = X[:train_size], X[train_size:]
y_train, y_stream = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
X_stream = np.reshape(X_stream, (X_stream.shape[0], look_back, 1))

# Train an initial LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=2)

# Step 2: Initialize the ADWIN, Page-Hinkley, and KS test drift detectors
adwin_detector = drift.ADWIN(delta=0.05)  # Adjust sensitivity with delta
page_hinkley_detector = drift.PageHinkley()
adwin_drifts = []
page_hinkley_drifts = []
ks_drifts = []

# Parameters for KS test
window_size = 50  # Size of the window for KS test

# Step 3: Process the synthetic data stream
predictions = []
for i in range(window_size, len(X_stream)):
    X_t = X_stream[i].reshape(1, look_back, 1)
    y_t = y_stream[i]
    
    # Predict and calculate the error
    y_pred = model.predict(X_t)
    predictions.append(y_pred[0][0])
    
    # Update the ADWIN detector with the current value
    adwin_detector.update(y_pred[0][0])
    
    # Update the Page-Hinkley detector with the current value
    page_hinkley_detector.update(y_pred[0][0])
    
    # Perform KS test
    if i >= 2 * window_size:
        window1 = y_stream[i-window_size:i]
        window2 = y_stream[i-window_size*2:i-window_size]
        ks_stat, ks_p_value = ks_2samp(window1, window2)
        
        # Check if a drift is detected by KS test
        if ks_p_value < 0.05:  # Using a significance level of 0.05
            print(f'KS test detected change at index {i}')  # Mark the index as a drift point
            ks_drifts.append(i)
    
    # Check if a drift is detected by ADWIN
    if adwin_detector.drift_detected:
        print(f'ADWIN detected change at index {i}')  # Mark the index as a drift point
        adwin_drifts.append(i)
        adwin_detector = drift.ADWIN(delta=0.05)  # Reinitialize the detector

    # Check if a drift is detected by Page-Hinkley
    if page_hinkley_detector.drift_detected:
        print(f'Page-Hinkley detected change at index {i}')  # Mark the index as a drift point
        page_hinkley_drifts.append(i)
        page_hinkley_detector = drift.PageHinkley()  # Reinitialize the detector

# Step 4: Auxiliary function to plot the data
def plot_data(synthetic_prices, predictions, drifts=None, title='Synthetic Stock Price with Detected Drifts'):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    ax1.grid()
    ax1.plot(synthetic_prices, label='Synthetic Close Price', color='blue')
    ax1.plot(range(len(predictions)), predictions, label='Predictions', linestyle='dashed')
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red', linestyle='--', label='Detected Drift' if drift_detected == drifts[0] else "")
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Step 5: Plot the synthetic data and detected drifts for ADWIN
plot_data(df['Close'].values[window_size:], predictions, adwin_drifts, title='Synthetic Stock Price with ADWIN Detected Drifts')

# Step 6: Plot the synthetic data and detected drifts for Page-Hinkley
plot_data(df['Close'].values[window_size:], predictions, page_hinkley_drifts, title='Synthetic Stock Price with Page-Hinkley Detected Drifts')

# Step 7: Plot the synthetic data and detected drifts for KS test
plot_data(df['Close'].values[window_size:], predictions, ks_drifts, title='Synthetic Stock Price with KS Test Detected Drifts')