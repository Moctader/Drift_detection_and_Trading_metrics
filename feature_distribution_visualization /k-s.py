import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

# Function to create dataset with sliding window
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# LSTM model creation
def build_and_train_lstm(X, y, epochs, validation_data=None):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    model.fit(X, y, epochs=epochs, batch_size=64, validation_data=validation_data, 
              callbacks=[early_stop, reduce_lr], verbose=1)
    
    return model

# Generate synthetic data with known drift points
def generate_synthetic_data(segment_length=100, num_segments=11, drift=True):
    if drift:
        data_segments = [np.random.normal(loc=50 + 50 * i, scale=5, size=segment_length) for i in range(num_segments)]
    else:
        data_segments = [np.random.normal(loc=50, scale=5, size=segment_length) for _ in range(num_segments)]
    drift_data = np.concatenate(data_segments)
    drift_points = [segment_length * i for i in range(1, num_segments)] if drift else []
    return drift_data, drift_points

# Pipeline to compare drift detection using K-S test
def run_comparison_pipeline(window_size, epochs, drift=True):
    # Generate synthetic data
    drift_data, known_drift_points = generate_synthetic_data(drift=drift)
    
    # Preprocess synthetic data
    drift_data = drift_data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(drift_data)

    # Split data into reference (70%) and current (30%) sets
    split_index = int(len(scaled_data) * 0.7)
    ref_data = scaled_data[:split_index]
    curr_data = scaled_data[split_index - window_size:]  # Include window_size overlap

    # Create LSTM datasets
    X_ref, y_ref = create_dataset(ref_data, window_size)
    X_curr, y_curr = create_dataset(curr_data, window_size)

    # Train and predict using LSTM
    model = build_and_train_lstm(X_ref, y_ref, epochs)
    predicted_ref = model.predict(X_ref)
    predicted_curr = model.predict(X_curr)
    
    predicted_ref = scaler.inverse_transform(predicted_ref)
    predicted_curr = scaler.inverse_transform(predicted_curr)
    
    actual_ref = scaler.inverse_transform(y_ref.reshape(-1, 1)).flatten()
    actual_curr = scaler.inverse_transform(y_curr.reshape(-1, 1)).flatten()
    predicted_ref = predicted_ref.flatten()
    predicted_curr = predicted_curr.flatten()

    # Calculate performance metrics
    ref_mae = mean_absolute_error(actual_ref, predicted_ref)
    curr_mae = mean_absolute_error(actual_curr, predicted_curr)
    print(f"Reference MAE: {ref_mae}, Current MAE: {curr_mae}")

    # Calculate maximum deviation
    ref_max_deviation = np.max(np.abs(actual_ref - predicted_ref))
    print(f"Reference Max Deviation: {ref_max_deviation}")

    # Drift detection using K-S Test
    drift_results = {'K-S Test': {'dates': [], 'values': [], 'types': []}}
    dates = np.arange(len(drift_data))[window_size:]  # Adjust the dates to account for the window size
    ref_dates = dates[:len(actual_ref)]  # Reference dates must match the length of actual_ref
    curr_dates = dates[len(actual_ref):]  # Current dates must match the length of actual_curr

    buffer_zone = 5  # Buffer zone around known drift points
    false_alarms = 0

    for idx in range(window_size, len(actual_curr)):
        ref_segment = actual_ref[idx-window_size:idx]
        curr_segment = actual_curr[idx-window_size:idx]
        
        ks_stat, p_value = ks_2samp(ref_segment, curr_segment)
        
        if p_value < 0.05:  # Drift detected
            is_true_detection = any(abs(curr_dates[idx] - drift_point) <= buffer_zone for drift_point in known_drift_points)
            if is_true_detection:
                drift_results['K-S Test']['dates'].append(curr_dates[idx])
                drift_results['K-S Test']['values'].append(predicted_curr[idx])
                drift_results['K-S Test']['types'].append('true_detection')
            else:
                false_alarms += 1
                drift_results['K-S Test']['dates'].append(curr_dates[idx])
                drift_results['K-S Test']['values'].append(predicted_curr[idx])
                drift_results['K-S Test']['types'].append('false_alarm')

    print(f'Total Detected Drifts: {len(drift_results["K-S Test"]["dates"])}')
    print(f'False Alarms: {false_alarms}')

    return drift_data, ref_dates, actual_ref, predicted_ref, curr_dates, actual_curr, predicted_curr, drift_results, known_drift_points

if __name__ == "__main__":
    window_size = 60
    epochs = 100

    # Run pipeline with concept drift
    print("Running pipeline with concept drift:")
    drift_data_with, ref_dates_with, actual_ref_with, predicted_ref_with, curr_dates_with, actual_curr_with, predicted_curr_with, drift_results_with, known_drift_points_with = run_comparison_pipeline(window_size, epochs, drift=True)

    # Run pipeline without concept drift
    print("\nRunning pipeline without concept drift:")
    drift_data_without, ref_dates_without, actual_ref_without, predicted_ref_without, curr_dates_without, actual_curr_without, predicted_curr_without, drift_results_without, known_drift_points_without = run_comparison_pipeline(window_size, epochs, drift=False)

    # Plotting Drift Points
    plt.figure(figsize=(12, 6))
    plt.plot(drift_data_with, label='Synthetic Data with Drift')
    for drift_point, drift_type in zip(drift_results_with['K-S Test']['dates'], drift_results_with['K-S Test']['types']):
        if drift_type == 'true_detection':
            plt.scatter(drift_point, drift_data_with[drift_point], color='blue', label='True Detection' if drift_point == drift_results_with['K-S Test']['dates'][0] else "")
        else:
            plt.scatter(drift_point, drift_data_with[drift_point], color='red', label='False Alarm' if drift_point == drift_results_with['K-S Test']['dates'][0] else "")
    for known_drift_point in known_drift_points_with:
        plt.axvline(known_drift_point, color='green', linestyle='-', label='Known Drift Point' if known_drift_point == known_drift_points_with[0] else "")
    plt.legend()
    plt.title('Synthetic Data with Drift')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(drift_data_without, label='Synthetic Data without Drift')
    for drift_point, drift_type in zip(drift_results_without['K-S Test']['dates'], drift_results_without['K-S Test']['types']):
        if drift_type == 'true_detection':
            plt.scatter(drift_point, drift_data_without[drift_point], color='blue', label='True Detection' if drift_point == drift_results_without['K-S Test']['dates'][0] else "")
        else:
            plt.scatter(drift_point, drift_data_without[drift_point], color='red', label='False Alarm' if drift_point == drift_results_without['K-S Test']['dates'][0] else "")
    for known_drift_point in known_drift_points_without:
        plt.axvline(known_drift_point, color='green', linestyle='-', label='Known Drift Point' if known_drift_point == known_drift_points_without[0] else "")
    plt.legend()
    plt.title('Synthetic Data without Drift')
    plt.show()

    # Plot reference and current segments for both scenarios
    plt.figure(figsize=(12, 6))
    plt.plot(ref_dates_with, actual_ref_with, label='Actual Stock Price (Reference) with Drift', color='blue')
    plt.plot(ref_dates_with, predicted_ref_with, label='Predicted Stock Price (Reference) with Drift', color='orange')
    plt.plot(curr_dates_with, actual_curr_with, label='Actual Stock Price (Current) with Drift', color='green')
    plt.plot(curr_dates_with, predicted_curr_with, label='Predicted Stock Price (Current) with Drift', color='red')

    plt.plot(ref_dates_without, actual_ref_without, label='Actual Stock Price (Reference) without Drift', color='cyan')
    plt.plot(ref_dates_without, predicted_ref_without, label='Predicted Stock Price (Reference) without Drift', color='magenta')
    plt.plot(curr_dates_without, actual_curr_without, label='Actual Stock Price (Current) without Drift', color='yellow')
    plt.plot(curr_dates_without, predicted_curr_without, label='Predicted Stock Price (Current) without Drift', color='black')

    # Mark detected drift points for the scenario with drift
    for drift_date, drift_value, drift_type in zip(drift_results_with['K-S Test']['dates'], drift_results_with['K-S Test']['values'], drift_results_with['K-S Test']['types']):
        if drift_type == 'true_detection':
            plt.scatter(drift_date, drift_value, color='blue', marker='x', s=100, label='Detected Drift (with Drift)' if drift_date == drift_results_with['K-S Test']['dates'][0] else "")
        # else:
        #     plt.scatter(drift_date, drift_value, color='red', marker='x', s=100, label='False Alarm (with Drift)' if drift_date == drift_results_with['K-S Test']['dates'][0] else "")

    # Mark detected drift points for the scenario without drift
    for drift_date, drift_value, drift_type in zip(drift_results_without['K-S Test']['dates'], drift_results_without['K-S Test']['values'], drift_results_without['K-S Test']['types']):
        if drift_type == 'true_detection':
            plt.scatter(drift_date, drift_value, color='cyan', marker='o', s=100, label='Detected Drift (without Drift)' if drift_date == drift_results_without['K-S Test']['dates'][0] else "")
        # else:
        #     plt.scatter(drift_date, drift_value, color='magenta', marker='o', s=100, label='False Alarm (without Drift)' if drift_date == drift_results_without['K-S Test']['dates'][0] else "")

    plt.title('Drift Detection using K-S Test')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()