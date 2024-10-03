import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from river.drift import PageHinkley  # For drift detection

# Define colors for plotting
GREY = '#808080'
RED = '#FF0000'
BLUE = '#0000FF'
GREEN = '#008000'
PURPLE = '#800080'

# Function to evaluate drift using statistical tests
def evaluate_drift(ref: pd.DataFrame, curr: pd.DataFrame):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="kl_div", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="jensenshannon", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="wasserstein", stattest_threshold=0.1)
    ])
    report.run(reference_data=ref, current_data=curr)
    results = report.as_dict()
  
    drift_report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, metric in enumerate(results['metrics']):
        stat_test_name = metric['result'].get('stattest_name', 'Unknown')
        drift_report.loc[i, 'stat_test'] = stat_test_name
        drift_report.loc[i, 'drift_score'] = metric['result']['drift_score']
        drift_report.loc[i, 'is_drifted'] = metric['result']['drift_detected']

    return drift_report

# Function to plot reference and current data with drift results
def plot_example(ref: pd.DataFrame, curr: pd.DataFrame, ref_name: str = "Reference", curr_name: str = "Current"):
    fig = plt.figure(constrained_layout=True, figsize=(15,7))
    gs = plt.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # Plot the actual data in time (split into segments for clarity)
    ref_points = int(np.round(150 * len(ref) / (len(ref) + len(curr))))
    curr_points = 150 - ref_points
    ref_in_time = [np.mean(x) for x in np.array_split(ref['value'], ref_points)]
    curr_in_time = [np.mean(x) for x in np.array_split(curr['value'], curr_points)]

    ax1.plot(range(ref_points), ref_in_time, color=GREY, label=ref_name, alpha=0.7)
    ax1.plot(range(ref_points, ref_points + curr_points), curr_in_time, color=RED, label=curr_name, alpha=0.7)
    ax1.legend()

    # Plot reference distribution
    sns.histplot(ref['value'], color=GREY, ax=ax2, alpha=0.7)
    ax2.set_title(f'{ref_name} Distribution')

    # Plot current distribution
    sns.histplot(curr['value'], color=RED, ax=ax3, alpha=0.7)
    ax3.set_title(f'{curr_name} Distribution')

    # Plot combined distributions
    sns.histplot(ref['value'], color=GREY, ax=ax4, label=ref_name, alpha=0.7)
    sns.histplot(curr['value'], color=RED, ax=ax4, label=curr_name, alpha=0.7)
    ax4.legend()
    plt.show()

# Function to detect and visualize data drift in LSTM model
def run_combined_drift_pipeline(window_size=60):
    # 1. Fetch Data
    data = yf.download('AAPL', start='2015-01-01', end='2023-01-01', interval="1d")
    data = data[['Close']].dropna()

    # 2. Preprocess Data
    scaler = MinMaxScaler()
    data['Scaled_Close'] = scaler.fit_transform(data[['Close']])
    data['LogReturn'] = np.log(data['Scaled_Close'] / data['Scaled_Close'].shift(1))
    data = data.dropna()

    # 3. Split Data into Reference (70%) and Current (30%)
    split_index = int(len(data) * 0.7)
    reference_data = data.iloc[:split_index]
    current_data = data.iloc[split_index:]

    # Prepare data for LSTM
    x_train, y_train = [], []
    for i in range(window_size, len(reference_data)):
        x_train.append(reference_data['Scaled_Close'].values[i-window_size:i])
        y_train.append(reference_data['Scaled_Close'].values[i])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 4. Train LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # 5. Make Predictions on Reference and Current Data
    # Reference (70%) Predictions
    ref_pred = []
    for i in range(window_size, len(reference_data)):
        ref_pred.append(model.predict(np.array([reference_data['Scaled_Close'].values[i-window_size:i]])))
    ref_pred = np.array(ref_pred).flatten()
    ref_pred = scaler.inverse_transform(ref_pred.reshape(-1, 1)).flatten()

    # Current (30%) Predictions
    curr_pred = []
    for i in range(window_size, len(current_data)):
        curr_pred.append(model.predict(np.array([current_data['Scaled_Close'].values[i-window_size:i]])))
    curr_pred = np.array(curr_pred).flatten()
    curr_pred = scaler.inverse_transform(curr_pred.reshape(-1, 1)).flatten()

    # Page-Hinkley drift detection (Prediction Drift)
    page_hinkley = PageHinkley()
    prediction_errors = np.abs(current_data['Close'].values[window_size:] - curr_pred)
    drift_dates = []
    for i, error in enumerate(prediction_errors):
        page_hinkley.update(error)
        if page_hinkley.drift_detected:
            drift_dates.append(current_data.index[window_size + i])

    # 6. Evaluate Data Drift
    reference_windows = pd.DataFrame(reference_data['LogReturn'].dropna().values, columns=['value'])
    current_windows = pd.DataFrame(current_data['LogReturn'].dropna().values, columns=['value'])
    drift_report = evaluate_drift(reference_windows, current_windows)
    print("Data Drift Report:", drift_report)

    # 7. Visualize
    plot_example(reference_windows, current_windows, ref_name="Reference", curr_name="Current")

    # Plot prediction drift detection
    plt.figure(figsize=(14, 7))
    combined_dates = np.concatenate((reference_data.index[window_size:], current_data.index[window_size:]))
    combined_actual_ref = reference_data['Close'].values[window_size:]
    combined_actual_curr = current_data['Close'].values[window_size:]
    combined_predicted_ref = np.concatenate((ref_pred, [np.nan] * len(curr_pred)))
    combined_predicted_curr = np.concatenate(([np.nan] * len(ref_pred), curr_pred))
    
    plt.plot(reference_data.index[window_size:], combined_actual_ref, label='Actual (Reference)', color=GREEN)
    plt.plot(current_data.index[window_size:], combined_actual_curr, label='Actual (Current)', color=GREY)
    plt.plot(combined_dates, combined_predicted_ref, label='Predicted (Reference)', color=BLUE)
    plt.plot(combined_dates, combined_predicted_curr, label='Predicted (Current)', color=RED)
    
    for drift_date in set(drift_dates):  # Use set to avoid duplicate labels
        plt.axvline(x=drift_date, color=PURPLE, linestyle='--', label='Page-Hinkley Drift')
    
    plt.title("Prediction Drift Detection using Page-Hinkley")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Run the combined pipeline
run_combined_drift_pipeline(window_size=60)