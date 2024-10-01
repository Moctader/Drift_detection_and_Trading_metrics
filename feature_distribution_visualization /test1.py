import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

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

# Step 6: Reverse the Trend for Drift Comparison
reversed_trend = -trend + 2 * np.mean(trend)

# Reconstruct the stock price with reversed trend, keeping seasonality and residual intact
reversed_stock_price = reversed_trend * seasonal * residual

# Ensure the lengths of aligned_data['Date'] and reversed_stock_price match
aligned_reversed_stock_price = reversed_stock_price.iloc[sequence_length + len(trend.dropna()) - len(predicted_prices):]
aligned_reversed_stock_price = aligned_reversed_stock_price.iloc[-len(final_predicted_prices):]

# Plot the reversed trend stock price vs actual stock price
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], aligned_reversed_stock_price.values, label='Reversed Trend Stock Price', color='red')
plt.title('Apple Stock Price with Reversed Trend')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 7: Reconstruct the Original Stock Price
reconstructed_stock_price = trend * seasonal * residual

# Ensure the lengths of aligned_data['Date'] and reconstructed_stock_price match
aligned_reconstructed_stock_price = reconstructed_stock_price.iloc[sequence_length + len(trend.dropna()) - len(predicted_prices):]
aligned_reconstructed_stock_price = aligned_reconstructed_stock_price.iloc[-len(final_predicted_prices):]

# Plot the reconstructed stock price vs actual stock price
plt.figure(figsize=(10, 6))
plt.plot(aligned_data['Date'], aligned_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(aligned_data['Date'], aligned_reconstructed_stock_price.values, label='Reconstructed Stock Price', color='green')
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 9: Monitor Model Quality Metrics
def monitor_model_quality(y_true, y_pred):
    """
    Monitor and print model quality metrics for regression tasks, and check for significant drops.

    Parameters:
    y_true (list or np.ndarray): Actual values.
    y_pred (list or np.ndarray): Predicted values.

    Returns:
    None
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared (R2): {r2:.4f}')

    # Define thresholds for significant drops
    mse_threshold = 0.05
    mae_threshold = 0.05
    r2_threshold = 0.5  # R2 should be high, so a low threshold indicates a significant drop

    # Check for significant drops
    if mse > mse_threshold:
        print("Warning: Significant increase in MSE detected.")
    if mae > mae_threshold:
        print("Warning: Significant increase in MAE detected.")
    if r2 < r2_threshold:
        print("Warning: Significant drop in R2 detected.")

# Calculate y_true and y_pred for this project
y_true = aligned_data['Close'].values
y_pred = final_predicted_prices

# Monitor model quality metrics
monitor_model_quality(y_true, y_pred)


# Step 10: Use Proxy Metrics
def observe_proxy_metrics(predictions):
    # Calculate the distribution of predictions
    prediction_distribution = pd.Series(predictions).value_counts(normalize=True)
    print("Prediction Distribution:")
    print(prediction_distribution)

    # Plot the distribution of predictions
    plt.figure(figsize=(10, 6))
    prediction_distribution.plot(kind='bar')
    plt.title('Distribution of Predictions')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.show()

# Example usage of proxy metrics
observe_proxy_metrics(final_predicted_prices)

# Step 11: Statistical Tests
def apply_statistical_tests(reference_data, current_data):
    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(reference_data['Prediction'], current_data['Prediction'])
    print(f"Kolmogorov-Smirnov test statistic: {ks_stat}, p-value: {ks_p_value}")

    # Bin the predictions into categories for Chi-Square test
    bins = np.linspace(min(reference_data['Prediction'].min(), current_data['Prediction'].min()),
                       max(reference_data['Prediction'].max(), current_data['Prediction'].max()), 10)
    reference_binned = np.digitize(reference_data['Prediction'], bins)
    current_binned = np.digitize(current_data['Prediction'], bins)

    # Create a contingency table
    contingency_table = pd.crosstab(reference_binned, current_binned)
    if contingency_table.size == 0:
        print("No overlapping values for Chi-Square test.")
    else:
        chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-Square test statistic: {chi2_stat}, p-value: {chi2_p_value}")

# Example usage of statistical tests
apply_statistical_tests(reference_data, current_data)

# Step 12: Distance Metrics
def measure_distance_metrics(reference_data, current_data):
    # Wasserstein Distance
    wasserstein_dist = wasserstein_distance(reference_data['Prediction'], current_data['Prediction'])
    print(f"Wasserstein Distance: {wasserstein_dist}")

    # Population Stability Index (PSI)
    def calculate_psi(expected, actual, buckets=10, epsilon=1e-10):
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        expected_percents = np.percentile(expected, breakpoints)
        actual_percents = np.percentile(actual, breakpoints)

        expected_percents = scale_range(expected_percents, 0, 1)
        actual_percents = scale_range(actual_percents, 0, 1)

        # Add epsilon to avoid division by zero
        psi_value = np.sum((expected_percents - actual_percents) * np.log((expected_percents + epsilon) / (actual_percents + epsilon)))
        return psi_value

    psi_value = calculate_psi(reference_data['Prediction'], current_data['Prediction'])
    print(f"Population Stability Index (PSI): {psi_value}")

# Example usage of distance metrics
measure_distance_metrics(reference_data, current_data)


#step 13
# Example usage of distance metrics
measure_distance_metrics(reference_data, current_data)

# Step 13: Rule-Based Checks
def rule_based_checks(predictions, threshold=0.1):
    # Calculate the percentage of predictions within a specific class
    prediction_distribution = pd.Series(predictions).value_counts(normalize=True)
    print("Prediction Distribution:")
    print(prediction_distribution)

    # Check if any class exceeds the threshold
    for value, percentage in prediction_distribution.items():
        if percentage > threshold:
            print(f"Alert: Class {value} exceeds the threshold with {percentage:.2%} of predictions.")

# Example usage of rule-based checks
rule_based_checks(final_predicted_prices)



# Step 14: Correlation Analysis
def correlation_analysis(data, predictions):
    """
    Evaluate shifts in correlations between input features and model predictions.

    Parameters:
    data (pd.DataFrame): The original dataset containing input features.
    predictions (np.ndarray): The model predictions.

    Returns:
    None
    """
    # Add predictions to the data
    data_with_predictions = data.copy()
    data_with_predictions['Prediction'] = predictions

    # Calculate correlation coefficients
    correlation_matrix = data_with_predictions.corr()

    # Display the correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.colorbar()
    plt.title('Correlation Matrix', pad=20)
    plt.show()

    # Highlight significant correlations
    significant_threshold = 0.5  # Define a threshold for significant correlations
    significant_correlations = correlation_matrix[abs(correlation_matrix) > significant_threshold]
    print("\nSignificant Correlations (|correlation| > 0.5):")
    print(significant_correlations.dropna(how='all').dropna(axis=1, how='all'))

# Example usage of correlation analysis
correlation_analysis(aligned_data, final_predicted_prices)




from river.drift import PageHinkley, ADWIN, KSWIN
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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