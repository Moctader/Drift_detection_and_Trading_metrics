import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import wasserstein_distance, ks_2samp

# Step 1: Load Data
data = yf.download("AAPL", start="2015-01-01", end="2023-01-01")
data = data['Close'].dropna()

# Step 2: Decompose Time Series to Identify Trends and Seasonality
result = seasonal_decompose(data, model='additive', period=365)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Plot the decomposition
def plot_decomposition(data, trend, seasonal, residual):
    plt.figure(figsize=(14, 8))
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

plot_decomposition(data, trend, seasonal, residual)

# Step 3: Differencing to remove trend
differenced_data = data.diff().dropna()

# Step 4: Detrending by subtracting the trend component
detrended_data = data - trend

# Step 5: Log transform
log_data = np.log(data)

# Step 6: Test for stationarity
result = adfuller(differenced_data)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Step 7: Feature Engineering
data = pd.DataFrame(data)
data['lag1'] = data['Close'].shift(1)
data['lag2'] = data['Close'].shift(2)
data['rolling_mean'] = data['Close'].rolling(window=5).mean()
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month

# Seasonal differencing (lag of 365 for yearly seasonality)
seasonally_differenced_data = data['Close'].diff(365).dropna()

# Step 8: Scaling
# Normalization
min_max_scaler = MinMaxScaler()
data['Close_normalized'] = min_max_scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Standardization
standard_scaler = StandardScaler()
data['Close_standardized'] = standard_scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Plot normalized and standardized data for comparison
plt.figure(figsize=(14, 8))
plt.subplot(311)
plt.plot(data['Close'], label='Original')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(data['Close_normalized'], label='Normalized')
plt.legend(loc='best')
plt.subplot(313)
plt.plot(data['Close_standardized'], label='Standardized')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot differenced and detrended data for comparison
plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.plot(differenced_data, label='Differenced')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(detrended_data, label='Detrended')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Step 9: Concept Drift Detection - Using Wasserstein Distance
def detect_concept_drift(data, window_size=60):
    distances = []
    for i in range(2 * window_size, len(data)):
        window1 = data[i - 2 * window_size:i - window_size].flatten()
        window2 = data[i - window_size:i].flatten()
        dist = wasserstein_distance(window1, window2)
        distances.append(dist)
    return distances

# Drift detection on differenced data
drift_scores_diff = detect_concept_drift(differenced_data.values, window_size=60)

# Drift detection on detrended data
drift_scores_detrend = detect_concept_drift(detrended_data.dropna().values, window_size=60)

# Plot drift scores to visualize changes
plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.plot(differenced_data.index[2 * 60:], drift_scores_diff, label='Wasserstein Distance Drift Score (Differenced)', color='purple')
plt.axhline(y=np.mean(drift_scores_diff) + 2 * np.std(drift_scores_diff), color='r', linestyle='--', label='Drift Threshold')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(detrended_data.dropna().index[2 * 60:], drift_scores_detrend, label='Wasserstein Distance Drift Score (Detrended)', color='green')
plt.axhline(y=np.mean(drift_scores_detrend) + 2 * np.std(drift_scores_detrend), color='r', linestyle='--', label='Drift Threshold')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Step 10: Concept Drift Detection - Using KS Test
def detect_drift_ks(data, window_size=60):
    p_values = []
    for i in range(2 * window_size, len(data)):
        window1 = data[i - 2 * window_size:i - window_size].flatten()
        window2 = data[i - window_size:i].flatten()
        stat, p_value = ks_2samp(window1, window2)
        p_values.append(p_value)
    return p_values

# KS test on differenced data
ks_p_values_diff = detect_drift_ks(differenced_data.values, window_size=60)

# KS test on detrended data
ks_p_values_detrend = detect_drift_ks(detrended_data.dropna().values, window_size=60)

# Plot KS test p-values to visualize changes
plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.plot(differenced_data.index[2 * 60:], ks_p_values_diff, label='KS Test p-values (Differenced)', color='blue')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(detrended_data.dropna().index[2 * 60:], ks_p_values_detrend, label='KS Test p-values (Detrended)', color='orange')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Step 11: Concept Drift Detection - Using CUSUM
def detect_cusum(data, threshold=0.5, drift=0.01):
    s_pos, s_neg = 0, 0
    drift_points = []
    for i in range(1, len(data)):
        diff = data[i] - data[i - 1]
        s_pos = max(0, s_pos + diff - drift)
        s_neg = max(0, s_neg - diff - drift)
        if s_pos > threshold or s_neg > threshold:
            drift_points.append(i)
            s_pos, s_neg = 0, 0
    return drift_points

# CUSUM on differenced data
drift_points_diff = detect_cusum(differenced_data.values)

# CUSUM on detrended data
drift_points_detrend = detect_cusum(detrended_data.dropna().values)

# Plot CUSUM drift points
plt.figure(figsize=(14, 8))
plt.subplot(211)
plt.plot(differenced_data.index, differenced_data.values, label='Close Price (Differenced)', color='blue')
plt.scatter(differenced_data.index[drift_points_diff], differenced_data.values[drift_points_diff], color='red', label='Drift Points')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(detrended_data.dropna().index, detrended_data.dropna().values, label='Close Price (Detrended)', color='green')
plt.scatter(detrended_data.dropna().index[drift_points_detrend], detrended_data.dropna().values[drift_points_detrend], color='red', label='Drift Points')
plt.legend(loc='best')
plt.tight_layout()
plt.show()