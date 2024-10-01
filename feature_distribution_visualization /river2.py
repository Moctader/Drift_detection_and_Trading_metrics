import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from river import drift
import yfinance as yf

# Fetch AAPL time series data from Yahoo Finance
aapl_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
aapl_close = aapl_data['Close'].values

# Initialize the ADWIN drift detector
drift_detector = drift.ADWIN()
drifts = []

# Process the data stream
for i, val in enumerate(aapl_close):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        print(f'Change detected at index {i}')
        drifts.append(i)
        drift_detector = drift.ADWIN()   # Reinitialize the detector to reset it

# Auxiliary function to plot the data
def plot_data(aapl_close, drifts=None):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    ax1.grid()
    ax1.plot(aapl_close, label='AAPL Close Price')
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red')
    plt.title('AAPL Close Price with Detected Drifts')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Plot the data and detected drifts
plot_data(aapl_close, drifts)