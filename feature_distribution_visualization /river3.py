import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from river import drift

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

# Step 2: Initialize the ADWIN and Page-Hinkley drift detectors
adwin_detector = drift.ADWIN(delta=0.05)  # Adjust sensitivity with delta
page_hinkley_detector = drift.PageHinkley()
adwin_drifts = []
page_hinkley_drifts = []

# Step 3: Process the synthetic data stream
for i, val in enumerate(synthetic_prices):
    adwin_detector.update(val)  # Update the ADWIN detector with the current value
    page_hinkley_detector.update(val)  # Update the Page-Hinkley detector with the current value

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
def plot_data(synthetic_prices, drifts=None, title='Synthetic Stock Price with Detected Drifts'):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    ax1.grid()
    ax1.plot(synthetic_prices, label='Synthetic Close Price', color='blue')
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red', linestyle='--', label='Detected Drift' if drift_detected == drifts[0] else "")
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Step 5: Plot the synthetic data and detected drifts for ADWIN
plot_data(synthetic_prices, adwin_drifts, title='Synthetic Stock Price with ADWIN Detected Drifts')

# Step 6: Plot the synthetic data and detected drifts for Page-Hinkley
plot_data(synthetic_prices, page_hinkley_drifts, title='Synthetic Stock Price with Page-Hinkley Detected Drifts')