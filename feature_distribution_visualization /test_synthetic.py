import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river.drift import PageHinkley, ADWIN, KSWIN

# Generate synthetic reference data (without drift)
np.random.seed(0)
reference_length = 100  # Length of reference data
ref_actual = np.random.normal(loc=50, scale=5, size=reference_length)

# Generate synthetic current data with different types of drift
current_length = 400  # Length of current data

# Initial segment without drift
initial_segment = np.random.normal(loc=50, scale=5, size=current_length // 4)

# Abrupt drift
abrupt_drift = np.random.normal(loc=100, scale=5, size=current_length // 4)

# Gradual drift
gradual_drift = np.linspace(100, 150, current_length // 4) + np.random.normal(loc=0, scale=5, size=current_length // 4)

# Recurrent drift (back to initial distribution)
recurrent_drift = np.random.normal(loc=50, scale=5, size=current_length // 4)

# Combine all segments to form the current data
curr_actual = np.concatenate([initial_segment, abrupt_drift, gradual_drift, recurrent_drift])

# Generate synthetic predictions for current data with added noise to simulate prediction errors
curr_predicted = curr_actual + np.random.normal(loc=0, scale=10, size=len(curr_actual))

# Create DataFrames for reference and current data
reference_data = pd.DataFrame({
    'Original': ref_actual
})

current_data = pd.DataFrame({
    'Actual': curr_actual,
    'Predicted': curr_predicted
})

# Define performance metrics
def calculate_metrics(detected_drifts, actual_drifts, total_points):
    if len(actual_drifts) == 0:
        return 0, 0, 0, 0  # Avoid division by zero

    detection_rate = len(detected_drifts & actual_drifts) / len(actual_drifts)
    false_negative_rate = len(actual_drifts - detected_drifts) / len(actual_drifts)
    false_positive_rate = len(detected_drifts - actual_drifts) / max((total_points - len(actual_drifts)), 1)  # Avoid division by zero
    detection_delay = np.mean([min(abs(d - a) for a in actual_drifts) for d in detected_drifts]) if detected_drifts else 0
    return detection_rate, false_negative_rate, false_positive_rate, detection_delay

# Evaluate algorithms with different configurations
def evaluate_drift_detection(curr_actual, curr_predicted):
    detectors = {
        "Page-Hinkley": PageHinkley(threshold=10, alpha=0.005),  # Adjusted sensitivity
        "ADWIN": ADWIN(delta=0.7),  # Adjusted sensitivity
        "KSWIN": KSWIN(alpha=0.5)   # Adjusted sensitivity
    }

    detected_drifts = {name: [] for name in detectors.keys()}

    for detector_name, detector in detectors.items():
        # Use error between current actual and current predicted to update detectors
        for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
            error = abs(actual - predicted)
            detector.update(error)
            if detector.drift_detected:
                detected_drifts[detector_name].append(idx)

    results = {}
    total_points = len(curr_actual)  # Total number of points in the current data

    for name, drifts in detected_drifts.items():
        print(f"Detector: {name}, Drift Points: {len(drifts)}")

        # Store metrics
        metrics = calculate_metrics(set(drifts), set(range(total_points)), total_points)
        results[name] = metrics

    return results, detected_drifts

# Evaluate drift detection
results, detected_drifts = evaluate_drift_detection(curr_actual, curr_predicted)

# Plotting drift points
def plot_drift_points(curr_actual, curr_predicted, detected_drifts):
    plt.figure(figsize=(12, 6))

    # Plot current actual and predicted data
    plt.plot(curr_actual, label='Current Actual Data', color='green', alpha=0.5)
    plt.plot(curr_predicted, label='Current Predicted Data', color='red', alpha=0.5)

    colors = ['purple', 'magenta', 'cyan']
    for i, (detector_name, drifts) in enumerate(detected_drifts.items()):
        if drifts:
            plt.scatter(drifts, curr_actual[drifts], color=colors[i], label=f'{detector_name} Drift Points', s=100)

    plt.title('Drift Detection Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Plot drift points
plot_drift_points(curr_actual, curr_predicted, detected_drifts)