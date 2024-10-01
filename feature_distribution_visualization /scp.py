import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river.drift import PageHinkley, ADWIN, KSWIN
from typing import List

class SPC:
    def __init__(self, delta: int, window: int = 20, minPoints: int = 10):
        self.delta = delta
        self.window = window
        self.minPoints = minPoints
        self.pointList = []
        self.sigma = 0
        self.upperBound = 0
        self.lowerBound = 0
        self.processMean = 0

    def calculate_stats(self, datastream: np.array) -> None:
        """Calculate the mean, standard deviation, and bounds of the datastream."""
        self.processMean = datastream.mean()
        self.sigma = datastream.std()
        self.upperBound = self.processMean + self.delta * self.sigma
        self.lowerBound = self.processMean - self.delta * self.sigma

    def add_element(self, datastream: np.array) -> List:
        """Test each datapoint inside datastream against upper and lower bound and store results in pointList."""
        self.calculate_stats(datastream)
        self.pointList = np.logical_or(datastream <= self.lowerBound, datastream >= self.upperBound)
        return self.pointList

    def detected_change(self) -> bool:
        """Test whether a certain number of points fall outside upper and lower bound within a certain number of days."""
        num_errors = np.sum(np.lib.stride_tricks.sliding_window_view(self.pointList, window_shape=(self.window,)), axis=1)
        drift_comparasion = num_errors >= self.minPoints
        drift_region = np.nonzero(drift_comparasion)
        return drift_comparasion, drift_region

# Generate synthetic reference data (without drift)
np.random.seed(0)
reference_length = 100  # Length of reference data
# Simulate realistic financial data (returns)
ref_actual = np.random.normal(loc=0.001, scale=0.02, size=reference_length)  # Daily returns

# Generate synthetic current data with different types of drift
current_length = 400  # Length of current data

# Initial segment without drift (simulating stable market)
initial_segment = np.random.normal(loc=0.001, scale=0.02, size=current_length // 4)

# Abrupt drift (e.g., market crash)
abrupt_drift = np.random.normal(loc=-0.1, scale=0.01, size=current_length // 4)  # Sharper decline

# Gradual drift (e.g., market recovery)
gradual_drift = np.linspace(-0.1, 0.02, current_length // 4) + np.random.normal(loc=0, scale=0.02, size=current_length // 4)

# Recurrent drift (back to stable market)
recurrent_drift = np.random.normal(loc=0.001, scale=0.02, size=current_length // 4)

# Combine all segments to form the current data
curr_actual = np.concatenate([initial_segment, abrupt_drift, gradual_drift, recurrent_drift])

# Generate synthetic predictions for current data with added noise
curr_predicted = curr_actual + np.random.normal(loc=0, scale=0.005, size=len(curr_actual))  # Smaller noise for realistic predictions

# Define actual drift points (for testing)
actual_drifts = set([
    len(initial_segment), 
    len(initial_segment) + len(abrupt_drift), 
    len(initial_segment) + len(abrupt_drift) + len(gradual_drift)
])

# Create DataFrames for reference and current data
reference_data = pd.DataFrame({'Original': ref_actual})
current_data = pd.DataFrame({'Actual': curr_actual, 'Predicted': curr_predicted})

# Define performance metrics
def calculate_metrics(detected_drifts, actual_drifts, total_points):
    if len(actual_drifts) == 0:
        return 0, 0, 0, 0  # Avoid division by zero

    detected_drifts_set = set(detected_drifts)
    detection_rate = len(detected_drifts_set & actual_drifts) / len(actual_drifts)
    false_negative_rate = len(actual_drifts - detected_drifts_set) / len(actual_drifts)
    false_positive_rate = len(detected_drifts_set - actual_drifts) / max((total_points - len(actual_drifts)), 1)  # Avoid division by zero
    detection_delay = np.mean([min(abs(d - a) for a in actual_drifts) for d in detected_drifts_set]) if detected_drifts_set else 0
    return detection_rate, false_negative_rate, false_positive_rate, detection_delay

# Evaluate algorithms with different configurations
def evaluate_drift_detection(curr_actual, curr_predicted, actual_drifts):
    detectors = {
        "Page-Hinkley": PageHinkley(min_instances=30, threshold=0.001),  # Lowered threshold for sensitivity
        "ADWIN": ADWIN(delta=0.09),  # More sensitive
        "KSWIN": KSWIN(alpha=0.1),   # Lowered alpha for better detection
        "SPC": SPC(delta=30, window=20, minPoints=10)  # SPC detector
    }

    detected_drifts = {name: [] for name in detectors.keys()}

    for detector_name, detector in detectors.items():
        if detector_name == "SPC":
            # Use SPC specific methods
            detector.add_element(curr_predicted)
            _, drift_region = detector.detected_change()
            detected_drifts[detector_name] = drift_region[0].tolist()
        else:
            for idx, (actual, predicted) in enumerate(zip(curr_actual, curr_predicted)):
                error = abs(actual - predicted)
                detector.update(error)
                if detector.drift_detected:
                    detected_drifts[detector_name].append(idx)

    results = {}
    total_points = len(curr_actual)  # Total number of points in the current data

    for name, drifts in detected_drifts.items():
        print(f"Detector: {name}, Drift Points: {len(drifts)}")  # Debugging output

        # Store metrics
        metrics = calculate_metrics(set(drifts), actual_drifts, total_points)
        results[name] = metrics

    return results, detected_drifts

# Evaluate drift detection
results, detected_drifts = evaluate_drift_detection(curr_actual, curr_predicted, actual_drifts)

# Print detection metrics for each detector
for detector_name, metrics in results.items():
    print(f"Detector: {detector_name}")
    print(f"Detection Rate: {metrics[0]:.4f}, False Negative Rate: {metrics[1]:.4f}, False Positive Rate: {metrics[2]:.4f}, Detection Delay: {metrics[3]:.4f}")

# Plotting drift points with opacity adjustments
def plot_drift_points(curr_actual, curr_predicted, detected_drifts, actual_drifts):
    plt.figure(figsize=(12, 6))

    # Plot current actual and predicted data with adjusted opacity
    plt.plot(curr_actual, label='Current Actual Data', color='green', alpha=0.6)  # Increased opacity
    plt.plot(curr_predicted, label='Current Predicted Data', color='red', alpha=0.6)  # Increased opacity

    colors = ['purple', 'magenta', 'cyan', 'orange']
    for i, (detector_name, drifts) in enumerate(detected_drifts.items()):
        if drifts:
            plt.scatter(drifts, curr_actual[drifts], color=colors[i], label=f'{detector_name} Drift Points', s=100, alpha=0.7)  # Increased opacity

    # Plot actual drift points with higher opacity for clarity
    plt.scatter(list(actual_drifts), curr_actual[list(actual_drifts)], color='black', label='Actual Drift Points', s=100, marker='x', alpha=1)  # Max opacity

    plt.title('Drift Detection Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(alpha=0.3)  # Light grid for better readability
    plt.show()

# Plot drift points
plot_drift_points(curr_actual, curr_predicted, detected_drifts, actual_drifts)