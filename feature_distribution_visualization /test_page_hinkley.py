import numpy as np
import matplotlib.pyplot as plt
from river.drift import PageHinkley

# Parameters
segment_length = 100
num_segments = 11  # 10 drifts, so 11 segments
drift_points = [segment_length * i for i in range(1, num_segments)]
buffer_zone = 5  # Buffer zone around known drift points

# Generate data segments with more pronounced changes in means
data_segments = [np.random.normal(loc=50 + 20 * i, scale=5, size=segment_length) for i in range(num_segments)]
drift_data = np.concatenate(data_segments)

# Initialize Page-Hinkley drift detector
ph_detector = PageHinkley(min_instances=5, threshold=140)  # Adjust sensitivity as needed

# Store drift points
detected_drifts = []
false_alarms = 0

# Evaluate detector on data with multiple drifts
for i, val in enumerate(drift_data):
    ph_detector.update(val)
    if ph_detector.drift_detected:
        # Check if the detected drift is within the buffer zone of any known drift point
        is_true_detection = any(abs(i - drift_point) <= buffer_zone for drift_point in drift_points)
        if is_true_detection:
            detected_drifts.append((i, 'true_detection'))
        else:
            false_alarms += 1
            detected_drifts.append((i, 'false_alarm'))
    # Debug print to understand the detection process
    print(f'Index: {i}, Value: {val}, Drift Detected: {ph_detector.drift_detected}')

print(f'Total Detected Drifts: {len(detected_drifts)}')
print(f'False Alarms: {false_alarms}')

# Plotting the data and detected drifts
plt.figure(figsize=(12, 6))
plt.plot(drift_data, label='Data with Drift')
for drift_point, drift_type in detected_drifts:
    if drift_type == 'true_detection':
        plt.axvline(drift_point, color='blue', linestyle='--', label='True Detection' if drift_point == detected_drifts[0][0] else "")
    else:
        plt.axvline(drift_point, color='red', linestyle='--', label='False Alarm' if drift_point == detected_drifts[0][0] else "")
for known_drift_point in drift_points:
    plt.axvline(known_drift_point, color='green', linestyle='-', label='Known Drift Point' if known_drift_point == drift_points[0] else "")
plt.legend()
plt.show()