import numpy as np

def create_dataset(data, time_step, forecast_step):
 
    X, y = [], []
    
    for i in range(time_step, len(data) - forecast_step + 1):
        X.append(data[i-time_step:i, 0])
        y.append(data[i + forecast_step - 1, 0])  
    return np.array(X), np.array(y)

# Example dynamic testing with varying time_step and forecast_step
data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Testing with time_step=3 and forecast_step=1
time_step = 3
forecast_step = 1
X, y = create_dataset(data, time_step, forecast_step)

print("X:", X)
print("y:", y)

# Expected results for time_step=3, forecast_step=1
expected_X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]
])

expected_y = np.array([4, 5, 6, 7, 8, 9, 10])

# Assert that the values are correct
assert np.array_equal(X, expected_X), f"X is incorrect. Expected {expected_X}, but got {X}"
assert np.array_equal(y, expected_y), f"y is incorrect. Expected {expected_y}, but got {y}"

print("Test passed for forecast_step =", forecast_step)

# Testing with a different forecast step 
forecast_step = 2
X, y = create_dataset(data, time_step, forecast_step)

print("X with forecast_step=2:", X)
print("y with forecast_step=2:", y)

# Expected results for time_step=3, forecast_step=2
expected_X_forecast_2 = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8]
])

expected_y_forecast_2 = np.array([5, 6, 7, 8, 9, 10])

# Assert for forecast_step=2
assert np.array_equal(X, expected_X_forecast_2), f"X is incorrect for forecast_step=2. Expected {expected_X_forecast_2}, but got {X}"
assert np.array_equal(y, expected_y_forecast_2), f"y is incorrect for forecast_step=2. Expected {expected_y_forecast_2}, but got {y}"

print("Test passed for forecast_step =", forecast_step)

# Testing dynamically with other parameters 
time_step = 4
forecast_step = 3
X, y = create_dataset(data, time_step, forecast_step)

print("X with time_step=4, forecast_step=3:", X)
print("y with time_step=4, forecast_step=3:", y)

