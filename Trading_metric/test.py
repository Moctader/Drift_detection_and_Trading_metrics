# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ffn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Step 2: Download Stock Data
# Fetch historical stock prices for Apple (AAPL)
prices = ffn.get('aapl', start='2010-01-01')

# Step 3: Analyze Performance
stats = prices.calc_stats()
stats.display()

# Extract the adjusted close price for modeling
data = prices['aapl'].values.reshape(-1, 1)  # Reshape for scaling

# Step 4: Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create dataset function
def create_dataset(dataset, time_step=1, forecast_step=1):
    X, y = [], []
    for i in range(time_step, len(dataset) - forecast_step + 1):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i + forecast_step - 1, 0])
    return np.array(X), np.array(y)

# Parameters for the dataset
time_step = 60  # Number of time steps to consider for prediction
forecast_step = 1  # Forecast 1 step ahead

# Create dataset
X, y = create_dataset(scaled_data, time_step, forecast_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

# Step 5: Build and Train the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=10, batch_size=64, callbacks=[early_stop], verbose=1)

# Step 6: Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 7: Evaluate Predictions
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Step 8: Create Trading Signals Based on Moving Averages
# Combine train and test predictions for overall analysis
predicted_prices = np.concatenate((train_predict, test_predict), axis=0)

# Create a DataFrame with predicted prices
predicted_prices_df = pd.DataFrame(predicted_prices, columns=['Predicted Price'])

# Define moving average windows
fast_window = 5  # Fast moving average window (short-term)
slow_window = 20  # Slow moving average window (long-term)

# Calculate moving averages
predicted_prices_df['Fast_MA'] = predicted_prices_df['Predicted Price'].rolling(window=fast_window).mean()
predicted_prices_df['Slow_MA'] = predicted_prices_df['Predicted Price'].rolling(window=slow_window).mean()

# Generate trading signals based on moving average crossovers
predicted_prices_df['Signal'] = 0  # Default signal to 0
predicted_prices_df['Signal'] = np.where(predicted_prices_df['Fast_MA'] > predicted_prices_df['Slow_MA'], 1, 0)  # 1 for buy
predicted_prices_df['Signal'] = np.where(predicted_prices_df['Fast_MA'].shift(1) > predicted_prices_df['Slow_MA'].shift(1), 
                                         np.where(predicted_prices_df['Fast_MA'] < predicted_prices_df['Slow_MA'], -1, predicted_prices_df['Signal']), 
                                         predicted_prices_df['Signal'])  # -1 for sell

# Step 9: Align signals with actual predictions
signals = predicted_prices_df['Signal'].fillna(0).values  # Fill NaN values with 0 for initial rows

# Step 10: Visualize Actual vs Predicted Prices and Signals
plt.figure(figsize=(14, 7))
plt.plot(data, label='Actual Prices', color='blue')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Predicted Prices (Train)', color='green')
plt.plot(np.arange(time_step + len(train_predict), time_step + len(train_predict) + len(test_predict)), test_predict, label='Predicted Prices (Test)', color='orange')

# Highlight buy/sell signals
buy_signals = predicted_prices_df[predicted_prices_df['Signal'] == 1]
sell_signals = predicted_prices_df[predicted_prices_df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Predicted Price'], marker='^', color='green', label='Buy Signal', s=5)
plt.scatter(sell_signals.index, sell_signals['Predicted Price'], marker='v', color='red', label='Sell Signal', s=5)

plt.title('AAPL Stock Price Prediction with Trading Signals Based on Moving Averages')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 11: Evaluate Trading Performance
# Convert prices to DataFrame for easier handling
prices_df = pd.DataFrame(data, columns=['Actual Price'])
prices_df['Predicted Price'] = np.nan

# Use loc to avoid chained assignment
prices_df.loc[time_step:time_step + len(train_predict) - 1, 'Predicted Price'] = train_predict.flatten()
prices_df.loc[time_step + len(train_predict):time_step + len(train_predict) + len(test_predict) - 1, 'Predicted Price'] = test_predict.flatten()

# Ensure the length of signals matches the DataFrame index
signals = np.append(signals, [0] * (len(prices_df) - len(signals)))

# Assign signals to DataFrame
prices_df['Signal'] = signals[:len(prices_df)]  # Adjusting signals to match the length of prices_df

# Calculate returns based on signals
initial_investment = 100000
prices_df['Daily Returns'] = prices_df['Actual Price'].pct_change()
prices_df['Strategy Returns'] = prices_df['Daily Returns'] * prices_df['Signal'].shift(1)  # Shift signals to align with next day returns

# Calculate cumulative returns
prices_df['Cumulative Market Returns'] = (1 + prices_df['Daily Returns']).cumprod() * initial_investment
prices_df['Cumulative Strategy Returns'] = (1 + prices_df['Strategy Returns']).cumprod() * initial_investment

# Calculate drawdown over time
prices_df['Cumulative Strategy Returns'] = prices_df['Cumulative Strategy Returns'].fillna(method='ffill')
prices_df['Peak'] = prices_df['Cumulative Strategy Returns'].cummax()
prices_df['Drawdown'] = (prices_df['Cumulative Strategy Returns'] - prices_df['Peak']) / prices_df['Peak']

# Calculate additional metrics
sharpe_ratio = ffn.calc_sharpe(prices_df['Strategy Returns'].dropna())

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(prices_df['Cumulative Market Returns'], label='Market Returns', color='blue')
plt.plot(prices_df['Cumulative Strategy Returns'], label='Strategy Returns', color='green')
plt.title('Cumulative Returns: Market vs. Strategy')
plt.xlabel('Days')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Plot drawdown over time
plt.figure(figsize=(14, 7))
plt.plot(prices_df['Drawdown'], label='Drawdown', color='red')
plt.title('Drawdown of Strategy Returns')
plt.xlabel('Days')
plt.ylabel('Drawdown')
plt.legend()
plt.show()

# Print additional metrics
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {prices_df['Drawdown'].min()}")