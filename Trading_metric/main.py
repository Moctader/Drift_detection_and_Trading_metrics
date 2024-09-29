import ffn
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Fetch stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Remove rows with NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Adj Close'])

# Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stop], verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Combine train and test predictions
predicted_prices = np.concatenate((train_predict, test_predict), axis=0)
predicted_prices_df = pd.DataFrame(predicted_prices, index=data.index[time_step + 1:], columns=['Predicted Price'])

# Calculate rolling performance metrics
window = 252  # 1 year of trading days
predicted_prices_df['Returns'] = predicted_prices_df['Predicted Price'].pct_change()
predicted_prices_df['Sharpe'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=False)
predicted_prices_df['Sortino'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: x.mean() / x[x < 0].std() * np.sqrt(252), raw=False)
predicted_prices_df['CAGR'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: (x.iloc[-1] / x.iloc[0]) ** (252 / len(x)) - 1, raw=False)
predicted_prices_df['Total Return'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1, raw=False)
predicted_prices_df['Max Drawdown'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min(), raw=False)
predicted_prices_df['YTD'] = predicted_prices_df['Predicted Price'].pct_change().cumsum()

# Generate trading signals based on rolling performance metrics
def generate_signals(df):
    signals = np.zeros(len(df))  # Default signal to 0
    
    # Generate buy signals
    buy_condition = (df['Sharpe'] > 1.0) & (df['Sortino'] > 2.0) & (df['CAGR'] > 0.0) & (df['YTD'] > 0.0)
    signals = np.where(buy_condition, 1, signals)
    
    # Generate sell signals
    sell_condition = (df['Total Return'] < 0) | (df['Max Drawdown'] > 0.2)
    signals = np.where(sell_condition, -1, signals)
    
    return signals

# Create signals based on the metrics
predicted_prices_df['Signal'] = generate_signals(predicted_prices_df)

# Plot the signals and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(data['Adj Close'], label='Actual Prices', color='blue')
plt.plot(predicted_prices_df.index, predicted_prices_df['Predicted Price'], label='Predicted Prices', color='orange')

# Highlight buy/sell signals
buy_signals = predicted_prices_df[predicted_prices_df['Signal'] == 1]
sell_signals = predicted_prices_df[predicted_prices_df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Predicted Price'], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_signals.index, sell_signals['Predicted Price'], marker='v', color='red', label='Sell Signal', s=100)

plt.title('AAPL Stock Price with Trading Signals Based on LSTM Predictions and Performance Metrics')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the max drawdown in a separate figure
plt.figure(figsize=(14, 7))
plt.plot(predicted_prices_df.index, predicted_prices_df['Max Drawdown'], label='Max Drawdown', color='purple', linestyle='--')
plt.title('Max Drawdown of Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Max Drawdown')
plt.legend()
plt.show()