import ffn
import yfinance as yf
import pandas as pd
import numpy as np

# Fetch stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Calculate returns
returns = ffn.to_returns(data['Adj Close'])

# Calculate performance metrics
stats = returns.calc_stats()
sharpe_ratio = stats.stats['daily_sharpe']
sortino_ratio = stats.stats['daily_sortino']
cagr = stats.stats['cagr']
total_return = stats.stats['total_return']
max_drawdown = stats.stats['max_drawdown']
ytd_performance = stats.stats['ytd']

# Generate trading signals
def generate_signals(sharpe, sortino, cagr, total_return, max_drawdown, ytd_performance):
    signals = np.zeros(len(data))  # Default signal to 0
    
    # Generate buy signals
    buy_condition = (sharpe > 1.0) & (sortino > 2.0) & (cagr > 0.0) & (ytd_performance > 0.0)
    signals = np.where(buy_condition, 1, signals)
    
    # Generate sell signals
    sell_condition = (total_return < 0) | (max_drawdown > 0.2)
    signals = np.where(sell_condition, -1, signals)
    
    return signals

# Create signals based on the metrics
signals = generate_signals(sharpe_ratio, sortino_ratio, cagr, total_return, max_drawdown, ytd_performance)
print(f"Generated Trading Signals: {signals}")

# Add signals to DataFrame for visualization
data['Signal'] = signals

# Plot the signals
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(data['Adj Close'], label='Actual Prices', color='blue')

# Highlight buy/sell signals
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='red', label='Sell Signal', s=100)

plt.title('AAPL Stock Price with Trading Signals Based on Performance Metrics')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()