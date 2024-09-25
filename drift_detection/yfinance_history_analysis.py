import yfinance as yf
import pandas as pd

apple_data = yf.Ticker('AAPL')
history_data = apple_data.history(period='max')
splits = history_data[history_data['Stock Splits'] >= 1]
days_before_after = 1

split_surrounding_data_list = []

for split_date in splits.index:
    start_date = split_date - pd.Timedelta(days=days_before_after)
    end_date = split_date + pd.Timedelta(days=days_before_after)
    surrounding_data = history_data.loc[start_date:end_date]
    split_surrounding_data_list.append(surrounding_data)

split_surrounding_data = pd.concat(split_surrounding_data_list)
selected_columns = split_surrounding_data[['Open','High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

print(selected_columns)









