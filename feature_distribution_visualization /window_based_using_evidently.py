import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler

# Define colors
GREY = '#808080'
RED = '#FF0000'

def evaluate_drift(ref: pd.DataFrame, curr: pd.DataFrame):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="kl_div", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="jensenshannon", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="wasserstein", stattest_threshold=0.1)
    ])
    report.run(reference_data=ref, current_data=curr)
    results = report.as_dict()
  
    drift_report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, metric in enumerate(results['metrics']):
        stat_test_name = metric['result'].get('stattest_name', 'Unknown')
        drift_report.loc[i, 'stat_test'] = stat_test_name
        drift_report.loc[i, 'drift_score'] = metric['result']['drift_score']
        drift_report.loc[i, 'is_drifted'] = metric['result']['drift_detected']

    return drift_report

def plot_example(ref: pd.DataFrame, curr: pd.DataFrame, ref_name: str = "Reference", curr_name: str = "Current"):
    fig = plt.figure(constrained_layout=True, figsize=(15,7))

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # plot feature in time
    ref_points = int(np.round(150 * len(ref) /(len(ref) + len(curr))))
    curr_points = 150 - ref_points

    ref_in_time = [np.mean(x) for x in np.array_split(ref['value'], ref_points)]
    curr_in_time = [np.mean(x) for x in np.array_split(curr['value'], curr_points)]

    ax1.plot(range(ref_points), ref_in_time, color=GREY, label=ref_name, alpha=0.7)
    ax1.plot(range(ref_points, ref_points + curr_points), curr_in_time, color=RED, label=curr_name, alpha=0.7)
    ax1.legend()

    # plot reference distribution
    sns.histplot(ref['value'], color=GREY, ax=ax2, alpha=0.7)
    ax2.set_title(f'{ref_name} Distribution')

    # plot current distribution
    sns.histplot(curr['value'], color=RED, ax=ax3, alpha=0.7)
    ax3.set_title(f'{curr_name} Distribution')

    # plot two distributions
    sns.histplot(ref['value'], color=GREY, ax=ax4, label=ref_name, alpha=0.7)
    sns.histplot(curr['value'], color=RED, ax=ax4, label=curr_name, alpha=0.7)
    ax4.legend()
    plt.show()

def create_windows(data, column, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[column].iloc[i:i+window_size].values)
    return pd.DataFrame(windows, columns=[f'{column}_{i}' for i in range(window_size)]).stack().reset_index(drop=True).to_frame(name='value')

def run_pipeline(window_size=60):
    # 1. Fetch Data
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    # 2. Preprocess Data
    data = data[['Close']]  # Use 'Close' prices
    
    # MinMax Scaling
    scaler = MinMaxScaler()
    data[['Close']] = scaler.fit_transform(data[['Close']])
    
    # Calculate log return
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate difference between two days
    data['Diff'] = data['Close'].diff()
    
    # Drop NaN values resulting from calculations
    data = data.dropna()

    # Create windows and evaluate drift iteratively
    drift_reports = []
    for start in range(0, len(data) - 2 * window_size, window_size):
        reference_data = data.iloc[start:start+window_size]
        current_data = data.iloc[start+window_size:start+2*window_size]

        # Create windows
        reference_windows = create_windows(reference_data, 'LogReturn', window_size)
        current_windows = create_windows(current_data, 'LogReturn', window_size)

        # Evaluate Drift
        drift_report = evaluate_drift(reference_windows, current_windows)
        drift_reports.append(drift_report)
        print(f"Drift report for window starting at index {start}:\n", drift_report)

        # Plot Example
        plot_example(reference_windows, current_windows, ref_name=f"Window {start}-{start+window_size-1}", curr_name=f"Window {start+window_size}-{start+2*window_size-1}")

# Main function to run everything
if __name__ == "__main__":
    window_size=60
    run_pipeline(window_size)