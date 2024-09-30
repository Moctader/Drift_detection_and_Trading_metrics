# Stock Price Prediction and Drift Detection using LSTM

This repository demonstrates the use of an LSTM (Long Short-Term Memory) model to predict Apple's stock prices (AAPL) and detect performance drift, distribution drift detection, and trading over time using various drift detection algorithms such as Page-Hinkley, ADWIN, KSWIN, and Evidently AI.


## Prerequisites

**Concept Drift** is a change in the relationship between the input data and the model target. It reflects the evolution of the underlying problem statement or process over time. Simply put, whatever you are trying to predict – it’s changing.[P(Y|X)]

**Types of concept drift**
There are different types of concept drift patterns one can observe in production. Let’s dive into those and explore some examples.

**Gradual concept drift**
When referring to “drift,” we usually expect a gradual change. After all, this is what the word itself means – a slow change or movement. This is the most frequent type of drift indeed. Gradual drift happens when the underlying data patterns change over time. 

![gradual](images/gradual_drift.png)

**Sudden concept drift**
Sudden concept drift is the opposite: an abrupt and unexpected change in the model environment. 

![sudden](images/sudden_concept_drift.png)

**Recurring concept drift**
Recurring concept drift, meaning pattern changes that happen repeatedly or follow a cycle. 

![Recurring](images/Recuring_concept_drift.png)

**Concept Drift vs. Model Drift**: Model drift is the decrease in model quality without a specified cause. Concept drift implies a change in the learned relationships. Model drift is often caused by concept drift.


- **Drift Detection Algorithm Strengths and Weaknesses**
      ![n_algo](images/comparison_between_drift_algorithom.png)

**Page-Hinkley (PH)** Page-Hinkley is a sequential analysis technique primarily used for detecting abrupt changes (drift) in the average of a signal. It was originally designed for quality control but is now adapted to concept drift detection in machine learning models.

**ADWIN (Adaptive Windowing)** ADWIN is a robust algorithm designed for detecting both abrupt and gradual drifts in a data stream. It is a sliding window-based method that automatically adjusts its window size based on the presence of drift.

**KSWIN (Kolmogorov-Smirnov Windowing)** KSWIN is a non-parametric drift detection algorithm based on the Kolmogorov-Smirnov (K-S) test, which measures the distance between two probability distributions. It is designed for detecting changes in data distribution using statistical hypothesis testing.



# Concept Drift Detection Methods

Concept drift occurs when the statistical properties of the target variable, which the model is predicting, change over time. Detecting concept drift is crucial for maintaining the performance of machine learning models. This document outlines several effective methods for detecting concept drift.

## Methods

### 1. Monitor Model Quality Metrics
    Track key performance metrics such as accuracy, precision, recall, and F1-score.
    A significant drop in these metrics can indicate that concept drift is occurring.

### 2. Use Proxy Metrics
   Observe changes in the distribution of predictions and predicted probabilities.
   These metrics can serve as early warning signs of potential drift.

### 3. Statistical Tests
   Apply statistical tests like the Kolmogorov-Smirnov test or Chi-Square test.
   Compare current data distributions with historical distributions to detect drift.

### 4. Distance Metrics
   Measure the differences between distributions using metrics like Wasserstein Distance or Population Stability Index (PSI).
   A breach of predefined thresholds indicates possible drift.

### 5. Rule-Based Checks
   Set specific rules to trigger alerts for significant changes, such as the percentage of predictions within a specific class. Helps in timely detection and response to potential drift.

### 6. Correlation Analysis
   Evaluate shifts in correlations between input features and model predictions using correlation coefficients.
   Significant changes in correlation may signal meaningful shifts in the underlying data patterns.




## Overview

feature_distribution_visualization /comparison_between_epochs_different_drift.py uses an LSTM model to make stock price predictions and analyzes model performance over time. Drift detection algorithms are used to identify periods where the model's prediction errors exceed a reference deviation, signaling changes in stock price behavior and model underperformance.

## Features

- **Stock Price Prediction**: Predict future stock prices using a trained LSTM model.
- **Drift Detection**: Identify points where the model's prediction errors exceed the reference deviation using drift detectors (Page-Hinkley, ADWIN, KSWIN).
- **Performance Metrics**: Calculate performance metrics such as MAE (Mean Absolute Error) and Maximum Deviation for both reference and current periods.
- **Visualization**: Plot actual and predicted prices, highlighting drift detection points.

## Workflow

1. **Data Collection**:
   - Fetch historical stock price data for Apple (AAPL) from Yahoo Finance using the `yfinance` library.
   
2. **Data Preprocessing**:
   - Normalize the stock 'Close' prices using `MinMaxScaler`.
   - Create time series datasets with sliding windows for LSTM training.

3. **LSTM Model Training**:
   - Train a multi-layered LSTM model with dropout and batch normalization to predict stock prices.
   - Experiment with different epoch values (50, 100, 150, 400) to compare performance.
   
4. **Prediction & Drift Detection**:
   - Split data into reference and current periods.
   - Calculate the maximum deviation between actual and predicted prices in the reference period.
   - Use drift detection algorithms to monitor the model's performance in the current period and detect drifts.

5. **Performance Evaluation**:
   - Compute key metrics such as MAE, Max Deviation, and analyze drift points for each drift detection algorithm.

## Performance Metrics

The following metrics are used to evaluate the LSTM model's prediction performance:
- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual prices.
- **Max Deviation**: The maximum deviation between predicted and actual prices in the reference period.
- **Drift Points**: The points where drift detection algorithms identified significant changes in the model's performance.


## Visualization:
   The plots include:
   - Actual vs. Predicted stock prices for both reference and current periods.
   - Drift detection points marked on the plots where each algorithm detected a drift.


### Outputs

- **Results based on different epoches and the drift algorithoms**:.
    ![n_epoches](images/drift_detection_epochs_50.png)
    ![n_epoches](images/drift_detection_epochs_100.png)
    ![n_epoches](images/drift_detection_epochs_150.png)
    ![n_epoches](images/drift_detection_epochs_400.png)



# Distribution based analysis by evidently

feature_distribution_visualization /aapl_data_synthetic_data_evidently.py demonstrates how to use synthetic data as a complementary dataset to Yahoo Finance data for comparison and drift detection using Evidently AI.

## Overview

The script performs the following tasks:
1. Fetches AAPL stock data from Yahoo Finance.
2. Preprocesses the data (calculates differences, applies Min-Max scaling).
3. Generates synthetic data that mimics the stock data.
4. Introduces drift in the synthetic prediction data.
5. Evaluates drift using Evidently AI.
6. Visualizes the original stock data, processed data, synthetic data, and drift detection in a single figure with subplots.

# Visualization 

This script visualizes the original stock data, processed data, synthetic data, and drift detection in a single figure with subplots. It demonstrates how Evidently AI can be used to detect drift in data.
### Plot the original stock data, processed data, synthetic data, and drift detection
![evidently](images/evidently_supply.png)

### Distrubtion for prediction and target
![n_evidently](images/Evidently_prediction.png)
![n_evidently](images/evidently_target.png)



## Trading 

# Stock Price Prediction and Trading Signal Generation using LSTM

Trading_metric/main.py demonstrates the use of Long Short-Term Memory (LSTM) neural networks to predict stock prices and generate trading signals based on performance metrics. The project uses historical stock data for Apple Inc. (AAPL) and includes the following steps:

1. **Data Preprocessing**: Fetching and cleaning historical stock data.
2. **LSTM Model Training**: Building and training an LSTM model to predict future stock prices.
3. **Performance Metrics Calculation**: Calculating rolling performance metrics such as Sharpe ratio, Sortino ratio, CAGR, Total Return, Max Drawdown, and YTD performance.
4. **Trading Signal Generation**: Generating buy and sell signals based on the calculated performance metrics.
5. **Visualization**: Plotting the actual and predicted stock prices, trading signals, and max drawdown.


### Outputs
![n_trading](images/trading_signal.png)
![n_trading](images/max_drawdown.png)



