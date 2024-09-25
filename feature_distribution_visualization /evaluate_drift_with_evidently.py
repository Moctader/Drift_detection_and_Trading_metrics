import pandas as pd
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import requests
import logging
import yaml
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib


GREY = '#808080'
RED = '#FF0000'

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 1. Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def data_ingestion(api_url: str) -> pd.DataFrame:

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        if not response.text.strip():
            logger.error("API response is empty")
            raise ValueError("Received empty response from the API")
        try:
            data = pd.DataFrame(response.json())
        except ValueError as e:
            logger.error("Failed to parse JSON from API response")
            raise ValueError("Invalid JSON response from the API") from e

        logger.info("Data ingestion completed successfully")
        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while fetching data: {e}")
        raise


def fetch_eurusd_data(config):
    api_base_url = config['data_ingestion']['api_url']
    start_date = pd.Timestamp(config['data_ingestion']['start_date']).timestamp()
    end_date = pd.Timestamp(config['data_ingestion']['end_date']).timestamp()

    api_url = f"{api_base_url}?api_token={config['data_ingestion']['api_token']}&fmt=json&from={int(start_date)}&to={int(end_date)}"

    try:
        data = data_ingestion(api_url)
        logger.info(" data fetched successfully from the API")

    except Exception as e:
        logger.warning(f"Failed to fetch real data from API: {e}")

    return data


def data_preprocessing(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        datetime_column = config['preprocessing']['datetime_column']
        required_columns = config['preprocessing']['required_columns']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Required columns are missing: {missing_columns}")
            raise KeyError(f"Missing columns: {missing_columns}")

        data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
        if data[datetime_column].isnull().any():
            logger.warning(f"Dropping {data[datetime_column].isnull().sum()} rows with invalid datetime values")
            data.dropna(subset=[datetime_column], inplace=True)

        data.set_index(datetime_column, inplace=True)
        logger.info("Filling missing values with the median value for each column")
        data.fillna(data.median(), inplace=True)

        remaining_nans = data[required_columns].isnull().sum().sum()
        if remaining_nans > 0:
            logger.error("Data preprocessing failed: Some columns still contain NaN values after filling")
            raise ValueError(f"NaN values remain after processing: {remaining_nans}")

        logger.info("Data preprocessing completed successfully")
        return data[required_columns].copy()

    except Exception as e:
        logger.error(f"Data preprocessing encountered an unexpected error: {e}")
        raise


def create_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
  
    try:
        window_size = config['feature_engineering']['moving_average_window']
        target_column = config['feature_engineering']['target_column']
        ma_column_name = f"MA{window_size}"

        if len(data) < window_size:
            logger.error(f"Feature engineering failed: Not enough data points for {window_size}-period MA")
            raise ValueError(f"Not enough data points for {window_size}-period MA feature")

        data[ma_column_name] = data[target_column].rolling(window=window_size).mean()
        num_nans = data[ma_column_name].isna().sum()
        logger.info(f"Rolling window created {num_nans} NaN values for {ma_column_name}")

        data[ma_column_name].ffill(inplace=True)  
        data[ma_column_name].bfill(inplace=True)

        if data[ma_column_name].isnull().all():
            logger.warning(f"All values in {ma_column_name} are still NaN after forward-filling. Applying fallback strategy.")

            if len(data[target_column].dropna()) > 0:
                fallback_value = data[target_column].mean()
                data[ma_column_name].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in {ma_column_name} with mean: {fallback_value:.2f}")
            else:
                fallback_value = 0  
                data[ma_column_name].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in {ma_column_name} with fallback value: {fallback_value:.2f}")

        logger.info(f"Feature engineering completed successfully with {ma_column_name} feature added")
        return data
    
    except Exception as e:
        logger.error(f"Feature engineering encountered an unexpected error: {e}")
        raise


def train_model(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:

    try:
        required_columns = ['open', 'high', 'low', 'close', 'MA3']
        data['shifted_close'] = data['close'].shift(-1)

        X = data[required_columns].copy()
        X.ffill(inplace=True)
        X.fillna(0, inplace=True)  
        y = data['shifted_close'].values  
        X = X[:-1]
        y = y[:-1]

        # Time-based train-test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model_type = config['model']['type']
        param_grid = config['model']['param_grid']
        model_name = config['model']['name']  

        if model_type == 'ElasticNet':
            model = ElasticNet(max_iter=10000)
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
        else:
            logger.error(f"Unknown model type '{model_type}' specified in config.")
            raise ValueError(f"Unknown model type: {model_type}")

        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate the model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Log model performance
        logger.info(f"Model training completed. Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
        logger.info(f"Model training completed. Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

        data['predicted'] = np.nan
        data.loc[data.index[:-1], 'predicted'] = best_model.predict(X)

        model_file_path = f"{model_name}.joblib"
        joblib.dump(best_model, model_file_path)
        logger.info(f"Model saved as {model_file_path}")
        data.fillna(data.median(), inplace=True)

        return data, best_model
    
    except Exception as e:
        logger.error(f"Model training encountered an unexpected error: {e}")
        raise


def prepare_data_for_prediction(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = data_preprocessing(data, config)
    data = create_features(data, config)
    return data

def make_predictions(model_runner, data: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        required_columns = config['prediction']['required_columns']
        data[required_columns] = data[required_columns].ffill()
        data[required_columns] = data[required_columns].bfill()
        data[required_columns] = data[required_columns].fillna(0)
        features = data[required_columns].values
        predictions = model_runner.predict(features)
        data['predicted'] = predictions
        logger.info("Predictions completed successfully.")
        return data

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def process_current_data(current_data_with_feature: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        model_runner = joblib.load("elastic_net_model.joblib")
        prepared_data = prepare_data_for_prediction(current_data_with_feature, config)
        prepared_data['shifted_close'] = prepared_data['close'].shift(-1)
        data_with_predictions = make_predictions(model_runner, prepared_data, config)
        data_with_predictions.fillna(data_with_predictions.median(), inplace=True)  
        return data_with_predictions

    except Exception as e:
        logger.error(f"Processing current data failed: {e}")
        raise


def evaluare_drift(ref: pd.Series, curr: pd.Series):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="kl_div", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="jensenshannon", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="wasserstein", stattest_threshold=0.1)
    ])
    report.run(reference_data=pd.DataFrame({"value": ref}), current_data=pd.DataFrame({"value": curr}))
    results = report.as_dict()
  
    
    drift_report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, metric in enumerate(results['metrics']):
        stat_test_name = metric['result'].get('stattest_name', 'Unknown')
        drift_report.loc[i, 'stat_test'] = stat_test_name
        drift_report.loc[i, 'drift_score'] = metric['result']['drift_score']
        drift_report.loc[i, 'is_drifted'] = metric['result']['drift_detected']

    return drift_report

def plot_example(ref: pd.Series, curr: pd.Series, ref_name: str = "Reference", curr_name: str = "Current"):
    fig = plt.figure(constrained_layout=True, figsize=(15,7))

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # plot feature in time
    ref_points = int(np.round(150 * len(ref) /(len(ref) + len(curr))))
    curr_points = 150 - ref_points

    ref_in_time = [np.mean(x) for x in np.array_split(ref, ref_points)]
    curr_in_time = [np.mean(x) for x in np.array_split(curr, curr_points)]

    ax1.plot(range(ref_points), ref_in_time, color=GREY, label=ref_name)
    ax1.plot(range(ref_points, ref_points + curr_points), curr_in_time, color=RED, label=curr_name)
    ax1.legend()

    # plot reference distribution
    sns.histplot(ref, color=GREY, ax=ax2)
    ax2.set_title(f'{ref_name} Distribution')

    # plot current distribution
    sns.histplot(curr, color=RED, ax=ax3)
    ax3.set_title(f'{curr_name} Distribution')

    # plot two distributions
    sns.histplot(ref, color=GREY, ax=ax4, label=ref_name)
    sns.histplot(curr, color=RED, ax=ax4, label=curr_name)
    ax4.legend()
    plt.show()



def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    data = data_preprocessing(data, config)
    training_data_with_features = create_features(data, config)
    data_with_prediction_shifted_close, model = train_model(training_data_with_features, config)
    current_data = pd.read_csv('current.csv')
    current_data_predict_shifted_close = process_current_data(current_data, config)
    target_col = 'open'
    drift_report = evaluare_drift(data_with_prediction_shifted_close[target_col], current_data_predict_shifted_close[target_col])
    print(drift_report)
    plot_example(data_with_prediction_shifted_close[target_col], current_data_predict_shifted_close[target_col], ref_name="Historical Data", curr_name="Current Data")    



# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)