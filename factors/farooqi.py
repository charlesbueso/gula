"""
    Resilient Macro-portfolio
    First, take a closer look at traditional defensive stocks. 
    Second, focus on industry themes that don’t depend on economic growth. 
    Third, pay close attention to balance sheets.

    F A C T O R
        Uses latest month FRED https://fred.stlouisfed.org/ 
        open source economic indicators

        Positive:
            Economic (higher)
            1. Real Gross Domestic Product (GDP)
            2. Industrial Production
            Employment (higher)
            3. Nonfarm Payrolls
            4. Unemployment Rate
            Inflation (moderate, set a>=x>=b):
            5. Consumer Price Index (CPI)
            6. Producer Price Index (PPI)
            Interest rates (lower):
            7. Federal Funds Rate
            8. Treasury Yields


"""

from fredapi import Fred
import datetime
import pandas as pd
import os
import logging
import json
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

DATABASE_DIR = r"C:/Users/papia/OneDrive/Databases/FRED"

LOGGER = logging.getLogger(__name__)

today = datetime.date.today()


def main():
    """
        Fetch the latest economic indicators from FRED
        Will rerun everyday to get the latest data
    """

    if os.path.exists(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv"):
        LOGGER.info("Data is up-to-date.")
        df = pd.read_csv(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv")
        with open(f"{DATABASE_DIR}/{str(today)}/latest_indicators.json", 'r') as f:
            latest = json.load(f)

        return df, latest
    
    os.makedirs(f"{DATABASE_DIR}/{str(today)}", exist_ok=True)

    # You need to register for an API key at the FRED website
    fred = Fred(api_key='4c3e5d4494e935f26c612cca3b83e4ff')

    # You can find these IDs on the FRED website
    series_ids = {
        'Real Gross Domestic Product (GDP)': 'GDPC1',
        'Industrial Production': 'INDPRO',
        'Nonfarm Payrolls': 'PAYEMS',
        'Unemployment Rate': 'UNRATE',
        'Consumer Price Index (CPI)': 'CPIAUCSL',
        'Producer Price Index (PPI)': 'PPIACO',
        'Federal Funds Rate': 'FEDFUNDS',
        'Treasury Yields': 'GS10'
    }

    data = pd.DataFrame()
    latest = series_ids.copy()

    # Fetch the latest values for each series
    for name, series_id in series_ids.items():
        series = fred.get_series(series_id)
        # latest_date = series.index[-1]
        latest_date = series.index[-1].strftime('%Y-%m-%d')  # Convert Timestamp to string
        latest_value = series.iloc[-1]
        print(f"{name} ({latest_date}): {latest_value}")
        #Save the series to a df
        data[name] = series
        latest[name] = [latest_date, latest_value]
        # Save latest as a dictionary in a json file in the same directory
        with open(f"{DATABASE_DIR}/{str(today)}/latest_indicators.json", 'w') as f:
            json.dump(latest, f)

    data.to_csv(f"{DATABASE_DIR}/{str(today)}/economic_indicators.csv")
    return data, latest


def farooqi_weight(data_file, weights):
  """
  Calculates a weighted influence value on a target variable.

  Args:
      data_file (str): Path to the CSV file containing economic data.
      weights (dict): Dictionary containing weights for each variable.

  Returns:
      float: Weighted influence value.
  """

  df = pd.read_csv(data_file)
  latest_data = df.iloc[-1]

  weighted_sum = np.sum(latest_data * weights.values())

  return weighted_sum


def train_test_split(df, training_percentage=0.75):
    """
    Splits a time series DataFrame into training and test sets.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.
    training_percentage (float): The percentage of data to use for training. Must be between 0 and 1.

    Returns:
    pandas.DataFrame, pandas.DataFrame: The training and test sets.
    """
    # Calculate the number of rows to use for training
    train_size = int(len(df) * training_percentage)

    # Split the DataFrame
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    return train, test

def make_VAR_predictions(fred_historical, plot=True, save=True):
    """
    Visualizes the predictions made by the model.
    """
    # Convert all columns to numeric, errors='coerce' will set non-numeric values to NaN
    fred_historical = fred_historical.apply(pd.to_numeric, errors='coerce')

    # Mean interpolation
    df_mean = fred_historical.fillna(fred_historical.mean())

    # Remove first column of df_mean
    df_mean = df_mean.iloc[:, 1:]

    # Split data
    train, test = train_test_split(df_mean, 0.86)

    # Fit the model
    model = VAR(train)
    results = model.fit()

    # Make forecast
    lag_order = results.k_ar
    forecast = results.forecast(train.values[-lag_order:], len(test))  # Forecast 5 steps ahead, now in len(test)
    # results.plot_forecast(15)
    # plt.show()

    predictions = pd.DataFrame({'Forecasted Real Gross Domestic Product (GDP)': forecast[:, 0],
                                'Real Gross Domestic Product (GDP)': test['Real Gross Domestic Product (GDP)'].values,
                                'Forecasted Industrial Production': forecast[:, 1],
                                'Industrial Production': test['Industrial Production'].values,
                                'Forecasted Nonfarm Payrolls': forecast[:, 2],
                                'Nonfarm Payrolls': test['Nonfarm Payrolls'].values,
                                'Forecasted Unemployment Rate': forecast[:, 3],
                                'Unemployment Rate': test['Unemployment Rate'].values,
                                'Forecasted Consumer Price Index (CPI)': forecast[:, 4],
                                'Consumer Price Index (CPI)': test['Consumer Price Index (CPI)'].values,
                                'Forecasted Producer Price Index (PPI)': forecast[:, 5],
                                'Producer Price Index (PPI)': test['Producer Price Index (PPI)'].values,
                                'Forecasted Federal Funds Rate': forecast[:, 6],
                                'Federal Funds Rate': test['Federal Funds Rate'].values,
                                'Forecasted Treasury Yields': forecast[:, 7],
                                'Treasury Yields': test['Treasury Yields'].values
                                })
    
    print(predictions)
    if save: predictions.to_csv(f"{DATABASE_DIR}/{str(today)}/farooqi_predictions.csv", index=False)

    if plot:
        # Assuming df is your DataFrame
        variables = ['Real Gross Domestic Product (GDP)', 'Industrial Production', 'Nonfarm Payrolls', 'Unemployment Rate', 
                     'Consumer Price Index (CPI)', 'Producer Price Index (PPI)', 'Federal Funds Rate', 'Treasury Yields']

        # Create a new figure with subplots
        fig, axes = plt.subplots(nrows=len(variables), figsize=(12, 6 * len(variables)))

        # Plot each pair of columns
        for i, variable in enumerate(variables):
            forecasted_variable = f'Forecasted {variable}'
            axes[i].plot(predictions[forecasted_variable], label=forecasted_variable)
            axes[i].plot(predictions[variable], label=variable)
            axes[i].legend()
            axes[i].set_title(f'{variable} vs Forecasted {variable}')

        # Display the plot
        plt.tight_layout()
        plt.show()

    return predictions

def var_accuracy(predictions, fred_latest):
    """
    Calculates the accuracy of the predictions made by the VAR model.

    Parameters:
    predictions (pandas.DataFrame): A DataFrame containing the actual and forecasted values.

    Returns:
    float: The mean absolute percentage error of the predictions.
    """
    # Assuming df is your DataFrame
    variables = ['Real Gross Domestic Product (GDP)', 'Industrial Production', 'Nonfarm Payrolls', 'Unemployment Rate', 'Consumer Price Index (CPI)', 'Producer Price Index (PPI)', 'Federal Funds Rate', 'Treasury Yields']
    accuracy = {}

    print("""Mean Squared Error (MSE):
        Thinks about how far off your guesses are from the actual heights, in terms of distance squared.
        Larger the square of the distance (bigger guess difference), the more it penalizes your prediction.
        Like this: Imagine you guess someone is 5 feet tall, but they're actually 6 feet tall. The difference 
        is 1 foot. But MSE squares that difference (1 squared = 1), so it contributes more to the overall error.
          """)
    
    print("""Mean Absolute Error (MAE):
          Just cares about how far off your guesses are, regardless of direction (over or underestimation).
          It simply takes the absolute value (distance without considering positive or negative) of the difference between 
          your guess and the actual height. So, in the same example (guessing 5 feet for a 6 feet tall friend), 
          MAE would just consider the difference as 1 foot (absolute value of 1).\n""")

    # Calculate the MSE and MAE for each pair of columns
    for variable in variables:
        forecasted_variable = f'Forecasted {variable}'
        mse = mean_squared_error(predictions[variable], predictions[forecasted_variable])
        mae = mean_absolute_error(predictions[variable], predictions[forecasted_variable])
        accuracy[variable] = {'MSE': mse, 'MAE': mae}
        print(f'{variable}: MSE = {mse}, MAE = {mae}')

    # TODO: Calculate the mean absolute percentage error

    return accuracy

def normalization(type='zscore', save=True, data=None, rolling_window=None):
    """
    Normalize the economic data using:
    1. Min-Max Normalization on each time series (minmax)
    2. Z-Score Normalization on each time series (zscore)
    
    Saves file by default to DATABASE_DIR/today/ as minmax_indicators.csv or zscore_indicators.csv
    
    """
    if type == 'minmax':
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

        if save: data.to_csv(f"{DATABASE_DIR}/{str(today)}/minmax_indicators.csv", index=False)
        return data
    
    elif type == 'zscore':
        if rolling_window is None:
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = (data[column] - data[column].mean()) / data[column].std()

            if save: data.to_csv(f"{DATABASE_DIR}/{str(today)}/zscore_indicators.csv", index=False)
            return data
        
        else:
            # Take latest dates (column) in df by rolling window param
            data = data.tail(rolling_window).copy()
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = (data[column] - data[column].mean()) / data[column].std()

            if save: data.to_csv(f"{DATABASE_DIR}/{str(today)}/zscore_indicators_{str(rolling_window)}_days.csv", index=False)
            return data        




if __name__ == '__main__':
    """
    Resilient Macro-portfolio
    Positive:
        Economic (higher)
        1. Real Gross Domestic Product (GDP)
        2. Industrial Production

        Employment (higher)
        3. Nonfarm Payrolls
        4. Unemployment Rate
        
        Inflation (moderate, set a>=x>=b):
        5. Consumer Price Index (CPI)
        6. Producer Price Index (PPI)

        Interest rates (lower):
        7. Federal Funds Rate
        8. Treasury Yields
    """

    fred_historical, fred_latest = main()

    # Run VAR model prediction on fred historical data
    predictions = make_VAR_predictions(fred_historical, plot=False, save=False)
    accuracy = var_accuracy(predictions, fred_latest)
    print('\n') 
    print(fred_latest)

    # TODO: Assign weights to each variable based on importance
    weights = {
        "Real Gross Domestic Product (GDP)": 0.25,
        "Industrial Production": 0.15,
        "Nonfarm Payrolls": 0.20,
        "Unemployment Rate": -0.30,  # Negative weight for unemployment (lower is better)
        "Consumer Price Index (CPI)": -0.10,  # Negative weight for inflation (lower is better)
        "Producer Price Index (PPI)": -0.05,  # Negative weight for producer prices (lower is better)
        "Federal Funds Rate": 0.10,
        "Treasury Yields": 0.15
    }

    # Remove latest date from fred_latest (leave col and value only)
    clean_data = {key: value[1] for key, value in fred_latest.items()}
    data = pd.DataFrame(clean_data, index=[0])
    
    # Normalize the fred_historical data
    normalized_data = normalization(type='minmax', data=fred_historical)

    exit()