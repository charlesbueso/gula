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

    # Define the series IDs for the indicators
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

def kennedy_weight(data, weights):
    """
        Takes in multiple time series normalized by z-score in data
        It also takes weights for each variable in the data
        Returns the Kennedy Factor as a weighted sum of the normalized values
        Between -1 and 1
    """
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = (data[column] - data[column].mean()) / data[column].std()
        


    # Normalize each variable between -1 and 1 based on typical economic ranges
    # normalized_data = {key: (value - np.mean(value)) / np.std(value) for key, value in data.items()}
    # Calculate the weighted sum of normalized values
    # sentiment_index = sum([data[key] * weights[key] for key in weights])

    # return sentiment_index


def min_max_normalization(data):
    """
        Normalize the data using min-max normalization
        The data takes every column after index and normalizes it between 0 and 1
    """
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    # Save the normalized data to a csv file in the same directory
    data.to_csv(f"{DATABASE_DIR}/{str(today)}/minmax_indicators.csv", index=False)

    return data

def zscore_normalization(data, rolling_window=None):
    """
        Normalize the data using z-score normalization
        The data takes every column after index and normalizes it between -1 and 1
        Rolling windows chooses last x days to calculate the mean and standard deviation
    """
    if rolling_window is None:
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()

        # Save the normalized data to a csv file in the same directory
        data.to_csv(f"{DATABASE_DIR}/{str(today)}/zscore_indicators.csv", index=False)
        return data
    
    else:
        # Take latest dates (column ) in data df
        data = data.tail(rolling_window).copy()
        for column in data.select_dtypes(include=[np.number]).columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()

        data.to_csv(f"{DATABASE_DIR}/{str(today)}/zscore_indicators_{str(rolling_window)}_days.csv", index=False)
        return data


if __name__ == '__main__':
    fred_historical, fred_latest = main()
    # print(fred_historical.head())
    # print(fred_latest)
    
    # Assign weights to each variable based on importance
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

    # Remove latest date from dictionary
    clean_data = {key: value[1] for key, value in fred_latest.items()}
    data = pd.DataFrame(clean_data, index=[0])
    # print(data)
    # exit()

    # Min-Max Normalization
    # df = min_max_normalization(fred_historical)
    # print(df)

    df = zscore_normalization(fred_historical, rolling_window=23) #around 5 years of data

    # Calculate the Kennedy factor
    kennedy_weight = kennedy_weight(df, weights)
    print("Kennedy Factor: ---------------------------", str(kennedy_weight))
    exit()

    if kennedy_weight > 0:
        print(f"Economic Sentiment Index: {kennedy_weight:.2f} (Bullish)")
    elif kennedy_weight < 0:
        print(f"Economic Sentiment Index: {kennedy_weight:.2f} (Bearish)")
    else:
        print(f"Economic Sentiment Index: {kennedy_weight:.2f} (Neutral)")
    
            # Positive:
            # Economic (higher)
            # 1. Real Gross Domestic Product (GDP)
            # 2. Industrial Production
            # Employment (higher)
            # 3. Nonfarm Payrolls
            # 4. Unemployment Rate
            # Inflation (moderate, set a>=x>=b):
            # 5. Consumer Price Index (CPI)
            # 6. Producer Price Index (PPI)
            # Interest rates (lower):
            # 7. Federal Funds Rate
            # 8. Treasury Yields